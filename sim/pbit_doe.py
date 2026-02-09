import os
import json
import datetime
from pathlib import Path
import itertools
import heapq
from typing import Callable
import urllib.request
import ssl
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np
import pandas as pd

MPL_CACHE_DIR = Path(".mplconfig")
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE_DIR.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR.resolve())

if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = Path(".cache")
    xdg_cache.mkdir(exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache.resolve())
fontconfig_cache = Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig"
fontconfig_cache.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency
    certifi = None

TSPLIB_BASE_URL = "https://raw.githubusercontent.com/jvkersch/tsplib95/master/tsplib95/data/tsplib"
GSET_BASE_URL = "https://web.stanford.edu/~yyye/yyye/Gset"

def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)

# --- Quantization helper ---
def quantize_I(I, bits, I_min=-5.0, I_max=5.0):
    if bits <= 0:
        return I  # no quantization
    levels = 2 ** bits
    I_min, I_max = -5.0, 5.0
    I = np.clip(I, I_min, I_max)
    step = (I_max - I_min) / (levels - 1)
    return np.round((I - I_min) / step) * step + I_min

# --- Energy function ---
def energy(m, J, h):
    return -0.5 * m @ (J @ m) - h @ m

def average_energy_plot(energy_runs: list[list[float]], time_points: list[float], tag: str):
    # 空でない run だけを残す
    energy_runs = [run for run in energy_runs if len(run) > 0]

    if len(energy_runs) == 0:
        print(f"⚠ Warning: No valid energy data to plot for tag '{tag}' — skipping.")
        return

    # 最小長に揃える（必要なら）
    min_len = min(len(run) for run in energy_runs)
    energy_runs = [run[:min_len] for run in energy_runs]
    time_points = time_points[:min_len]

    energy_array = np.array(energy_runs)
    avg_energy = np.mean(energy_array, axis=0)
    std_energy = np.std(energy_array, axis=0)

    df = pd.DataFrame({"time_ns": time_points, "avg_energy": avg_energy, "std_energy": std_energy})
    Path("energy_logs").mkdir(exist_ok=True)
    df.to_csv(f"energy_logs/avg_energy_{tag}.csv", index=False)

    plt.figure()
    plt.plot(time_points, avg_energy, label="Average Energy", color="blue")
    plt.fill_between(time_points, avg_energy - std_energy, avg_energy + std_energy, color="blue", alpha=0.3)
    plt.xlabel("Time (ns)")
    plt.ylabel("Energy")
    plt.title(f"Energy vs Time - {tag}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"energy_logs/avg_energy_plot_{tag}.png")
    plt.close()

def write_energy_trace(energy_runs: list[list[float]], time_points: list[float], tag: str, out_dir: str):
    energy_runs = [run for run in energy_runs if len(run) > 0]
    if not energy_runs:
        print(f"⚠ Warning: No valid energy data to dump for tag '{tag}' — skipping.")
        return
    min_len = min(len(run) for run in energy_runs)
    energy_runs = [run[:min_len] for run in energy_runs]
    time_points = time_points[:min_len]

    data = {"time_ns": time_points}
    for i, run in enumerate(energy_runs):
        data[f"energy_r{i}"] = run
    arr = np.array(energy_runs)
    data["energy_mean"] = np.mean(arr, axis=0)
    data["energy_std"] = np.std(arr, axis=0)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    csv = out_path / f"energy_trace_{tag}.csv"
    df.to_csv(csv, index=False)
    print(f"✔ saved {csv}")

# --- Samplers ---
def tick_sampler_with_energy(J, h, beta_fn, tau_ns, sim_time_ns, dt_ns, rng, share, bits, I_min=-5.0, I_max=5.0, quantize=True, schedule_mode="random", track_c1=False, c1_skip_frac=0.2):
    steps = int(sim_time_ns / dt_ns)
    n = J.shape[0]
    m = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    p_flip = (dt_ns / tau_ns) / share
    energies = []
    c1_vals = [] if track_c1 else None
    prev_m_rec = None
    idx_order = np.arange(n, dtype=np.int64)
    ptr = 0

    for k in range(steps):
        if track_c1:
            prev_m = m.copy()
        beta = beta_fn(k * dt_ns)
        if schedule_mode in ("rr", "rr-jitter", "rr_jitter"):  # round-robin with optional random offset per step
            if schedule_mode != "rr":  # light shuffle: randomize start pointer each tick to break deterministic resonance
                ptr = (ptr + rng.integers(n)) % n
            updates = int(round(p_flip * n))
            if updates <= 0:
                updates = 1 if rng.random() < (p_flip * n) else 0
            flip_mask = np.zeros(n, dtype=bool)
            for _ in range(updates):
                flip_mask[idx_order[ptr]] = True
                ptr = (ptr + 1) % n
        elif schedule_mode in ("block", "block-random", "block_random", "banked"):  # banked/contiguous block selection
            updates = int(round(p_flip * n))
            if updates <= 0:
                updates = 1 if rng.random() < (p_flip * n) else 0
            start = int(rng.integers(n))
            flip_mask = np.zeros(n, dtype=bool)
            end = start + updates
            if end <= n:
                flip_mask[start:end] = True
            else:
                flip_mask[start:n] = True
                flip_mask[0:(end % n)] = True
        elif schedule_mode in ("block-random-stride", "block_random_stride", "block-stride"):  # contiguous generator with random stride (banked-friendly)
            updates = int(round(p_flip * n))
            if updates <= 0:
                updates = 1 if rng.random() < (p_flip * n) else 0
            start = int(rng.integers(n))
            stride = int(rng.integers(1, n))
            # ensure stride and n not sharing large factor to avoid tiny cycles
            import math
            tries = 0
            while math.gcd(stride, n) > 1 and tries < 5:
                stride = int(rng.integers(1, n))
                tries += 1
            flip_mask = np.zeros(n, dtype=bool)
            idx = start
            for _ in range(updates):
                flip_mask[idx] = True
                idx = (idx + stride) % n
        else:  # random independent
            flip_mask = rng.random(n) < p_flip
        I = beta * (h + J @ m)
        if quantize:
            I = quantize_I(I, bits, I_min=I_min, I_max=I_max)
        p_up = 0.5 * (1 + np.tanh(I))
        r = rng.random(n)
        m[(flip_mask) & (r < p_up)] = 1
        m[(flip_mask) & (r >= p_up)] = -1
        if track_c1:
            c1_vals.append(float(np.mean(prev_m * m)))
        energies.append(energy(m, J, h))

    times = np.linspace(0, sim_time_ns, steps)
    if track_c1:
        skip = int(len(c1_vals) * c1_skip_frac) if c1_skip_frac else 0
        c1_used = c1_vals[skip:] if skip < len(c1_vals) else []
        c1_mean = float(np.mean(c1_used)) if c1_used else float("nan")
        return m, times.tolist(), energies, c1_mean
    return m, times.tolist(), energies

def gillespie_sampler_with_energy(J, h, beta_fn, tau_ns, t_end_ns, rng, share, bits, I_min=-5.0, I_max=5.0, apply_delay_ns=None, quantize=True, track_c1=False, c1_skip_frac=0.2):
    n = J.shape[0]
    m = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    neigh = [np.nonzero(J[i])[0] for i in range(n)]
    lam_spin = (1.0 / tau_ns) / share
    delay_ns = tau_ns if apply_delay_ns is None else apply_delay_ns
    t = 0.0
    E_list = []
    times = []
    c1_vals = [] if track_c1 else None
    record_interval = 1.0
    next_record_t = 0.0
    pending = []  # (apply_time, idx, new_state)
    # next event time per spin
    event_heap = [(rng.exponential(1.0 / lam_spin), i) for i in range(n)]
    heapq.heapify(event_heap)

    while t < t_end_ns and event_heap:
        next_event_t = event_heap[0][0]
        next_commit_t = pending[0][0] if pending else float("inf")
        next_t = min(next_event_t, next_commit_t, t_end_ns)
        t = next_t

        # apply commits due now
        while pending and pending[0][0] <= t:
            _, idx, new_state = heapq.heappop(pending)
            m[idx] = new_state

        if next_event_t <= t and t < t_end_ns:
            # process all events scheduled at this time (allow exact ties)
            events_now = []
            while event_heap and event_heap[0][0] <= t:
                ev_t, idx = heapq.heappop(event_heap)
                events_now.append((ev_t, idx))
            for _, i in events_now:
                beta = beta_fn(t)
                I_i = beta * (h[i] + (J[i, neigh[i]] @ m[neigh[i]]))
                if quantize:
                    I_i = quantize_I(I_i, bits, I_min=I_min, I_max=I_max)
                p_up = 0.5 * (1 + np.tanh(I_i))
                new_state = 1 if rng.random() < p_up else -1
                heapq.heappush(pending, (t + delay_ns, i, new_state))
                heapq.heappush(event_heap, (t + rng.exponential(1.0 / lam_spin), i))

        while t >= next_record_t and next_record_t <= t_end_ns:
            E_list.append(energy(m, J, h))
            times.append(t)
            if track_c1:
                if prev_m_rec is not None:
                    c1_vals.append(float(np.mean(prev_m_rec * m)))
                prev_m_rec = m.copy()
            next_record_t += record_interval

    while t >= next_record_t and next_record_t <= t_end_ns:
        E_list.append(energy(m, J, h))
        times.append(t)
        if track_c1:
            if prev_m_rec is not None:
                c1_vals.append(float(np.mean(prev_m_rec * m)))
            prev_m_rec = m.copy()
        next_record_t += record_interval

    if track_c1:
        skip = int(len(c1_vals) * c1_skip_frac) if c1_skip_frac else 0
        c1_used = c1_vals[skip:] if skip < len(c1_vals) else []
        c1_mean = float(np.mean(c1_used)) if c1_used else float("nan")
        return m.astype(np.int8), times, E_list, c1_mean
    return m.astype(np.int8), times, E_list

# --- Network loader ---
def load_or_generate_ising(n_pbit, density, rng):
    J = np.zeros((n_pbit, n_pbit), dtype=np.float32)
    mask = rng.random((n_pbit, n_pbit)) < density
    vals = rng.choice([-1.0, 1.0], size=(n_pbit, n_pbit)) * mask
    J += np.triu(vals, k=1)
    J += J.T
    h = rng.choice([-1.0, 1.0], size=n_pbit).astype(np.float32) * 0.1
    return J, h

# --- Benchmark helpers ---
def ensure_remote_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"⬇ Downloading {url} -> {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pbit-doe/1.0"})
        context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
        with urllib.request.urlopen(req, context=context) as resp:
            data = resp.read()
    except Exception as exc:
        manual_hint = f"Please download it manually via:\n  curl -L {url} -o \"{dest}\"\nthen re-run the script.\n"
        raise RuntimeError(f"Failed to download benchmark from {url}: {exc}\n{manual_hint}") from exc
    dest.write_bytes(data)
    return dest

def parse_tsplib_file(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    coords = []
    dimension = None
    reading_coords = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("DIMENSION"):
            parts = line.split(":")
            if len(parts) == 2:
                dimension = int(parts[1])
            else:
                dimension = int(parts[-1].split()[-1])
        elif line.upper().startswith("NODE_COORD_SECTION"):
            reading_coords = True
            continue
        elif line.upper().startswith("EOF"):
            break
        elif reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    if dimension is None:
        dimension = len(coords)
    if len(coords) != dimension:
        raise ValueError(f"Malformed TSPLIB file {path}: expected {dimension} coords, got {len(coords)}")
    coords = np.array(coords, dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.rint(dist).astype(np.float32)

def build_tsp_qubo(distance: np.ndarray, penalty: float, weight: float) -> np.ndarray:
    n_city = distance.shape[0]
    n_var = n_city * n_city
    Q = np.zeros((n_var, n_var), dtype=np.float64)

    def vid(city, pos):
        return city * n_city + pos

    for city in range(n_city):
        idxs = [vid(city, pos) for pos in range(n_city)]
        for idx in idxs:
            Q[idx, idx] += -penalty
        for p in range(n_city):
            for q in range(p + 1, n_city):
                Q[idxs[p], idxs[q]] += 2 * penalty

    for pos in range(n_city):
        idxs = [vid(city, pos) for city in range(n_city)]
        for idx in idxs:
            Q[idx, idx] += -penalty
        for c1 in range(n_city):
            for c2 in range(c1 + 1, n_city):
                Q[idxs[c1], idxs[c2]] += 2 * penalty

    for pos in range(n_city):
        nxt = (pos + 1) % n_city
        for i in range(n_city):
            a = vid(i, pos)
            for j in range(n_city):
                if i == j:
                    continue
                b = vid(j, nxt)
                coeff = weight * float(distance[i, j])
                if a <= b:
                    Q[a, b] += coeff
                else:
                    Q[b, a] += coeff
    return Q

def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Q = np.triu(Q)
    Q = Q + np.triu(Q, 1).T
    n = Q.shape[0]
    linear = np.zeros(n, dtype=np.float64)
    quad = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        linear[i] += 0.5 * Q[i, i]
        for j in range(i + 1, n):
            q = Q[i, j]
            if q == 0.0:
                continue
            linear[i] += 0.25 * q
            linear[j] += 0.25 * q
            quad[i, j] += 0.25 * q
    J = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            beta = quad[i, j]
            if beta == 0.0:
                continue
            val = -2.0 * beta
            J[i, j] = J[j, i] = val
    h = (-linear).astype(np.float32)
    return J, h

def _resolve_local_file(path: Path, preferred_name: str | None = None) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        if preferred_name:
            candidate = path / preferred_name
            if candidate.is_file():
                return candidate
        files = [p for p in path.iterdir() if p.is_file()]
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(f"{path} is not a file. Please point to the benchmark file (not a directory).")

def build_tsplib_ising(instance: str, data_dir: Path, penalty: float, weight: float):
    name = Path(instance).stem
    if Path(instance).exists():
        tsp_path = _resolve_local_file(Path(instance))
    else:
        suffix = ".tsp" if not instance.lower().endswith(".tsp") else ""
        tsp_path = data_dir / f"{instance}{suffix}"
        url = f"{TSPLIB_BASE_URL}/{tsp_path.name}"
        tsp_path = ensure_remote_file(url, tsp_path)
    dist = parse_tsplib_file(tsp_path)
    Q = build_tsp_qubo(dist, penalty=penalty, weight=weight)
    J, h = qubo_to_ising(Q)
    return {"kind": "tsp", "label": name, "instance": name, "n": J.shape[0], "J": J, "h": h, "path": str(tsp_path), "distance": dist}

def parse_gset_graph(path: Path) -> np.ndarray:
    with path.open() as f:
        header = f.readline().strip().split()
        if len(header) < 2:
            raise ValueError(f"Malformed G-set file {path}")
        n = int(header[0])
        adj = np.zeros((n, n), dtype=np.float32)
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            w = float(parts[2]) if len(parts) > 2 else 1.0
            adj[i, j] += w
            adj[j, i] += w
    return adj

def build_maxcut_ising(adj: np.ndarray, weight_scale: float = 1.0):
    n = adj.shape[0]
    J = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w == 0.0:
                continue
            J[i, j] = J[j, i] = -1.0 * weight_scale * w
    h = np.zeros(n, dtype=np.float32)
    return J, h

def load_maxcut_instance(instance: str, data_dir: Path, weight_scale: float = 1.0):
    if Path(instance).exists():
        graph_path = _resolve_local_file(Path(instance))
    else:
        graph_path = data_dir / instance
        url = f"{GSET_BASE_URL}/{instance}"
        graph_path = ensure_remote_file(url, graph_path)
        if graph_path.is_dir():
            graph_path = _resolve_local_file(graph_path, preferred_name=f"{graph_path.name}.txt")
    adj = parse_gset_graph(graph_path)
    J, h = build_maxcut_ising(adj, weight_scale=weight_scale)
    name = graph_path.stem
    return {"kind": "maxcut", "label": name, "instance": name, "n": J.shape[0], "J": J, "h": h, "path": str(graph_path), "adjacency": adj}

def print_ising_couplings(spec: dict):
    J = spec["J"]
    h = spec["h"]
    print(f"\n[Ising couplings] kind={spec.get('kind')} label={spec.get('label')} n={spec.get('n')}")
    print("J matrix:")
    print(J)
    print("h vector:")
    print(h)

def compute_tsp_path_length(spins: np.ndarray, distance: np.ndarray) -> tuple[float, dict]:
    n_city = distance.shape[0]
    n_var = n_city * n_city
    if spins.shape[0] < n_var:
        return float("nan"), {"missing_vars": n_var - spins.shape[0]}
    bits = ((spins[:n_var] + 1) // 2).reshape(n_city, n_city)
    diag: dict = {"bad_rows": [], "bad_cols": [], "zero_columns": []}
    column_choices: list[list[int]] = []
    order = []
    for pos in range(n_city):
        column = bits[:, pos]
        ones = np.nonzero(column)[0].tolist()
        column_choices.append(ones)
        if len(ones) == 0:
            diag["zero_columns"].append(int(pos))
            diag["bad_cols"].append(int(pos))
            continue
        if len(ones) != 1:
            diag["bad_cols"].append(int(pos))
            continue
        order.append(ones[0])
    for city in range(n_city):
        if bits[city].sum() != 1:
            diag["bad_rows"].append(int(city))
    diag["raw_order"] = [choices[0] if len(choices) == 1 else (choices if choices else None) for choices in column_choices]
    if diag["bad_rows"] or len(order) != n_city:
        recovered_order = _recover_tsp_order(spins[:n_var], n_city)
        if recovered_order is None:
            return float("nan"), diag
        diag["recovered"] = True
        diag["recovered_order"] = recovered_order
        order = recovered_order
    length = 0.0
    for idx in range(n_city):
        a = order[idx]
        b = order[(idx + 1) % n_city]
        length += float(distance[a, b])
    return length, diag

def _recover_tsp_order(spin_slice: np.ndarray, n_city: int) -> list[int] | None:
    scores = spin_slice.reshape(n_city, n_city)
    remaining = set(range(n_city))
    order = []
    for pos in range(n_city):
        ranked = sorted(range(n_city), key=lambda c: (-scores[c, pos], c))
        chosen = None
        for city in ranked:
            if city in remaining:
                chosen = city
                break
        if chosen is None:
            if not remaining:
                return None
            chosen = min(remaining)
        remaining.remove(chosen)
        order.append(chosen)
    return order

class QuboBuilder:
    def __init__(self, n: int, aux_penalty: float):
        self.Q = np.zeros((n, n), dtype=np.float64)
        self.aux_penalty = aux_penalty
        self._and_cache: dict[tuple[int, int], int] = {}

    def ensure_size(self, size: int):
        if size <= self.Q.shape[0]:
            return
        old = self.Q
        self.Q = np.zeros((size, size), dtype=np.float64)
        self.Q[: old.shape[0], : old.shape[1]] = old

    def add_linear(self, idx: int, coeff: float):
        if coeff == 0.0:
            return
        self.ensure_size(idx + 1)
        self.Q[idx, idx] += coeff

    def add_quadratic(self, i: int, j: int, coeff: float):
        if coeff == 0.0:
            return
        self.ensure_size(max(i, j) + 1)
        if i == j:
            self.Q[i, i] += coeff
        else:
            self.Q[i, j] += coeff
            self.Q[j, i] += coeff

    def new_var(self) -> int:
        idx = self.Q.shape[0]
        self.ensure_size(idx + 1)
        return idx

    def get_and_var(self, i: int, j: int) -> int:
        key = tuple(sorted((i, j)))
        if key in self._and_cache:
            return self._and_cache[key]
        anc = self.new_var()
        self._and_cache[key] = anc
        self.add_and_constraint(i, j, anc)
        return anc

    def add_and_constraint(self, x: int, y: int, anc: int):
        pen = self.aux_penalty
        if pen <= 0:
            return
        self.add_linear(x, -pen)
        self.add_linear(y, -pen)
        self.add_linear(anc, 5 * pen)
        self.add_quadratic(x, y, 2 * pen)
        self.add_quadratic(anc, x, -3 * pen)
        self.add_quadratic(anc, y, -3 * pen)

def parse_dimacs_cnf(path: Path) -> tuple[int, list[list[int]]]:
    clauses: list[list[int]] = []
    num_vars = 0
    with path.open() as f:
        current: list[int] = []
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4:
                    num_vars = int(parts[2])
                continue
            tokens = line.split()
            for tok in tokens:
                if tok.startswith("%"):
                    break
                lit = int(tok)
                if lit == 0:
                    if current:
                        clauses.append(current)
                        current = []
                    break
                current.append(lit)
        if current:
            clauses.append(current)
    if num_vars == 0:
        num_vars = max(abs(l) for clause in clauses for l in clause)
    return num_vars, clauses

def multiply_poly(poly_a: dict[tuple[int, ...], float], poly_b: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    result: dict[tuple[int, ...], float] = {}
    for vars_a, coef_a in poly_a.items():
        for vars_b, coef_b in poly_b.items():
            combined = tuple(sorted(vars_a + vars_b))
            result[combined] = result.get(combined, 0.0) + coef_a * coef_b
    # prune zeros
    return {vars_tuple: coef for vars_tuple, coef in result.items() if abs(coef) > 1e-12}

def add_monomial(builder: QuboBuilder, coeff: float, vars_tuple: tuple[int, ...]):
    if abs(coeff) < 1e-12:
        return
    vars_list = list(vars_tuple)
    while len(vars_list) > 2:
        anc = builder.get_and_var(vars_list[0], vars_list[1])
        vars_list = [anc] + vars_list[2:]
    if len(vars_list) == 1:
        builder.add_linear(vars_list[0], coeff)
    elif len(vars_list) == 2:
        builder.add_quadratic(vars_list[0], vars_list[1], coeff)

def build_sat_qubo(num_vars: int, clauses: list[list[int]], clause_penalty: float, aux_penalty: float) -> np.ndarray:
    builder = QuboBuilder(num_vars, aux_penalty)
    for clause in clauses:
        poly = {(): 1.0}
        for lit in clause:
            var_idx = abs(lit) - 1
            if lit > 0:
                lit_poly = {(): 1.0, (var_idx,): -1.0}
            else:
                lit_poly = {(var_idx,): 1.0}
            poly = multiply_poly(poly, lit_poly)
        for vars_tuple, coeff in poly.items():
            if not vars_tuple:
                continue  # constant offset
            add_monomial(builder, clause_penalty * coeff, vars_tuple)
    return builder.Q
def compute_maxcut_value(spins: np.ndarray, adjacency: np.ndarray) -> float:
    n = adjacency.shape[0]
    if spins.shape[0] < n:
        return float("nan")
    s = np.where(spins[:n] >= 0, 1, -1)
    diff = s[:, None] != s[None, :]
    return float(np.sum(np.triu(adjacency * diff, 1)))

def build_satlib_ising(instance: str, data_dir: Path, clause_penalty: float, aux_penalty: float):
    inst_path = Path(instance)
    if not inst_path.exists():
        inst_path = data_dir / instance
    if not inst_path.exists():
        raise FileNotFoundError(f"SAT instance not found: {instance}")
    num_vars, clauses = parse_dimacs_cnf(inst_path)
    Q = build_sat_qubo(num_vars, clauses, clause_penalty, aux_penalty)
    J, h = qubo_to_ising(Q)
    label = inst_path.stem
    return {"kind": "sat", "label": label, "instance": str(inst_path), "n": J.shape[0], "J": J, "h": h, "clauses": len(clauses), "path": str(inst_path)}

# --- Main sweep ---

def make_beta_fn(schedule: str, beta0: float, beta1: float, t_end: float, steps: int = 1) -> Callable[[float], float]:
    if schedule == "const":
        return lambda t: beta0
    if schedule == "exp":
        log0, log1 = np.log(beta0), np.log(beta1)
        return lambda t: float(np.exp(log0 + (log1 - log0) * (t / t_end)))
    if schedule == "linear":
        return lambda t: float(beta0 + (beta1 - beta0) * (t / t_end))
    if schedule == "step":
        step_len = t_end / steps
        return lambda t: beta0 if t < step_len else beta1
    raise ValueError("unknown schedule")

def run_single_configuration(bits, share, n_pbit, tau, *, sampler, schedule, beta0, beta1, steps, sim_time_ns, dt_ns, repeats, I_min, I_max, problem_spec, rng_seed, density, log_invalid=False, dump_spins=False, quantize=True, apply_delay_ns=None, tick_mode="random", track_c1=False, c1_skip_frac=0.2):
    beta_fn = make_beta_fn(schedule, beta0, beta1, t_end=sim_time_ns, steps=steps)
    if problem_spec is None:
        problem_spec = {"kind": "random", "label": "random", "density": density}
    metric_key = None
    metric_data = None
    metric_diag_tpl = None
    if problem_spec["kind"] == "tsp":
        metric_key = "tsp_path_length"
        metric_data = problem_spec.get("distance")
        metric_diag_tpl = {
            "invalid_runs": "tsp_invalid_run_ratio",
            "bad_cols": "tsp_bad_cols_avg",
            "bad_rows": "tsp_bad_rows_avg",
            "missing_vars": "tsp_missing_vars_avg",
            "zero_cols": "tsp_zero_cols_avg",
        }
    elif problem_spec["kind"] == "maxcut":
        metric_key = "maxcut_cut_value"
        metric_data = problem_spec.get("adjacency")

    rng = np.random.default_rng(rng_seed)
    if problem_spec["kind"] == "random":
        J, h = load_or_generate_ising(n_pbit, density=density, rng=rng)
        instance_label = problem_spec.get("label", "random")
    else:
        if n_pbit != problem_spec["n"]:
            raise ValueError(f"Problem size ({problem_spec['n']}) does not match DOE sweep ({n_pbit})")
        J = problem_spec["J"]
        h = problem_spec["h"]
        instance_label = problem_spec["label"]
    instance_name = problem_spec.get("instance", instance_label)

    energy_runs = []
    time_ref = []
    c1_runs = [] if track_c1 else None
    agg = {"energy": 0.0, "energy_ratio": 0.0, "power_mw": 0.0}
    metric_runs = []
    diag_accum = (
        {"invalid_runs": 0.0, "bad_cols": 0.0, "bad_rows": 0.0, "missing_vars": 0.0, "zero_cols": 0.0}
        if metric_key == "tsp_path_length"
        else None
    )
    diag_log_budget = 3 if (metric_key == "tsp_path_length" and log_invalid) else 0

    t0 = time.perf_counter()
    for r in range(repeats):
        if sampler == "tick":
            if track_c1:
                m_final, times, E_list, c1_mean = tick_sampler_with_energy(
                    J, h, beta_fn, tau, sim_time_ns, dt_ns, rng, share, bits,
                    I_min=I_min, I_max=I_max, quantize=quantize, schedule_mode=tick_mode,
                    track_c1=True, c1_skip_frac=c1_skip_frac,
                )
                c1_runs.append(c1_mean)
            else:
                m_final, times, E_list = tick_sampler_with_energy(
                    J, h, beta_fn, tau, sim_time_ns, dt_ns, rng, share, bits,
                    I_min=I_min, I_max=I_max, quantize=quantize, schedule_mode=tick_mode,
                )
        elif sampler == "gillespie":
            if track_c1:
                m_final, times, E_list, c1_mean = gillespie_sampler_with_energy(
                    J, h, beta_fn, tau, sim_time_ns, rng, share, bits,
                    I_min=I_min, I_max=I_max, quantize=quantize, apply_delay_ns=apply_delay_ns,
                    track_c1=True, c1_skip_frac=c1_skip_frac,
                )
                c1_runs.append(c1_mean)
            else:
                m_final, times, E_list = gillespie_sampler_with_energy(
                    J, h, beta_fn, tau, sim_time_ns, rng, share, bits,
                    I_min=I_min, I_max=I_max, quantize=quantize, apply_delay_ns=apply_delay_ns,
                )
        else:
            raise ValueError("Unknown sampler")

        energy_runs.append(E_list)
        time_ref = times
        E_raw = energy(m_final, J, h)
        E_ratio = abs(E_raw) / n_pbit
        dac_power_mw = 1.0 * bits / 6 * (1 / share)
        core_pw_mw = 0.002 * n_pbit
        agg["energy"] += E_raw
        agg["energy_ratio"] += E_ratio
        agg["power_mw"] += (dac_power_mw + core_pw_mw)
        if metric_key == "tsp_path_length" and metric_data is not None:
            path_len, diag = compute_tsp_path_length(m_final, metric_data)
            if np.isfinite(path_len):
                metric_runs.append(path_len)
            else:
                recovered = diag.get("recovered")
                if diag_accum is not None:
                    diag_accum["invalid_runs"] += 1
                    diag_accum["bad_cols"] += len(diag.get("bad_cols", []))
                    diag_accum["bad_rows"] += len(diag.get("bad_rows", []))
                    diag_accum["missing_vars"] += diag.get("missing_vars", 0)
                    diag_accum["zero_cols"] += len(diag.get("zero_columns", []))
                    if recovered:
                        diag_accum["invalid_runs"] -= 1  # recovered counts as valid for ratio
                if diag_log_budget > 0:
                    note = " (recovered)" if diag.get("recovered") else ""
                    print(f"    ⚠ Invalid TSP tour{note}: cols={diag.get('bad_cols', [])} rows={diag.get('bad_rows', [])}")
                    if "raw_order" in diag:
                        print(f"       raw_order: {diag['raw_order']}")
                    if "recovered_order" in diag:
                        print(f"       recovered: {diag['recovered_order']}")
                    diag_log_budget -= 1
        elif metric_key == "maxcut_cut_value" and metric_data is not None:
            metric_runs.append(compute_maxcut_value(m_final, metric_data))

    for k in agg:
        agg[k] /= repeats
    wall_s = time.perf_counter() - t0
    wall_per_repeat_s = wall_s / repeats if repeats > 0 else float("nan")

    if dump_spins:
        print(f"Final spins (n={len(m_final)}): {m_final.tolist()}")
        if problem_spec["kind"] == "tsp":
            n_city = problem_spec["distance"].shape[0]
            mat = ((m_final[: n_city * n_city] + 1) // 2).reshape(n_city, n_city)
            print(f"TSP row sums (per city): {mat.sum(axis=1).tolist()}")
            print(f"TSP column sums (per tour position): {mat.sum(axis=0).tolist()}")
    record = {
        "bits": bits,
        "share": share,
        "n_pbit": n_pbit,
        "tau_ns": tau,
        "sim_time_ns": sim_time_ns,
        "dt_ns": dt_ns,
        "wall_time_s": wall_s,
        "wall_time_per_repeat_s": wall_per_repeat_s,
        "E_k": float(n_pbit * (dt_ns / tau) / share) if sampler == "tick" else float("nan"),
        "E_k_norm": float((dt_ns / tau) / share) if sampler == "tick" else float("nan"),
        "tick_mode": tick_mode if sampler == "tick" else None,
        "sampler": sampler,
        "schedule": schedule,
        "beta0": beta0,
        "beta1": beta1,
        "steps": steps,
        "repeats": repeats,
        "problem": problem_spec["kind"],
        "instance": instance_name,
        "label": instance_label,
        "Imin": I_min,
        "Imax": I_max,
        **agg,
    }
    if track_c1:
        record["c1_mean"] = float(np.nanmean(c1_runs)) if c1_runs else float("nan")
        record["c1_std"] = float(np.nanstd(c1_runs)) if c1_runs else float("nan")
    if metric_key == "tsp_path_length":
        record["tsp_path_length"] = float(np.nanmean(metric_runs)) if metric_runs else float("nan")
        if diag_accum is not None and metric_diag_tpl is not None:
            record[metric_diag_tpl["invalid_runs"]] = diag_accum["invalid_runs"] / repeats
            record[metric_diag_tpl["bad_cols"]] = diag_accum["bad_cols"] / repeats
            record[metric_diag_tpl["bad_rows"]] = diag_accum["bad_rows"] / repeats
            record[metric_diag_tpl["missing_vars"]] = diag_accum["missing_vars"] / repeats
            record[metric_diag_tpl["zero_cols"]] = diag_accum["zero_cols"] / repeats
    elif metric_key == "maxcut_cut_value" and metric_runs:
        record[metric_key] = float(np.nanmean(metric_runs))
    elif metric_key == "maxcut_cut_value":
        record[metric_key] = float("nan")
    return record, energy_runs, time_ref

def _evaluate_config_worker(payload):
    config, shared = payload
    record, _, _ = run_single_configuration(
        config["bits"], config["share"], config["n_pbit"], config["tau_ns"],
        sampler=shared["sampler"], schedule=shared["schedule"],
        beta0=config.get("beta0", shared["beta0"]), beta1=config.get("beta1", shared["beta1"]), steps=shared["steps"],
        sim_time_ns=shared["sim_time_ns"], dt_ns=shared["dt_ns"], repeats=shared["repeats"],
        I_min=config.get("Imin", shared["Imin"]), I_max=config.get("Imax", shared["Imax"]),
        problem_spec=shared["problem_spec"], rng_seed=config["seed"], density=shared["density"], log_invalid=False,
        dump_spins=shared.get("dump_spins", False), quantize=shared.get("quantize", True), apply_delay_ns=shared.get("apply_delay_ns")
    )
    record["iteration"] = config["iteration"]
    return record

def _sample_config(rng, iteration, bits_list, share_list, n_pbit_list, tau_ns_list,
                   Imin_range, Imax_range, beta0_range, beta1_range,
                   Imin_default, Imax_default, beta0_default, beta1_default, min_gap):
    bits = rng.choice(bits_list)
    share = rng.choice(share_list)
    n_pbit = rng.choice(n_pbit_list)
    tau = rng.choice(tau_ns_list)
    if Imin_range:
        imin = rng.uniform(Imin_range[0], Imin_range[1])
    else:
        imin = Imin_default
    if Imax_range:
        max_low = Imax_range[0]
        if Imin_range:
            max_low = max(max_low, imin + min_gap)
        if max_low >= Imax_range[1]:
            max_low = Imax_range[1]
        imax = rng.uniform(max_low, Imax_range[1])
    else:
        imax = Imax_default
    if beta0_range:
        beta0 = rng.uniform(beta0_range[0], beta0_range[1])
    else:
        beta0 = beta0_default
    if beta1_range:
        beta1 = rng.uniform(beta1_range[0], beta1_range[1])
    else:
        beta1 = beta1_default
    return {
        "iteration": iteration,
        "bits": bits,
        "share": share,
        "n_pbit": n_pbit,
        "tau_ns": tau,
        "Imin": imin,
        "Imax": imax,
        "beta0": beta0,
        "beta1": beta1,
        "seed": rng.integers(0, 2**32 - 1, dtype=np.uint64).item()
    }

def _mutate_config(rng, config, bits_list, share_list, n_pbit_list, tau_ns_list,
                   Imin_range, Imax_range, beta0_range, beta1_range,
                   Imin_default, Imax_default, beta0_default, beta1_default, min_gap, scale=0.1):
    new_cfg = dict(config)
    if rng.random() < 0.5:
        new_cfg["bits"] = int(rng.choice(bits_list))
    if rng.random() < 0.5:
        new_cfg["share"] = float(rng.choice(share_list))
    if rng.random() < 0.5:
        new_cfg["n_pbit"] = int(rng.choice(n_pbit_list))
    if rng.random() < 0.5:
        new_cfg["tau_ns"] = float(rng.choice(tau_ns_list))
    if Imin_range:
        span = Imin_range[1] - Imin_range[0]
        new_cfg["Imin"] = float(np.clip(new_cfg["Imin"] + rng.normal(0, scale * span), Imin_range[0], Imin_range[1]))
    else:
        new_cfg["Imin"] = Imin_default
    if Imax_range:
        span = Imax_range[1] - Imax_range[0]
        lower = max(Imax_range[0], new_cfg["Imin"] + min_gap)
        new_cfg["Imax"] = float(np.clip(new_cfg["Imax"] + rng.normal(0, scale * span), lower, Imax_range[1]))
    else:
        new_cfg["Imax"] = Imax_default
    if beta0_range:
        span = beta0_range[1] - beta0_range[0]
        new_cfg["beta0"] = float(np.clip(new_cfg["beta0"] + rng.normal(0, scale * span), beta0_range[0], beta0_range[1]))
    else:
        new_cfg["beta0"] = beta0_default
    if beta1_range:
        span = beta1_range[1] - beta1_range[0]
        new_cfg["beta1"] = float(np.clip(new_cfg["beta1"] + rng.normal(0, scale * span), beta1_range[0], beta1_range[1]))
    else:
        new_cfg["beta1"] = beta1_default
    return new_cfg

def _plot_optimizer_summary(df: pd.DataFrame, run_tag: str):
    if df.empty:
        return
    Path("results").mkdir(exist_ok=True)
    if {"tau_ns", "energy"}.issubset(df.columns):
        plt.figure()
        for bits in sorted(df["bits"].unique()):
            subset = df[df["bits"] == bits]
            plt.plot(subset["tau_ns"], subset["energy"], marker="o", label=f"bits={bits}")
        plt.xlabel("tau_ns")
        plt.ylabel("energy")
        plt.title(f"Energy vs tau_ns - {run_tag}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{run_tag}_energy_vs_tau.png")
        plt.close()
    if {"bits", "energy"}.issubset(df.columns):
        plt.figure()
        for tau in sorted(df["tau_ns"].unique()):
            subset = df[df["tau_ns"] == tau]
            plt.plot(subset["bits"], subset["energy"], marker="o", label=f"tau={tau}")
        plt.xlabel("bits")
        plt.ylabel("energy")
        plt.title(f"Energy vs bits - {run_tag}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{run_tag}_energy_vs_bits.png")
        plt.close()

def _save_best_config(best: dict | None, best_out: str | Path | None):
    if best is None or best_out is None:
        return
    p = Path(best_out)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(val):
        if isinstance(val, (np.floating, np.integer)):
            return val.item()
        return val

    serializable = {k: _coerce(v) for k, v in best.items()}
    p.write_text(json.dumps(serializable, indent=2))
    print(f"✔ best configuration saved to {p}")

def _build_run_tag(problem_spec, sampler, schedule, optimizer=None):
    problem = problem_spec["kind"] if problem_spec else "random"
    instance = sanitize_label(problem_spec.get("instance", problem)) if problem_spec else "random"
    tag = f"{problem}_{instance}_{sampler}_{schedule}"
    if optimizer:
        tag += f"_{optimizer}"
    return tag

def _infer_beta_range(problem_spec, I_min, I_max, margin=0.2):
    """Infer beta ranges from coupling statistics.

    Uses sigma = sqrt((n-1) * Var(J_i,:)) averaged over rows (as in prior paper),
    then sets beta0=0.1/sigma, beta1=10/sigma, each widened by ±margin.
    Falls back to None if stats are degenerate.
    """
    if not problem_spec or "J" not in problem_spec:
        return None
    J = problem_spec["J"]
    if J is None:
        return None
    n = J.shape[0]
    if n <= 1:
        return None
    row_var = np.var(J, axis=1)
    sigma_rows = np.sqrt((n - 1) * row_var)
    sigma = float(np.mean(sigma_rows))
    if sigma <= 1e-9:
        return None
    base_beta0 = 0.1 / sigma
    base_beta1 = 10.0 / sigma
    beta0_lo = max(base_beta0 * (1.0 - margin), 1e-6)
    beta0_hi = base_beta0 * (1.0 + margin)
    beta1_lo = max(base_beta1 * (1.0 - margin), beta0_hi)  # ensure beta1 >= beta0_hi
    beta1_hi = base_beta1 * (1.0 + margin)
    return (beta0_lo, beta0_hi), (beta1_lo, beta1_hi)

def _infer_beta_fixed(problem_spec):
    """Return fixed (beta0, beta1) using sigma-based heuristic."""
    if not problem_spec or "J" not in problem_spec:
        return None
    J = problem_spec["J"]
    if J is None:
        return None
    n = J.shape[0]
    if n <= 1:
        return None
    row_var = np.var(J, axis=1)
    sigma_rows = np.sqrt((n - 1) * row_var)
    sigma = float(np.mean(sigma_rows))
    if sigma <= 1e-9:
        return None
    beta0 = 0.1 / sigma
    beta1 = 10.0 / sigma
    return beta0, beta1

def simulate_and_plot_list(bits_list, share_list, n_pbit_list, tau_ns_list, *, sampler: str, sim_time_ns=5000.0, dt_ns=1.0, repeats=5, schedule="const", beta0=1.0, beta1=1.0, steps=1, I_min=-5.0, I_max=5.0, problem_spec=None, rng_seed=0, density=0.1, dump_spins=False, quantize=True, apply_delay_ns=None, tick_mode="random", dump_trace=False, trace_dir=None, track_c1=False, c1_skip_frac=0.2):
    total = len(bits_list) * len(share_list) * len(n_pbit_list) * len(tau_ns_list)
    cnt = 0
    records = []
    for bits, share, n_pbit, tau in itertools.product(bits_list, share_list, n_pbit_list, tau_ns_list):
        cnt += 1
        record, energy_runs, time_ref = run_single_configuration(
            bits, share, n_pbit, tau,
            sampler=sampler, schedule=schedule,
            beta0=beta0, beta1=beta1, steps=steps,
            sim_time_ns=sim_time_ns, dt_ns=dt_ns,
            repeats=repeats,
            I_min=I_min, I_max=I_max,
            problem_spec=problem_spec, rng_seed=rng_seed,
            density=density, log_invalid=True, dump_spins=dump_spins, quantize=quantize,
            apply_delay_ns=apply_delay_ns, tick_mode=tick_mode, track_c1=track_c1, c1_skip_frac=c1_skip_frac,
        )
        instance_label = record["label"]
        print(f"[{cnt:>3}/{total}] bits={bits} share=1:{share} n={n_pbit} τ={tau}ns problem={record['problem']}:{record['instance']}")
        records.append(record)

        tag = f"{sanitize_label(instance_label)}_{n_pbit}pbit_{tau}tau_{sampler}_b{bits}_s{share}"
        average_energy_plot(energy_runs, time_ref, tag)
        if dump_trace and trace_dir:
            write_energy_trace(energy_runs, time_ref, tag, trace_dir)

    df = pd.DataFrame(records)
    today = datetime.date.today().strftime("%Y%m%d")
    Path("results").mkdir(exist_ok=True)
    run_tag = _build_run_tag(problem_spec or {"kind": "random", "instance": "random"}, sampler, schedule)
    # Include optional label and tick mode in the run tag so filenames stay unique per variant
    label = sanitize_label(problem_spec.get("label")) if problem_spec and problem_spec.get("label") else None
    if label and label not in run_tag:
        run_tag += f"_{label}"
    if sampler == "tick" and tick_mode:
        run_tag += f"_mode-{tick_mode}"
    if len(tau_ns_list) == 1:
        run_tag += f"_tau{tau_ns_list[0]}"
    if len(share_list) == 1:
        run_tag += f"_share{share_list[0]}"
    csv = Path("results") / f"{today}-doe-{run_tag}.csv"
    df.to_csv(csv, index=False)
    print(f"\n✔ saved {csv}\n")
    print(df.nsmallest(10, "energy").to_string(index=False))

def _init_shared_optimizer_context(sampler, schedule, steps, sim_time_ns, dt_ns, repeats,
                                   problem_spec, density, Imin_default, Imax_default,
                                   beta0_default, beta1_default, dump_spins, quantize, apply_delay_ns=None):
    return {
        "sampler": sampler,
        "schedule": schedule,
        "beta0": beta0_default,
        "beta1": beta1_default,
        "steps": steps,
        "sim_time_ns": sim_time_ns,
        "dt_ns": dt_ns,
        "repeats": repeats,
        "problem_spec": problem_spec,
        "density": density,
        "Imin": Imin_default,
        "Imax": Imax_default,
        "dump_spins": dump_spins,
        "quantize": quantize,
        "apply_delay_ns": apply_delay_ns,
    }

def _run_random_sampling(configs, shared, jobs, bits_list, share_list, n_pbit_list, tau_ns_list,
                         Imin_range, Imax_range, beta0_range, beta1_range,
                         Imin_default, Imax_default, beta0_default, beta1_default,
                         min_gap, rng):
    records = []
    best = None
    def handle_record(record):
        nonlocal best
        records.append(record)
        if best is None or record["energy"] < best["energy"]:
            best = record
            print(f"⭐ New best at iter {record['iteration']}: energy={record['energy']:.3f}, bits={record['bits']}, "
                  f"share={record['share']}, tau={record['tau_ns']}, I=({record['Imin']:.2f},{record['Imax']:.2f}), "
                  f"beta=({record['beta0']:.2f}->{record['beta1']:.2f})")
    if jobs <= 1:
        for cfg in configs:
            record = _evaluate_config_worker((cfg, shared))
            handle_record(record)
    else:
        print(f"Running search with {jobs} parallel workers ...")
        try:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(_evaluate_config_worker, (cfg, shared)) for cfg in configs]
                for future in as_completed(futures):
                    record = future.result()
                    handle_record(record)
        except (PermissionError, OSError) as exc:
            print(f"⚠ Parallel execution unavailable ({exc}); falling back to single-core.")
            for cfg in configs:
                record = _evaluate_config_worker((cfg, shared))
                handle_record(record)
    return records, best

def random_optimize_parameters(bits_list, share_list, n_pbit_list, tau_ns_list, *,
                               sampler, schedule, steps, sim_time_ns, dt_ns, repeats,
                               problem_spec, rng_seed, density, opt_iters,
                               Imin_default, Imax_default, beta0_default, beta1_default,
                               Imin_range=None, Imax_range=None, beta0_range=None, beta1_range=None,
                               min_gap=0.5, jobs=1, dump_spins=False, best_out=None, quantize=True, apply_delay_ns=None):
    rng = np.random.default_rng(rng_seed)
    configs = [
        _sample_config(rng, it, bits_list, share_list, n_pbit_list, tau_ns_list,
                       Imin_range, Imax_range, beta0_range, beta1_range,
                       Imin_default, Imax_default, beta0_default, beta1_default, min_gap)
        for it in range(1, opt_iters + 1)
    ]
    shared = _init_shared_optimizer_context(sampler, schedule, steps, sim_time_ns, dt_ns, repeats,
                                            problem_spec, density, Imin_default, Imax_default,
                                            beta0_default, beta1_default, dump_spins=dump_spins, quantize=quantize, apply_delay_ns=apply_delay_ns)
    records, best = _run_random_sampling(configs, shared, jobs, bits_list, share_list, n_pbit_list, tau_ns_list,
                                         Imin_range, Imax_range, beta0_range, beta1_range,
                                         Imin_default, Imax_default, beta0_default, beta1_default, min_gap, rng)
    records.sort(key=lambda r: r["iteration"])
    df = pd.DataFrame(records)
    today = datetime.date.today().strftime("%Y%m%d")
    Path("results").mkdir(exist_ok=True)
    run_tag = _build_run_tag(shared["problem_spec"], shared["sampler"], shared["schedule"], optimizer="random")
    csv = Path("results") / f"{today}-autoopt-{run_tag}.csv"
    df.to_csv(csv, index=False)
    _plot_optimizer_summary(df, run_tag)
    print(f"\n✔ optimization log saved to {csv}")
    if best:
        print("\nBest configuration:")
        print(pd.Series(best))
    _save_best_config(best, best_out)
    return best

def simulated_annealing_parameters(bits_list, share_list, n_pbit_list, tau_ns_list, *,
                                   sampler, schedule, steps, sim_time_ns, dt_ns, repeats,
                                   problem_spec, rng_seed, density, opt_iters,
                                   Imin_default, Imax_default, beta0_default, beta1_default,
                                   Imin_range=None, Imax_range=None, beta0_range=None, beta1_range=None,
                                   min_gap=0.5, temp_start=1.0, temp_end=0.01, dump_spins=False, best_out=None, quantize=True, apply_delay_ns=None):
    rng = np.random.default_rng(rng_seed)
    shared = _init_shared_optimizer_context(sampler, schedule, steps, sim_time_ns, dt_ns, repeats,
                                            problem_spec, density, Imin_default, Imax_default,
                                            beta0_default, beta1_default, dump_spins=dump_spins, quantize=quantize, apply_delay_ns=apply_delay_ns)
    configs = []
    current = _sample_config(rng, 1, bits_list, share_list, n_pbit_list, tau_ns_list,
                             Imin_range, Imax_range, beta0_range, beta1_range,
                             Imin_default, Imax_default, beta0_default, beta1_default, min_gap)
    current_record = _evaluate_config_worker((current, shared))
    best = current_record
    records = [current_record]
    cut_val = current_record.get("maxcut_cut_value", None)
    cut_str = f", cut={cut_val:.3f}" if cut_val is not None else ""
    print(f"[iter {current_record['iteration']:>3}] energy={current_record['energy']:.3f}{cut_str} "
          f"beta=({current_record['beta0']:.2f}->{current_record['beta1']:.2f}) "
          f"bits={current_record['bits']} share={current_record['share']} tau={current_record['tau_ns']}")
    print(f"⭐ New best at iter {current_record['iteration']}: energy={current_record['energy']:.3f}, bits={current_record['bits']}, "
          f"share={current_record['share']}, tau={current_record['tau_ns']}, I=({current_record['Imin']:.2f},{current_record['Imax']:.2f}), "
          f"beta=({current_record['beta0']:.2f}->{current_record['beta1']:.2f})")
    for it in range(2, opt_iters + 1):
        t_frac = (it - 2) / max(1, opt_iters - 2)
        temp = temp_start + (temp_end - temp_start) * t_frac
        proposal = _mutate_config(rng, current, bits_list, share_list, n_pbit_list, tau_ns_list,
                                  Imin_range, Imax_range, beta0_range, beta1_range,
                                  Imin_default, Imax_default, beta0_default, beta1_default, min_gap)
        proposal["iteration"] = it
        proposal_record = _evaluate_config_worker((proposal, shared))
        delta = proposal_record["energy"] - current_record["energy"]
        accept = delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-6))
        if accept:
            current = proposal
            current_record = proposal_record
        records.append(proposal_record)
        cut_val = proposal_record.get("maxcut_cut_value", None)
        cut_str = f", cut={cut_val:.3f}" if cut_val is not None else ""
        print(f"[iter {proposal_record['iteration']:>3}] energy={proposal_record['energy']:.3f}{cut_str} "
              f"beta=({proposal_record['beta0']:.2f}->{proposal_record['beta1']:.2f}) "
              f"bits={proposal_record['bits']} share={proposal_record['share']} tau={proposal_record['tau_ns']} "
              f"accept={'Y' if accept else 'N'} temp={temp:.4f}")
        if best is None or proposal_record["energy"] < best["energy"]:
            best = proposal_record
            print(f"⭐ New best at iter {proposal_record['iteration']}: energy={proposal_record['energy']:.3f}, "
                  f"bits={proposal_record['bits']}, share={proposal_record['share']}, tau={proposal_record['tau_ns']}, "
                  f"I=({proposal_record['Imin']:.2f},{proposal_record['Imax']:.2f}), "
                  f"beta=({proposal_record['beta0']:.2f}->{proposal_record['beta1']:.2f})")
    df = pd.DataFrame(records)
    today = datetime.date.today().strftime("%Y%m%d")
    Path("results").mkdir(exist_ok=True)
    run_tag = _build_run_tag(shared["problem_spec"], shared["sampler"], shared["schedule"], optimizer="anneal")
    csv = Path("results") / f"{today}-autoopt-{run_tag}.csv"
    df.to_csv(csv, index=False)
    _plot_optimizer_summary(df, run_tag)
    print(f"\n✔ optimization log saved to {csv}")
    if best is not None:
        print("\nBest configuration:")
        print(pd.Series(best))
    _save_best_config(best, best_out)
    return best

def evolutionary_optimize_parameters(bits_list, share_list, n_pbit_list, tau_ns_list, *,
                                     sampler, schedule, steps, sim_time_ns, dt_ns, repeats,
                                     problem_spec, rng_seed, density, opt_iters,
                                     Imin_default, Imax_default, beta0_default, beta1_default,
                                     Imin_range=None, Imax_range=None, beta0_range=None, beta1_range=None,
                                     min_gap=0.5, pop_size=10, elite_frac=0.4, dump_spins=False, best_out=None, quantize=True, apply_delay_ns=None):
    rng = np.random.default_rng(rng_seed)
    shared = _init_shared_optimizer_context(sampler, schedule, steps, sim_time_ns, dt_ns, repeats,
                                            problem_spec, density, Imin_default, Imax_default,
                                            beta0_default, beta1_default, dump_spins=dump_spins, quantize=quantize)
    population = [
        _sample_config(rng, i + 1, bits_list, share_list, n_pbit_list, tau_ns_list,
                       Imin_range, Imax_range, beta0_range, beta1_range,
                       Imin_default, Imax_default, beta0_default, beta1_default, min_gap)
        for i in range(pop_size)
    ]
    records = []
    best = None
    iteration = 0
    for gen in range(opt_iters):
        evaluated = []
        for cfg in population:
            iteration += 1
            cfg["iteration"] = iteration
            record = _evaluate_config_worker((cfg, shared))
            evaluated.append((cfg, record))
            records.append(record)
            if best is None or record["energy"] < best["energy"]:
                best = record
                print(f"⭐ New best at iter {record['iteration']}: energy={record['energy']:.3f}, bits={record['bits']}, "
                      f"share={record['share']}, tau={record['tau_ns']}, I=({record['Imin']:.2f},{record['Imax']:.2f}), "
                      f"beta=({record['beta0']:.2f}->{record['beta1']:.2f})")
        evaluated.sort(key=lambda item: item[1]["energy"])
        elite_count = max(1, int(pop_size * elite_frac))
        elites = [cfg for cfg, _ in evaluated[:elite_count]]
        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent = rng.choice(elites)
            child = _mutate_config(rng, parent, bits_list, share_list, n_pbit_list, tau_ns_list,
                                   Imin_range, Imax_range, beta0_range, beta1_range,
                                   Imin_default, Imax_default, beta0_default, beta1_default, min_gap)
            new_population.append(child)
        population = new_population
    df = pd.DataFrame(records)
    today = datetime.date.today().strftime("%Y%m%d")
    Path("results").mkdir(exist_ok=True)
    run_tag = _build_run_tag(shared["problem_spec"], shared["sampler"], shared["schedule"], optimizer="evo")
    csv = Path("results") / f"{today}-autoopt-{run_tag}.csv"
    df.to_csv(csv, index=False)
    _plot_optimizer_summary(df, run_tag)
    print(f"\n✔ optimization log saved to {csv}")
    if best is not None:
        print("\nBest configuration:")
        print(pd.Series(best))
    _save_best_config(best, best_out)
    return best

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["gillespie", "tick"], nargs='+', default=["gillespie"])
    ap.add_argument("--schedule", choices=["const", "exp", "linear", "step"], nargs='+', default=["const"])
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--bits-list", type=int, nargs="+", help="Explicit list of ADC bits values to sweep/optimize")
    ap.add_argument("--share-list", type=float, nargs="+", help="Explicit list of share (fan-out) values to sweep/optimize")
    ap.add_argument("--tau-list", type=float, nargs="+", help="Explicit list of tau_ns values to sweep/optimize")
    ap.add_argument("--tick-mode", choices=["random", "rr", "rr-jitter", "block-random", "block-random-stride"], default="random", help="Tick update selection: random (default), round-robin (rr), rr-jitter (RR with random start offset), block-random (contiguous block), or block-random-stride (contiguous generator with random stride)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--Imin", type=float, default=-5.0, help="Minimum I value for quantization")
    ap.add_argument("--Imax", type=float, default=5.0, help="Maximum I value for quantization")
    ap.add_argument("--no-quantize", action="store_true", help="Disable input quantization (use full-precision I)")
    ap.add_argument("--sim-time-ns", type=float, default=5000.0, help="Simulation duration per configuration (ns)")
    ap.add_argument("--dt-ns", type=float, default=1.0, help="Time step for tick sampler (ns)")
    ap.add_argument("--dump-spins", action="store_true", help="Print final spin vector for each run")
    ap.add_argument("--dump-trace", action="store_true", help="Dump energy time series to CSV")
    ap.add_argument("--trace-dir", default=None, help="Directory for energy trace CSVs (used with --dump-trace)")
    ap.add_argument("--track-c1", action="store_true", help="Compute average spin autocorrelation C(1)")
    ap.add_argument("--c1-transient-frac", type=float, default=0.2, help="Fraction of steps to discard when averaging C(1)")
    ap.add_argument("--density", type=float, default=0.1, help="Edge density for random Ising graphs")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for the samplers")
    ap.add_argument("--problem", choices=["random", "tsp", "maxcut", "sat"], default="random", help="Problem class to analyze")
    ap.add_argument("--problem-label", type=str, default=None, help="Optional label used in plots/CSV")
    ap.add_argument("--bench-root", type=str, default="benchmarks", help="Root directory where benchmarks will be cached")
    ap.add_argument("--tsp-instance", type=str, help="TSPLIB instance name or path")
    ap.add_argument("--tsp-penalty", type=float, default=500.0, help="Penalty weight A for TSP QUBO constraints")
    ap.add_argument("--tsp-distance-weight", type=float, default=1.0, help="Objective scale B for TSPLIB distances")
    ap.add_argument("--maxcut-instance", type=str, help="G-set instance name or path")
    ap.add_argument("--maxcut-weight-scale", type=float, default=1.0, help="Scale applied to G-set edge weights")
    ap.add_argument("--sat-instance", type=str, help="SATLIB CNF instance name or path")
    ap.add_argument("--sat-clause-penalty", type=float, default=5.0, help="Clause penalty for SAT QUBO")
    ap.add_argument("--sat-aux-penalty", type=float, default=10.0, help="Penalty for auxiliary AND constraints in SAT QUBO")
    ap.add_argument("--download-only", action="store_true", help="Download/cache the specified benchmark and exit without running DOE")
    ap.add_argument("--optimize", action="store_true", help="Use optimization routines instead of exhaustive DOE")
    ap.add_argument("--optimizer", choices=["random", "anneal", "evo"], default="random", help="Optimizer to use when --optimize is set")
    ap.add_argument("--opt-iters", type=int, default=20, help="Number of random-search iterations when --optimize is set")
    ap.add_argument("--Imin-range", type=float, nargs=2, metavar=("IMIN_MIN", "IMIN_MAX"), help="Range for sampling Imin during optimization")
    ap.add_argument("--Imax-range", type=float, nargs=2, metavar=("IMAX_MIN", "IMAX_MAX"), help="Range for sampling Imax during optimization")
    ap.add_argument("--I-range-gap", type=float, default=0.5, help="Minimum gap between sampled Imin and Imax during optimization")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel worker count for --optimize (default: 1)")
    ap.add_argument("--beta0-range", type=float, nargs=2, metavar=("BETA0_MIN", "BETA0_MAX"), help="Range for sampling beta0 during optimization")
    ap.add_argument("--beta1-range", type=float, nargs=2, metavar=("BETA1_MIN", "BETA1_MAX"), help="Range for sampling beta1 during optimization")
    ap.add_argument("--sa-temp-start", type=float, default=1.0, help="Initial temperature for annealing optimizer")
    ap.add_argument("--sa-temp-end", type=float, default=0.01, help="Final temperature for annealing optimizer")
    ap.add_argument("--evo-pop", type=int, default=10, help="Population size for evolutionary optimizer")
    ap.add_argument("--best-out", type=str, help="If set, save the best configuration from optimization to this path (JSON)")
    ap.add_argument("--auto-beta-range", action="store_true", help="Infer beta0/beta1 ranges from J,h magnitudes when optimizing")
    ap.add_argument("--auto-beta-fixed", action="store_true", help="Set beta0/beta1 using sigma-based heuristic (0.1/sigma, 10/sigma)")
    ap.add_argument("--apply-delay-ns", type=float, default=None, help="Apply delay for Gillespie updates (ns); default: tau")
    ap.add_argument("--print-ising", action="store_true", help="Print the loaded Ising J/h matrices for non-random problems")
    args = ap.parse_args()

    if args.quick:
        bits_list = [4, 6]
        share_list = [2, 4]
        n_pbit_list = [64]
        tau_ns_list = [5, 10, 20]
    else:
        bits_list = [4, 5, 6, 7, 8]
        share_list = [1]
        n_pbit_list = [256, 1024]
        tau_ns_list = [0.1, 0.5, 1, 5, 10]
    if args.bits_list:
        bits_list = args.bits_list
    if args.share_list:
        share_list = args.share_list
    if args.tau_list:
        tau_ns_list = args.tau_list
    quantize = not args.no_quantize

    bench_root = Path(args.bench_root)
    problem_spec = {"kind": "random", "label": args.problem_label or "random", "instance": args.problem_label or "random", "density": args.density, "path": None}
    if args.problem == "tsp":
        if not args.tsp_instance:
            ap.error("--tsp-instance is required when --problem=tsp")
        tsp_dir = bench_root / "tsplib"
        problem_spec = build_tsplib_ising(args.tsp_instance, tsp_dir, args.tsp_penalty, args.tsp_distance_weight)
        if args.problem_label:
            problem_spec["label"] = args.problem_label
        if args.print_ising:
            print_ising_couplings(problem_spec)
        n_pbit_list = [problem_spec["n"]]
    elif args.problem == "sat":
        if not args.sat_instance:
            ap.error("--sat-instance is required when --problem=sat")
        sat_dir = bench_root / "satlib"
        problem_spec = build_satlib_ising(args.sat_instance, sat_dir, args.sat_clause_penalty, args.sat_aux_penalty)
        if args.problem_label:
            problem_spec["label"] = args.problem_label
        if args.print_ising:
            print_ising_couplings(problem_spec)
        n_pbit_list = [problem_spec["n"]]
    elif args.problem == "maxcut":
        if not args.maxcut_instance:
            ap.error("--maxcut-instance is required when --problem=maxcut")
        gset_dir = bench_root / "gset"
        problem_spec = load_maxcut_instance(args.maxcut_instance, gset_dir, weight_scale=args.maxcut_weight_scale)
        if args.problem_label:
            problem_spec["label"] = args.problem_label
        if args.print_ising:
            print_ising_couplings(problem_spec)
        n_pbit_list = [problem_spec["n"]]

    if args.download_only:
        if problem_spec["kind"] == "random":
            print("Random problem does not pull any remote benchmark. Nothing to download.")
        else:
            print(f"Benchmark cached at {problem_spec['path']}")
        return

    if args.auto_beta_fixed:
        inferred = _infer_beta_fixed(problem_spec)
        if inferred:
            args.beta0, args.beta1 = inferred
            print(f"⚙ auto beta fixed (sigma-based): beta0={args.beta0:.4f}, beta1={args.beta1:.4f}")
        else:
            print("⚠ auto beta fixed inference failed; using provided beta0/beta1.")

    if args.optimize:
        if len(args.sampler) != 1 or len(args.schedule) != 1:
            ap.error("--optimize currently supports exactly one sampler and one schedule")
        imin_range = tuple(args.Imin_range) if args.Imin_range else None
        imax_range = tuple(args.Imax_range) if args.Imax_range else None
        beta0_range = tuple(args.beta0_range) if args.beta0_range else None
        beta1_range = tuple(args.beta1_range) if args.beta1_range else None
        if args.auto_beta_range and (beta0_range is None or beta1_range is None):
            inferred = _infer_beta_range(problem_spec, args.Imin, args.Imax)
            if inferred:
                beta0_inf, beta1_inf = inferred
                if beta0_range is None:
                    beta0_range = beta0_inf
                if beta1_range is None:
                    beta1_range = beta1_inf
                print(f"⚙ auto beta range (sigma-based): beta0_range={beta0_range}, beta1_range={beta1_range}")
        optimizer = args.optimizer
        common_kwargs = dict(
            bits_list=bits_list,
            share_list=share_list,
            n_pbit_list=n_pbit_list,
            tau_ns_list=tau_ns_list,
            sampler=args.sampler[0],
            schedule=args.schedule[0],
            steps=args.steps,
            sim_time_ns=args.sim_time_ns,
            dt_ns=args.dt_ns,
            repeats=args.repeats,
            problem_spec=problem_spec,
            rng_seed=args.seed,
            density=args.density,
            opt_iters=args.opt_iters,
            Imin_default=args.Imin,
            Imax_default=args.Imax,
            beta0_default=args.beta0,
            beta1_default=args.beta1,
            Imin_range=imin_range,
            Imax_range=imax_range,
            beta0_range=beta0_range,
            beta1_range=beta1_range,
            min_gap=args.I_range_gap,
            dump_spins=args.dump_spins,
            best_out=args.best_out,
            quantize=quantize,
            apply_delay_ns=args.apply_delay_ns,
        )
        if optimizer == "random":
            random_optimize_parameters(jobs=args.jobs, **common_kwargs)
        elif optimizer == "anneal":
            simulated_annealing_parameters(temp_start=args.sa_temp_start, temp_end=args.sa_temp_end, **common_kwargs)
        elif optimizer == "evo":
            evolutionary_optimize_parameters(pop_size=args.evo_pop, **common_kwargs)
        else:
            raise ValueError(f"Unknown optimizer {optimizer}")
        return

    trace_dir = args.trace_dir
    if args.dump_trace and not trace_dir:
        trace_dir = "results/traces"

    for sampler in args.sampler:
        for schedule in args.schedule:
            simulate_and_plot_list(bits_list, share_list, n_pbit_list, tau_ns_list,
                                   sampler=sampler,
                                   sim_time_ns=args.sim_time_ns,
                                   dt_ns=args.dt_ns,
                                   repeats=args.repeats, schedule=schedule,
                                   beta0=args.beta0, beta1=args.beta1, steps=args.steps, I_min=args.Imin, I_max=args.Imax,
                                   problem_spec=problem_spec, rng_seed=args.seed, density=args.density,
                                   dump_spins=args.dump_spins, quantize=quantize, apply_delay_ns=args.apply_delay_ns, tick_mode=args.tick_mode,
                                   dump_trace=args.dump_trace, trace_dir=trace_dir,
                                   track_c1=args.track_c1, c1_skip_frac=args.c1_transient_frac)

if __name__ == "__main__":
    main()
