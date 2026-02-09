import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh

# ============================================================
# Config
# ============================================================
GRAPHS = ["Toy-1", "Toy-2", "Toy-3", "Toy-4", "Toy-5", "Toy-6a", "Toy-6b", "Toy-7", "G1", "G6", "G11", "G35", "G39", "G58", "G63", "G64"]
BASE_DIR = "./gset"
OUT_DIR = "./theory_transient_6graphs_ipr"
os.makedirs(OUT_DIR, exist_ok=True)
SKIP_IF_EXISTS = True
GRAPH_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x"]

# --- c search (continuous) ---
C_LOW = 1.0
C_HIGH = 10.0          # if not enough, it will auto-expand
C_TOL = 1e-3
MAX_BISECT_IT = 60

# --- time horizon ---
# 5ns = 1 step
T_STEPS = 40  # 100ns=20, 200ns=40, 500ns=100 etc.

# --- transient growth criterion ---
# 判定：Tステップの間に “観測される不安定モード” が R倍以上に増幅したら「発振が見える」
R_GROWTH = 10.0

# --- I0 schedule mode ---
MODE = "fixed"   # "fixed" or "anneal"
NUM_I0 = 25

# Gaussian alpha_eff settings
N_GH = 60  # Gauss-Hermite points

# ============================================================
# NEW: observable-mode selection (IPR filtering)
# ============================================================
# how many smallest eigenpairs of A to inspect at each step
K_EIG = 12     # 8-20 is typical; increase if you want more robustness
# score = |lambda| * (1/IPR)^BETA  (prefer unstable AND delocalized)
BETA = 1.0
# optional: only accept delocalized modes. Set None to disable hard threshold.
# Rough guide: delocalized mode has IPR ~ O(1/N). Try IP_THRESHOLD = 50/N or 100/N.
IPR_THRESHOLD_SCALE = 80.0  # IPR_thresh = IPR_THRESHOLD_SCALE / N ; set 0 to disable hard filtering

# Environment overrides for quick sweeps
if "K_EIG" in os.environ:
    K_EIG = int(os.environ["K_EIG"])
if "BETA" in os.environ:
    BETA = float(os.environ["BETA"])
if "IPR_THRESHOLD_SCALE" in os.environ:
    IPR_THRESHOLD_SCALE = float(os.environ["IPR_THRESHOLD_SCALE"])

# If IPR_SWEEP=1, run multiple parameter sets in one execution.
SWEEP_PRESET = [
    (8, 1.0, 50.0),
    (12, 1.0, 80.0),
    (16, 0.5, 80.0),
]
RUN_SWEEP = os.environ.get("IPR_SWEEP", "0") == "1"

# ============================================================
# 1) Load G-set graph and build J
# ============================================================
def load_gset(path: str) -> np.ndarray:
    with open(path, "r") as f:
        lines = f.readlines()
    n, _ = map(int, lines[0].split())
    J = np.zeros((n, n), dtype=np.float64)
    for line in lines[1:]:
        if not line.strip():
            continue
        i, j, w = line.split()
        i = int(i) - 1
        j = int(j) - 1
        w = float(w)
        # MAX-CUT -> Ising (antiferromagnetic)
        J[i, j] = -w
        J[j, i] = -w
    return J

def normalize_J_and_sigma(J: np.ndarray) -> tuple[np.ndarray, float]:
    n = J.shape[0]
    var = np.var(J, axis=1).mean()
    sigma = np.sqrt((n - 1) * var)
    return J / sigma, float(sigma)

# ============================================================
# 2) Gaussian alpha_eff(I0) with sigma_h estimated from J
# ============================================================
def estimate_sigma_h_from_J(J: np.ndarray) -> float:
    row_sumsq = np.sum(J * J, axis=1)
    sigma_h2 = float(np.mean(row_sumsq))
    return float(np.sqrt(sigma_h2))

def alpha_eff_gaussian(I0: float, sigma_h: float, n_gh: int = 60) -> float:
    if I0 <= 0 or sigma_h <= 0:
        return 1.0
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    z = np.sqrt(2.0) * x
    h = sigma_h * z
    u = I0 * h
    sech2 = 1.0 / (np.cosh(u) ** 2)
    return float((w @ sech2) / np.sqrt(np.pi))

# ============================================================
# 3) Build Jacobian A(c)
# ============================================================
def build_A(Jsp: csr_matrix, c: float, I0: float, alpha_eff: float) -> csr_matrix:
    n = Jsp.shape[0]
    Isp = identity(n, format="csr")
    A = (1.0 - 1.0 / c) * Isp + (1.0 / c) * (alpha_eff * I0) * Jsp
    return A

def ipr_of_vec(v: np.ndarray) -> float:
    # v is assumed normalized by eigsh
    return float(np.sum(v**4))

def select_observable_mode(A: csr_matrix, n: int, k_eig: int) -> tuple[float, float, float]:
    """
    Returns:
      lam_sel: selected eigenvalue (real)
      mag_sel: |lam_sel|
      ipr_sel: IPR of selected eigenvector

    Strategy:
      - compute k smallest algebraic eigenpairs (SA)
      - among them, choose one with large |lam| and delocalized (small IPR)
        by maximizing score = |lam| * (1/IPR)^BETA
      - optional hard filter IPR < IPR_threshold (scaled by 1/N)
    """
    # eigsh requires k < N for sparse; fall back to dense if N is small.
    if n <= 2:
        vals = np.array([1.0], dtype=float)
        vecs = np.ones((n, 1), dtype=float)
    else:
        k_eff = min(int(k_eig), n - 1)
        if k_eff < 1:
            k_eff = 1
        if k_eff >= n:
            # dense path for tiny graphs
            dense = A.toarray()
            vals_all, vecs_all = np.linalg.eigh(dense)
            vals = vals_all
            vecs = vecs_all
        else:
            vals, vecs = eigsh(A, k=k_eff, which="SA", return_eigenvectors=True)
    # vals: (k,), vecs: (n,k)

    ipr_thresh = None
    if IPR_THRESHOLD_SCALE and IPR_THRESHOLD_SCALE > 0:
        ipr_thresh = IPR_THRESHOLD_SCALE / float(n)

    best = None
    best_score = -np.inf

    for j in range(len(vals)):
        lam = float(vals[j])
        v = vecs[:, j]
        ipr = ipr_of_vec(v)

        if ipr_thresh is not None and ipr > ipr_thresh:
            # too localized -> likely not visible in global energy
            continue

        mag = abs(lam)
        # avoid division by zero
        score = mag * (1.0 / max(ipr, 1e-18)) ** BETA

        if score > best_score:
            best_score = score
            best = (lam, mag, ipr)

    # if everything got filtered out, fall back to the most unstable (largest |lam|) among computed modes
    if best is None:
        j = int(np.argmax(np.abs(vals)))
        lam = float(vals[j])
        v = vecs[:, j]
        ipr = ipr_of_vec(v)
        best = (lam, abs(lam), ipr)

    return best

# ============================================================
# 4) Transient growth metric over T steps (observable mode)
# ============================================================
def build_I0_schedule(I0_min: float, I0_max: float, T: int) -> np.ndarray:
    if T <= 1:
        return np.array([I0_max], dtype=float)
    return np.linspace(I0_min, I0_max, T, dtype=float)

def log_growth_over_T(Jsp: csr_matrix, c: float, I0_seq: np.ndarray, sigma_h: float) -> float:
    """
    log G = sum_t log(|lambda_observable(t)|)
    """
    n = Jsp.shape[0]
    lg = 0.0
    for I0 in I0_seq:
        aeff = alpha_eff_gaussian(float(I0), sigma_h, n_gh=N_GH)
        A = build_A(Jsp, c, float(I0), aeff)
        lam_sel, mag_sel, ipr_sel = select_observable_mode(A, n=n, k_eig=K_EIG)

        mag = max(float(mag_sel), 1e-12)
        lg += np.log(mag)
    return float(lg)

def is_visible_oscillation(Jsp: csr_matrix, c: float, I0_seq: np.ndarray, sigma_h: float, R: float) -> bool:
    return log_growth_over_T(Jsp, c, I0_seq, sigma_h) >= np.log(R)

# ============================================================
# 5) Find c* by bisection (continuous c)
# ============================================================
def find_c_star_continuous(Jsp: csr_matrix, I0_min: float, I0_max: float, sigma_h: float,
                           T: int, mode: str, R: float,
                           c_low: float = 1.0, c_high: float = 10.0, tol: float = 1e-3) -> float:
    if mode == "fixed":
        I0_seq = np.array([I0_max] * T, dtype=float)
    elif mode == "anneal":
        I0_seq = build_I0_schedule(I0_min, I0_max, T)
    else:
        raise ValueError("MODE must be 'fixed' or 'anneal'")

    lo = c_low
    hi = c_high

    # if already stable at lo, return lo
    if not is_visible_oscillation(Jsp, lo, I0_seq, sigma_h, R):
        return float(lo)

    # expand hi until stable
    while is_visible_oscillation(Jsp, hi, I0_seq, sigma_h, R):
        hi *= 1.5
        if hi > 200:
            return float("nan")

    # bisection
    for _ in range(MAX_BISECT_IT):
        mid = 0.5 * (lo + hi)
        if is_visible_oscillation(Jsp, mid, I0_seq, sigma_h, R):
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break

    return float(hi)

# ============================================================
# 6) Per-graph run
# ============================================================
def run_graph(gname: str):
    gfile = os.path.join(BASE_DIR, gname, f"{gname}.txt")
    if not os.path.isfile(gfile):
        raise FileNotFoundError(gfile)

    J_raw = load_gset(gfile)
    J, sigma = normalize_J_and_sigma(J_raw)
    Jsp = csr_matrix(J)
    sigma_h = estimate_sigma_h_from_J(J)

    I0_min = 0.1 / sigma
    I0_max = 10.0 / sigma

    rows = []

    if MODE == "fixed":
        I0_list = np.logspace(np.log10(I0_min), np.log10(I0_max), NUM_I0)
        cstars = []
        for I0 in I0_list:
            c_star = find_c_star_continuous(
                Jsp, I0_min=float(I0), I0_max=float(I0), sigma_h=sigma_h,
                T=T_STEPS, mode="fixed", R=R_GROWTH,
                c_low=C_LOW, c_high=C_HIGH, tol=C_TOL
            )
            cstars.append(c_star)
            rows.append([gname, J.shape[0], sigma, sigma_h, float(I0), MODE, T_STEPS, R_GROWTH, c_star])
        return I0_list, np.array(cstars, dtype=float), rows

    elif MODE == "anneal":
        I0_final_list = np.logspace(np.log10(I0_min), np.log10(I0_max), NUM_I0)
        cstars = []
        for I0_final in I0_final_list:
            c_star = find_c_star_continuous(
                Jsp, I0_min=float(I0_min), I0_max=float(I0_final), sigma_h=sigma_h,
                T=T_STEPS, mode="anneal", R=R_GROWTH,
                c_low=C_LOW, c_high=C_HIGH, tol=C_TOL
            )
            cstars.append(c_star)
            rows.append([gname, J.shape[0], sigma, sigma_h, float(I0_final), MODE, T_STEPS, R_GROWTH, c_star])
        return I0_final_list, np.array(cstars, dtype=float), rows

    else:
        raise ValueError("MODE must be 'fixed' or 'anneal'")

# ============================================================
# 7) Main
# ============================================================
def main_for_params(k_eig: int, beta: float, ipr_scale: float):
    tag = f"{MODE}_T{T_STEPS}_R{R_GROWTH:g}_Keig{k_eig}_beta{beta:g}_ipr{ipr_scale:g}"
    out_csv = os.path.join(OUT_DIR, f"cstar_transient_6graphs_{tag}.csv")

    header = ["graph", "N", "sigma", "sigma_h", "I0_x", "mode", "T_steps", "R_growth", "c_star"]
    all_curves = []
    all_rows = []

    existing_rows = {}
    if SKIP_IF_EXISTS and os.path.exists(out_csv):
        with open(out_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    if r.get("mode") != MODE:
                        continue
                    if int(float(r.get("T_steps", 0))) != int(T_STEPS):
                        continue
                    if float(r.get("R_growth", 0.0)) != float(R_GROWTH):
                        continue
                    g = r.get("graph")
                    if not g:
                        continue
                    existing_rows.setdefault(g, []).append(r)
                except Exception:
                    continue

    for gi, g in enumerate(GRAPHS):
        marker = GRAPH_MARKERS[gi % len(GRAPH_MARKERS)]
        existing_plot = os.path.exists(os.path.join(OUT_DIR, f"phase_curve_{g}_{tag}.png"))
        if SKIP_IF_EXISTS and g in existing_rows:
            rows = existing_rows[g]
            rows = sorted(rows, key=lambda r: float(r["I0_x"]))
            xlist = np.array([float(r["I0_x"]) for r in rows], dtype=float)
            cstars = np.array([float(r["c_star"]) for r in rows], dtype=float)
            all_rows.extend(rows)
            print(f"Skip compute {g} (use existing CSV).")
        elif SKIP_IF_EXISTS and existing_plot:
            print(f"Skip compute {g} (found existing plot, but no CSV rows).")
            continue
        else:
            print(f"Compute {g} ...")
            # override globals for this run
            globals()["K_EIG"] = int(k_eig)
            globals()["BETA"] = float(beta)
            globals()["IPR_THRESHOLD_SCALE"] = float(ipr_scale)
            xlist, cstars, rows = run_graph(g)
            for r in rows:
                all_rows.append(dict(zip(header, r)))

        # per-graph plot
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(xlist, cstars, marker=marker, linewidth=1.6)
        plt.xscale("log")
        plt.xlabel(r"$I_0$" if MODE == "fixed" else r"$I_{0,\max}$ (final)")
        plt.ylabel(r"$c^*$ (observable transient)")
        plt.title(f"{g}: observable transient boundary ({MODE}, T={T_STEPS})")
        plt.grid(True, which="both", alpha=0.35)
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"phase_curve_{g}_{tag}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()

        all_curves.append((g, xlist, cstars, marker))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # combined plots (toy vs G-set)
    toy_curves = [(g, x, y, m) for (g, x, y, m) in all_curves if g.startswith("Toy-")]
    gset_curves = [(g, x, y, m) for (g, x, y, m) in all_curves if not g.startswith("Toy-")]

    def plot_group(curves, title, out_png):
        if not curves:
            return
        plt.figure(figsize=(7.8, 5.4))
        for g, xlist, cstars, marker in curves:
            plt.plot(xlist, cstars, marker=marker, linewidth=1.4, label=g)
        plt.xscale("log")
        plt.xlabel(r"$I_0$" if MODE == "fixed" else r"$I_{0,\max}$ (final)")
        plt.ylabel(r"$c^*$ (observable transient)")
        plt.title(title)
        plt.grid(True, which="both", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

    out_png_toy = os.path.join(OUT_DIR, f"phase_curves_toy_{tag}.png")
    out_png_gset = os.path.join(OUT_DIR, f"phase_curves_gset_{tag}.png")
    plot_group(toy_curves,
               f"Stability boundaries (toy graphs, Gaussian $\\alpha_{{\\rm eff}}$, observable-mode, {MODE}, T={T_STEPS})",
               out_png_toy)
    plot_group(gset_curves,
               f"Stability boundaries (G-set graphs, Gaussian $\\alpha_{{\\rm eff}}$, observable-mode, {MODE}, T={T_STEPS})",
               out_png_gset)

    print("\nSaved:")
    print(" ", out_csv)
    print(" ", out_png_toy)
    print(" ", out_png_gset)
    print(" ", f"{OUT_DIR}/phase_curve_<Gxx>_{tag}.png")

def main():
    if RUN_SWEEP:
        for k_eig, beta, ipr_scale in SWEEP_PRESET:
            print(f"\n=== IPR sweep: K_EIG={k_eig}, BETA={beta}, IPR_THRESHOLD_SCALE={ipr_scale} ===")
            main_for_params(k_eig, beta, ipr_scale)
        return
    main_for_params(K_EIG, BETA, IPR_THRESHOLD_SCALE)

if __name__ == "__main__":
    main()
