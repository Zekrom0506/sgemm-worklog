"""Generate every figure in plots/ from the CSVs in results/.

    python scripts/plots.py

Missing inputs are skipped with a warning; nothing is ever made up.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GPU = {
    "name": "Tesla T4",
    "peak_gflops": 8100.0,       # 40 SMs * 64 FP32 lanes * 2 FLOP/FMA * 1.59 GHz
    "bandwidth_gbs": 320.0,      # GDDR6
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

KERNEL_LABELS = {
    0: "cuBLAS",
    1: "K1 naive",
    2: "K2 coalesced",
    3: "K3 shared mem",
    4: "K4 1D tiling",
    5: "K5 2D tiling",
    6: "K6 vectorized",
    7: "K7 autotuned",
}

# Fixed categorical order so a kernel keeps its colour in every figure.
# cuBLAS is the reference line, drawn in neutral ink rather than a series colour.
KERNEL_COLORS = {
    1: "#2a78d6",
    2: "#eb6834",
    3: "#1baf7a",
    4: "#eda100",
    5: "#e87ba4",
    6: "#008300",
    7: "#6250d6",
}
REFERENCE_COLOR = "#52514e"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#8a8984"
GRID_COLOR = "#e4e3df"
FP32_KERNELS = list(range(0, 8))

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.edgecolor": MUTED_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "legend.frameon": False,
})


def warn(message):
    print(f"[plots] WARNING: {message}", file=sys.stderr)


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[plots] wrote {os.path.relpath(path, ROOT)}")


def load_results():
    path = os.path.join(RESULTS, "results.csv")
    if not os.path.exists(path):
        warn("results/results.csv not found; run scripts/run_all.sh first")
        return None
    df = pd.read_csv(path)
    return df[df["kernel"].isin(FP32_KERNELS)]


def plot_gflops_vs_size(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = sorted(df["size"].unique())

    cublas = df[df["kernel"] == 0].sort_values("size")
    ax.plot(cublas["size"], cublas["gflops"], color=REFERENCE_COLOR, linewidth=2,
            linestyle="--", marker="o", markersize=5, label="cuBLAS")

    for kernel in range(1, 8):
        rows = df[df["kernel"] == kernel].sort_values("size")
        if rows.empty:
            continue
        ax.plot(rows["size"], rows["gflops"], color=KERNEL_COLORS[kernel], linewidth=2,
                marker="o", markersize=5, label=KERNEL_LABELS[kernel])

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Matrix size (M = N = K)")
    ax.set_ylabel("GFLOPS (FP32)")
    ax.set_title(f"SGEMM throughput vs size on {GPU['name']}", loc="left", color=TEXT_COLOR)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "gflops_vs_size.png")


def plot_bar_4096(df):
    size = 4096 if 4096 in df["size"].values else df["size"].max()
    rows = df[df["size"] == size].sort_values("kernel")
    if rows.empty:
        warn("no rows for the bar chart")
        return

    labels = [KERNEL_LABELS[k] for k in rows["kernel"]]
    colors = [REFERENCE_COLOR if k == 0 else KERNEL_COLORS[k] for k in rows["kernel"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rows["gflops"], color=colors, width=0.65)
    for bar, pct in zip(bars, rows["pct_cublas"]):
        ax.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, color=TEXT_COLOR)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("GFLOPS (FP32)")
    ax.set_title(f"GFLOPS at {size} (labels: % of cuBLAS)", loc="left", color=TEXT_COLOR)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, "bar_4096.png")


def read_ncu_csv(path):
    """Return {metric_name: float} from an `ncu --page raw --csv` export."""
    data = pd.read_csv(path, dtype=str)
    # ncu writes a units row (empty ID column) directly under the header.
    if not data.empty and pd.isna(data.iloc[0].get("ID")):
        data = data.iloc[1:]
    if data.empty:
        return {}
    row = data.iloc[0]
    metrics = {}
    for column, value in row.items():
        try:
            metrics[column] = float(str(value).replace(",", ""))
        except ValueError:
            pass
    return metrics


def load_ncu():
    files = sorted(glob.glob(os.path.join(RESULTS, "ncu", "k*_*.csv")))
    table = {}
    for path in files:
        kernel = int(os.path.basename(path)[1:].split("_")[0])
        try:
            metrics = read_ncu_csv(path)
        except Exception as error:  # a truncated or empty export
            warn(f"could not parse {path}: {error}")
            continue
        if "dram__bytes_read.sum" in metrics:
            table[kernel] = metrics
    return table


def plot_roofline(df, ncu):
    fp32 = {k: v for k, v in ncu.items() if k in FP32_KERNELS}
    if not fp32:
        warn("no ncu data in results/ncu/; skipping the roofline plot")
        return

    size = 4096
    flops = 2.0 * size ** 3
    peak = GPU["peak_gflops"]
    bandwidth = GPU["bandwidth_gbs"]
    ridge = peak / bandwidth

    fig, ax = plt.subplots(figsize=(8, 5.5))
    intensity = [0.5, ridge, 2000]
    roof = [min(peak, i * bandwidth) for i in intensity]
    ax.plot(intensity, roof, color=TEXT_COLOR, linewidth=2)
    ax.annotate(f"{peak / 1000:.1f} TFLOPS FP32 peak", (intensity[-1], peak),
                xytext=(-4, 6), textcoords="offset points", ha="right", color=TEXT_COLOR)
    ax.annotate(f"{bandwidth:.0f} GB/s DRAM", (1.0, bandwidth),
                xytext=(4, 8), textcoords="offset points", rotation=33, color=TEXT_COLOR)

    for kernel, metrics in sorted(fp32.items()):
        dram_bytes = metrics["dram__bytes_read.sum"] + metrics.get("dram__bytes_write.sum", 0.0)
        ai = flops / dram_bytes
        # Prefer the harness timing; ncu locks clocks to base, which is slower.
        measured = None
        if df is not None:
            match = df[(df["kernel"] == kernel) & (df["size"] == size)]
            if not match.empty:
                measured = float(match["gflops"].iloc[0])
        if measured is None and "gpu__time_duration.sum" in metrics:
            measured = flops / (metrics["gpu__time_duration.sum"] * 1e-9) / 1e9
        if measured is None:
            continue
        color = REFERENCE_COLOR if kernel == 0 else KERNEL_COLORS[kernel]
        ax.scatter([ai], [measured], s=60, color=color, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(KERNEL_LABELS[kernel], (ai, measured), xytext=(6, -3),
                    textcoords="offset points", fontsize=9, color=TEXT_COLOR)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / DRAM byte, from ncu)")
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"Roofline, {GPU['name']}, size {size}", loc="left", color=TEXT_COLOR)
    save(fig, "roofline.png")


def plot_autotune_heatmap():
    path = os.path.join(RESULTS, "autotune.csv")
    if not os.path.exists(path):
        warn("results/autotune.csv not found; run scripts/autotune.sh first")
        return
    df = pd.read_csv(path)
    if df.empty:
        warn("autotune.csv is empty")
        return

    best = df.loc[df["gflops"].idxmax()]
    bk, tm, tn = int(best["BK"]), int(best["TM"]), int(best["TN"])
    subset = df[(df["BK"] == bk) & (df["TM"] == tm) & (df["TN"] == tn)]
    grid = subset.pivot_table(index="BM", columns="BN", values="gflops")
    grid = grid.reindex(index=[64, 128], columns=[64, 128])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(grid.values, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("BN")
    ax.set_ylabel("BM")
    values = pd.Series(grid.values.ravel()).dropna()
    midpoint = (values.min() + values.max()) / 2
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            text = "n/a" if pd.isna(value) else f"{value:.0f}"
            dark_cell = not pd.isna(value) and value > midpoint
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if dark_cell else TEXT_COLOR)
    fig.colorbar(image, ax=ax, label="GFLOPS")
    ax.set_title(f"K7 GFLOPS at BK={bk}, TM={tm}, TN={tn}", loc="left", color=TEXT_COLOR)
    save(fig, "autotune_heatmap.png")


def main():
    df = load_results()
    if df is not None and not df.empty:
        plot_gflops_vs_size(df)
        plot_bar_4096(df)
    plot_roofline(df, load_ncu())
    plot_autotune_heatmap()


if __name__ == "__main__":
    main()
