"""Fill the data tables in README.md from the files in results/.

    python scripts/update_readme.py

Every number in the README comes from here. Each table lives between
<!-- BEGIN:name --> and <!-- END:name --> markers; anything between the markers
is regenerated, everything else in the README is left alone.
"""

import glob
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
SIZE = 4096

NAMES = {
    0: "cuBLAS (reference)",
    1: "K1 naive",
    2: "K2 global memory coalescing",
    3: "K3 shared memory tiling",
    4: "K4 1D block tiling",
    5: "K5 2D block tiling",
    6: "K6 vectorized + transposed A",
    7: "K7 autotuned",
}
PTXAS_NAMES = {
    1: "sgemm_naive",
    2: "sgemm_coalesced",
    3: "sgemm_shared_mem",
    4: "sgemm_1d_blocktiling",
    5: "sgemm_2d_blocktiling",
    6: "sgemm_vectorized",
}


def pending(script):
    return f"_Not measured yet: run `{script}` on the GPU, then `python scripts/update_readme.py`._"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def load_results():
    path = os.path.join(RESULTS, "results.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def results_at_size(df):
    rows = df[(df["size"] == SIZE) & (df["kernel"] <= 7)]
    return {int(r.kernel): r for r in rows.itertuples()}


def results_table(df):
    if df is None:
        return pending("bash scripts/run_all.sh")
    by_kernel = results_at_size(df)
    rows = []
    previous = None
    for kernel in range(0, 8):
        if kernel not in by_kernel:
            continue
        r = by_kernel[kernel]
        speedup = "" if kernel <= 1 or previous is None else f"{r.gflops / previous:.2f}x"
        rows.append([NAMES[kernel], f"{r.gflops:.0f}", f"{r.pct_cublas:.1f}%", speedup])
        if kernel >= 1:
            previous = r.gflops
    return md_table(["Kernel", f"GFLOPS @ {SIZE}", "% of cuBLAS", "Speedup vs previous"], rows)


def sweep_table(df):
    if df is None:
        return pending("bash scripts/run_all.sh")
    fp32 = df[df["kernel"] <= 7]
    pivot = fp32.pivot_table(index="kernel", columns="size", values="gflops")
    rows = [[NAMES[int(k)]] + [f"{v:.0f}" for v in pivot.loc[k]] for k in pivot.index]
    header = ["GFLOPS"] + [str(int(s)) for s in pivot.columns]
    worst = fp32["max_abs_error"].max()
    note = f"\n\nLargest max-abs-error against cuBLAS across all kernels and sizes: {worst:.2e} (tolerance 1e-2)."
    return md_table(header, rows) + note


def kernel_result(df, kernel):
    if df is None:
        return pending("bash scripts/run_all.sh")
    by_kernel = results_at_size(df)
    if kernel not in by_kernel:
        return pending("bash scripts/run_all.sh")
    r = by_kernel[kernel]
    text = f"**Measured at {SIZE}:** {r.gflops:.0f} GFLOPS, {r.pct_cublas:.1f}% of cuBLAS"
    if kernel - 1 in by_kernel and kernel > 1:
        text += f", {r.gflops / by_kernel[kernel - 1].gflops:.2f}x over K{kernel - 1}"
    return text + f" (median of 20 runs, {r.median_ms:.2f} ms)."


def device_line():
    log = os.path.join(RESULTS, "run_all.log")
    if not os.path.exists(log):
        return pending("bash scripts/run_all.sh")
    with open(log) as f:
        for line in f:
            if line.startswith("Device:"):
                return "Measured on: `" + line.strip()[len("Device: "):] + "`"
    return pending("bash scripts/run_all.sh")


def read_ncu(path):
    data = pd.read_csv(path, dtype=str)
    if not data.empty and pd.isna(data.iloc[0].get("ID")):
        data = data.iloc[1:]
    if data.empty:
        return None
    row = data.iloc[0]
    values = {}
    for column, value in row.items():
        try:
            values[column] = float(str(value).replace(",", ""))
        except ValueError:
            pass
    return values


def load_ncu():
    table = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "ncu", "k*_*.csv"))):
        kernel = int(os.path.basename(path)[1:].split("_")[0])
        try:
            values = read_ncu(path)
        except Exception:
            continue
        if values:
            table[kernel] = values
    return table


def ncu_table(ncu):
    if not ncu:
        return pending("bash scripts/profile.sh")

    def get(values, name, scale=1.0, fmt="{:.1f}"):
        return fmt.format(values[name] * scale) if name in values else "-"

    rows = []
    for kernel in sorted(k for k in ncu if k <= 7):
        v = ncu[kernel]
        rows.append([
            NAMES[kernel],
            get(v, "gpu__time_duration.sum", 1e-6, "{:.2f}"),
            get(v, "dram__bytes_read.sum", 1e-6, "{:.0f}"),
            get(v, "dram__bytes_write.sum", 1e-6, "{:.0f}"),
            get(v, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            get(v, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            get(v, "sm__warps_active.avg.pct_of_peak_sustained_active"),
            get(v, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 1.0, "{:.0f}"),
            get(v, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 1.0, "{:.0f}"),
        ])
    header = ["Kernel", "time (ms)", "DRAM read (MB)", "DRAM write (MB)", "SM %", "DRAM %",
              "achieved occupancy %", "smem ld bank conflicts", "smem st bank conflicts"]
    return md_table(header, rows) + "\n\n(ncu locks clocks to base frequency, so its times are longer than the harness timings.)"


def stall_table(ncu):
    if not ncu:
        return pending("bash scripts/profile.sh")
    # e.g. smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    pattern = re.compile(r"^smsp__average_warps?_(?:latency_)?issue_stalled_(\w+?)_per_issue_active\.ratio$")
    rows = []
    for kernel in sorted(k for k in ncu if k <= 7):
        stalls = {}
        for name, value in ncu[kernel].items():
            match = pattern.match(name)
            if match and not match.group(1).endswith("not_issued"):
                stalls[match.group(1)] = value
        if not stalls:
            continue
        top = sorted(stalls.items(), key=lambda item: item[1], reverse=True)[:3]
        rows.append([NAMES[kernel]] + [f"{reason} ({value:.2f})" for reason, value in top])
    if not rows:
        return "_The WarpStateStats section was not in the ncu export._"
    return md_table(["Kernel", "Top stall reason", "2nd", "3rd"], rows) + \
        "\n\n(cycles stalled per issued instruction, from the WarpStateStats section)"


def resource_table():
    path = os.path.join(RESULTS, "resource_usage.txt")
    if not os.path.exists(path):
        return pending("bash scripts/profile.sh")
    usage = {}
    header_line = ""
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                header_line = line.lstrip("# ").strip()
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 4 and parts[1].isdigit():
                usage[parts[0]] = parts[1:]
    rows = []
    for kernel, name in PTXAS_NAMES.items():
        if name in usage:
            regs, smem, spills = usage[name]
            rows.append([f"K{kernel}", name, regs, smem, spills])
    k7 = [(name, values) for name, values in usage.items() if name.startswith("sgemm_autotuned")]
    for name, (regs, smem, spills) in sorted(k7, key=lambda item: int(item[1][0])):
        rows.append(["K7", name.replace("sgemm_autotuned", "autotuned"), regs, smem, spills])
    if "sgemm_wmma_fp16" in usage:
        rows.append(["K8", "sgemm_wmma_fp16"] + usage["sgemm_wmma_fp16"])
    table = md_table(["", "kernel", "registers / thread", "static smem (bytes)", "spill stores (bytes)"], rows)
    return f"From `{header_line}`:\n\n" + table if header_line else table


def load_autotune():
    path = os.path.join(RESULTS, "autotune.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def autotune_table(at, best=True, count=5):
    if at is None or at.empty:
        return pending("bash scripts/autotune.sh")
    ordered = at.sort_values("gflops", ascending=not best).head(count)
    rows = [[int(r.BM), int(r.BN), int(r.BK), int(r.TM), int(r.TN), int(r.threads),
             f"{r.gflops:.0f}", f"{r.pct_cublas:.1f}%"] for r in ordered.itertuples()]
    note = f"\n\n{len(at)} valid configurations were measured at size {int(at['size'].iloc[0])}."
    return md_table(["BM", "BN", "BK", "TM", "TN", "threads", "GFLOPS", "% cuBLAS"], rows) + (note if best else "")


def fp16_table(df):
    if df is None:
        return pending("bash scripts/run_all.sh")
    rows = df[df["kernel"] == 8]
    if rows.empty:
        return pending("bash scripts/run_all.sh")
    out = []
    for size in sorted(rows["size"].unique()):
        at_size = rows[rows["size"] == size]
        ours = at_size[at_size["name"] == "wmma_fp16"]
        ref = at_size[at_size["name"] == "cublas_fp16"]
        if ours.empty or ref.empty:
            continue
        out.append([int(size), f"{ref['gflops'].iloc[0]:.0f}", f"{ours['gflops'].iloc[0]:.0f}",
                    f"{ours['pct_cublas'].iloc[0]:.1f}%", f"{ours['max_abs_error'].iloc[0]:.1e}"])
    return md_table(["size", "cublasGemmEx FP16->FP32 GFLOPS", "K8 WMMA GFLOPS",
                     "% of cublasGemmEx", "max abs error"], out)


def replace_block(text, name, content):
    pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", re.S)
    if not pattern.search(text):
        print(f"[update_readme] marker {name} not found, skipping")
        return text
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text)


def main():
    df = load_results()
    ncu = load_ncu()
    at = load_autotune()
    blocks = {
        "device": device_line(),
        "results_table": results_table(df),
        "sweep_table": sweep_table(df),
        "ncu_table": ncu_table(ncu),
        "stall_table": stall_table(ncu),
        "resource_table": resource_table(),
        "autotune_best": autotune_table(at, best=True),
        "autotune_worst": autotune_table(at, best=False, count=3),
        "fp16_table": fp16_table(df),
    }
    for kernel in range(1, 8):
        blocks[f"k{kernel}_result"] = kernel_result(df, kernel)

    with open(README) as f:
        text = f.read()
    for name, content in blocks.items():
        text = replace_block(text, name, content)
    with open(README, "w") as f:
        f.write(text)
    print("[update_readme] README.md updated")


if __name__ == "__main__":
    main()
