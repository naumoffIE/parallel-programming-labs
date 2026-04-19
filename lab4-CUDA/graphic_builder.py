import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FILE_NAME   = "results.csv"
OUTPUT_FILE = "analysis_plot.png"

df = pd.read_csv(FILE_NAME)
df = df.dropna(subset=["time_kernel_ms"])

print(f"✓ Загружено {len(df)} записей из {FILE_NAME}")
print(f"  Размеры матриц:  {sorted(df['size_N'].unique())}")
print(f"  Размеры блоков:  {sorted(df['block_size'].unique())}")
print(f"  Режимы:          {df['mode'].unique().tolist()}")

BLOCK_COLORS = {8: "#e15759", 16: "#4e79a7", 32: "#59a14f"}
MODE_LS      = {"global": "--", "tiled": "-"}
MODE_MK      = {"global": "^",  "tiled": "o"}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("CUDA-умножение матриц: анализ производительности (sm_100)",
             fontsize=15, fontweight="bold")

# ── 1: Время ядра vs N ──────────────────────────────────────────
ax = axes[0, 0]
tiled_df = df[df["mode"] == "tiled"]
for bs in sorted(df["block_size"].unique()):
    sub = tiled_df[tiled_df["block_size"] == bs].sort_values("size_N")
    ax.plot(sub["size_N"], sub["time_kernel_ms"], marker="o",
            color=BLOCK_COLORS.get(bs, "gray"), label=f"block={bs}×{bs}", lw=2)
ax.set_title("Время ядра vs N  (tiled, без передачи данных)", fontsize=12)
ax.set_xlabel("Размер матрицы N")
ax.set_ylabel("Время (мс)")
ax.set_yscale("log")
ax.legend(); ax.grid(True, alpha=0.3)

# ── 2: Ядро vs Полное время (честное сравнение) ─────────────────
ax = axes[0, 1]
for bs in sorted(df["block_size"].unique()):
    sub_t = (df[(df["block_size"] == bs) & (df["mode"] == "tiled")]
             .sort_values("size_N"))
    ax.plot(sub_t["size_N"], sub_t["time_kernel_ms"],
            linestyle="-", marker="o", color=BLOCK_COLORS.get(bs, "gray"),
            label=f"ядро {bs}×{bs}", lw=1.8)
    ax.plot(sub_t["size_N"], sub_t["time_total_ms"],
            linestyle=":", marker="s", color=BLOCK_COLORS.get(bs, "gray"),
            label=f"полное {bs}×{bs}", lw=1.2, alpha=0.75)

cpu_sub = df[["size_N","time_cpu_ms"]].dropna().drop_duplicates("size_N").sort_values("size_N")
ax.plot(cpu_sub["size_N"], cpu_sub["time_cpu_ms"],
        color="black", linestyle="-.", marker="D", lw=2, label="CPU (lab1)")

ax.set_title("Ядро (—) vs Полное время · · · vs CPU  (tiled)", fontsize=12)
ax.set_xlabel("Размер матрицы N")
ax.set_ylabel("Время (мс)")
ax.set_yscale("log")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

# ── 3: Speedup честный (total) vs kernel ────────────────────────
ax = axes[1, 0]
has = df.dropna(subset=["speedup_kernel", "speedup_total"])
tiled_has = has[has["mode"] == "tiled"]
for bs in sorted(df["block_size"].unique()):
    sub = tiled_has[tiled_has["block_size"] == bs].sort_values("size_N")
    ax.plot(sub["size_N"], sub["speedup_kernel"],
            linestyle="-", marker="o", color=BLOCK_COLORS.get(bs, "gray"),
            label=f"sp_ядро {bs}×{bs}", lw=2)
    ax.plot(sub["size_N"], sub["speedup_total"],
            linestyle=":", marker="s", color=BLOCK_COLORS.get(bs, "gray"),
            label=f"sp_полный {bs}×{bs}", lw=1.2, alpha=0.75)
ax.axhline(y=1, color="red", linestyle="--", alpha=0.6, label="Нет ускорения")
ax.set_title("Speedup: только ядро (—) vs полный · · ·  (tiled)", fontsize=12)
ax.set_xlabel("Размер матрицы N")
ax.set_ylabel("Ускорение (×)")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

# ── 4: global vs tiled — разница в ядре ─────────────────────────
ax = axes[1, 1]
sizes = sorted(df["size_N"].unique())
for bs in sorted(df["block_size"].unique()):
    diffs = []
    for n in sizes:
        g = df[(df["size_N"]==n)&(df["block_size"]==bs)&(df["mode"]=="global")]["time_kernel_ms"]
        t = df[(df["size_N"]==n)&(df["block_size"]==bs)&(df["mode"]=="tiled")]["time_kernel_ms"]
        if not g.empty and not t.empty:
            diffs.append(float(g.values[0]) - float(t.values[0]))
        else:
            diffs.append(None)
    ax.plot(sizes, diffs, marker="o", color=BLOCK_COLORS.get(bs, "gray"),
            label=f"block={bs}×{bs}", lw=2)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Выигрыш tiled vs global (мс)\n> 0 означает tiled быстрее", fontsize=12)
ax.set_xlabel("Размер матрицы N")
ax.set_ylabel("Δ время (мс)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=120, bbox_inches="tight")
print(f"✓ График сохранён: {OUTPUT_FILE}")
plt.show()
print("✓ Готово!")