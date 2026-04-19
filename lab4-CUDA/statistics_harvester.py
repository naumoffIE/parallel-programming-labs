import numpy as np
import subprocess
import os
import csv
import pandas as pd

MATRIX_SIZES = [100, 250, 500, 1000, 1500, 2000, 3000, 5000]
BLOCK_SIZES  = [8, 16, 32]
TILED_MODES  = {"global": 0, "tiled": 1}

A_FILE      = "matrixA.txt"
B_FILE      = "matrixB.txt"
C_FILE      = "matrixC.txt"
RESULTS_CSV = "results.csv"

LAB1_CSV = r"D:/Dev/parallel programming/lab1-single_thread/results.csv"


def load_cpu_baseline(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"  ⚠️  Файл lab1 не найден: {path}")
        return {}
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        n_col = next((c for c in df.columns if c.lower() in ("size_n", "n", "size")), None)
        t_col = next((c for c in df.columns if "cpp" in c.lower()), None)
        if n_col is None or t_col is None:
            print(f"  ⚠️  Не удалось найти нужные колонки. Найдены: {df.columns.tolist()}")
            return {}
        baseline = {int(row[n_col]): float(row[t_col]) for _, row in df.iterrows()}
        print(f"  ✓ CPU baseline загружена: {len(baseline)} точек")
        return baseline
    except Exception as e:
        print(f"  ⚠️  Ошибка чтения lab1 CSV: {e}")
        return {}


def compile_cuda(src: str = "main.cu", out: str = "matmul_cuda") -> str:
    exec_name = out + ".exe" if os.name == "nt" else out
    flags = ["-O3", "-std=c++17", "-arch=sm_90"]
    cmd = ["nvcc"] + flags + [src, "-o", exec_name]
    print(f"[Компиляция] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("Ошибка компиляции CUDA кода!")
    print(f"[OK] Скомпилирован: {exec_name}\n")
    return exec_name


def save_matrix(filename: str, matrix: np.ndarray) -> None:
    with open(filename, "w") as f:
        f.write(f"{matrix.shape[0]}\n")
        np.savetxt(f, matrix, fmt="%.5f")


def read_result(filename: str):
    """
    Читает результат из C++.
    Формат файла:
      строка 1: N
      строка 2: time_kernel_ms
      строка 3: time_total_ms
      строка 4: mode
      строка 5: blockSize
      далее:    матрица C
    """
    with open(filename, "r") as f:
        N              = int(f.readline().strip())
        time_kernel_ms = float(f.readline().strip())
        time_total_ms  = float(f.readline().strip())
    return N, time_kernel_ms, time_total_ms


def run_cuda(exec_name: str, block_size: int, tiled: int):
    """Возвращает (time_kernel_ms, time_total_ms)."""
    cmd = [f"./{exec_name}" if os.name != "nt" else exec_name, str(block_size), str(tiled)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка выполнения:\n{result.stderr}")
    _, t_kernel, t_total = read_result(C_FILE)
    return t_kernel, t_total


def main():
    cpu_baseline = load_cpu_baseline(LAB1_CSV)
    cuda_exec    = compile_cuda()

    fieldnames = [
        "size_N", "block_size", "mode",
        "time_kernel_ms",   # только ядро (CUDA Events)
        "time_total_ms",    # H→D + ядро + D→H (chrono)
        "time_cpu_ms",
        "speedup_kernel",   # CPU / kernel  (показывает мощь GPU)
        "speedup_total",    # CPU / total   (честное сравнение)
    ]
    rows = []
    generated = set()

    experiments = [
        (N, bs, mn, tv)
        for N in MATRIX_SIZES
        for bs in BLOCK_SIZES
        for mn, tv in TILED_MODES.items()
    ]
    total = len(experiments)

    for done, (N, bs, mode_name, tiled_val) in enumerate(experiments, 1):
        print(f"\n{'='*62}")
        print(f"  [{done}/{total}]  N={N}x{N}  |  block={bs}x{bs}  |  mode={mode_name}")
        print(f"{'='*62}")

        if N not in generated:
            A = np.random.uniform(-10, 10, (N, N))
            B = np.random.uniform(-10, 10, (N, N))
            save_matrix(A_FILE, A)
            save_matrix(B_FILE, B)
            generated.add(N)

        t_cpu = cpu_baseline.get(N)

        try:
            t_kernel, t_total = run_cuda(cuda_exec, bs, tiled_val)
            t_kernel = round(t_kernel, 6)
            t_total  = round(t_total,  6)

            sp_kernel = round(t_cpu / t_kernel, 4) if t_cpu else None
            sp_total  = round(t_cpu / t_total,  4) if t_cpu else None

            print(f"  Ядро   : {t_kernel:.4f} мс")
            print(f"  Полное : {t_total:.4f} мс  (H→D + ядро + D→H)")
            print(f"  CPU    : {t_cpu} мс" if t_cpu else "  CPU   : нет данных")
            print(f"  Speedup (ядро): {sp_kernel}x  |  Speedup (полный): {sp_total}x"
                  if sp_kernel else "  Speedup: —")

            rows.append({
                "size_N": N, "block_size": bs, "mode": mode_name,
                "time_kernel_ms": t_kernel,
                "time_total_ms":  t_total,
                "time_cpu_ms":    t_cpu,
                "speedup_kernel": sp_kernel,
                "speedup_total":  sp_total,
            })

        except Exception as e:
            print(f"  ⚠️  Пропуск (ошибка): {e}")
            rows.append({
                "size_N": N, "block_size": bs, "mode": mode_name,
                "time_kernel_ms": None, "time_total_ms": None,
                "time_cpu_ms": t_cpu,
                "speedup_kernel": None, "speedup_total": None,
            })

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*62}")
    print(f"  Готово. Результаты в: {RESULTS_CSV}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()