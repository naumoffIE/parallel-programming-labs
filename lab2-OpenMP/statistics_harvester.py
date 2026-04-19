import numpy as np
import subprocess
import os
import csv

# ──────────────────────────────────────────────
#  Параметры экспериментов
# ──────────────────────────────────────────────
MATRIX_SIZES  = [100, 250, 500, 1000, 1500, 2000, 3000, 5000]
THREAD_COUNTS = [1, 2, 4, 8, 12, 16, 24, 32, 48]

A_FILE      = "matrixA.txt"
B_FILE      = "matrixB.txt"
C_FILE      = "matrixC.txt"
RESULTS_CSV = "results.csv"

# ──────────────────────────────────────────────
#  Вспомогательные функции
# ──────────────────────────────────────────────

def compile_cpp(src: str = "main.cpp", out: str = "matmul") -> str:
    """
    Компилирует C++ файл с поддержкой OpenMP и оптимизациями.
    Возвращает имя исполняемого файла.
    """
    exec_name = out + ".exe" if os.name == "nt" else out

    compiler = "g++"
    omp_flag = "-fopenmp"

    flags = [
        "-O3",            # (как в первой лабе)
        "-std=c++17",
    ]

    cmd = [compiler] + flags + omp_flag.split() + [src, "-o", exec_name]

    print(f"[Компиляция] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("Ошибка компиляции C++ кода!")
    print(f"[OK] Скомпилирован: {exec_name}\n")
    return exec_name


def save_matrix(filename: str, matrix: np.ndarray) -> None:
    with open(filename, "w") as f:
        f.write(f"{matrix.shape[0]}\n")
        np.savetxt(f, matrix, fmt="%.4f")


def read_cpp_result(filename: str):
    """Возвращает (N, time_ms, matrix)."""
    with open(filename, "r") as f:
        N       = int(f.readline().strip())
        time_ms = float(f.readline().strip())
        mat     = np.loadtxt(f)
    return N, time_ms, mat


def run_cpp(exec_name: str, threads: int) -> float:
    """Запускает C++ исполняемый файл с заданным числом потоков.
    Возвращает время выполнения (мс) из файла результата."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    exec_cmd = f"./{exec_name}" if os.name != "nt" else exec_name
    result = subprocess.run([exec_cmd], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C++ завершился с ошибкой:\n{result.stderr}")

    _, time_ms, _ = read_cpp_result(C_FILE)
    return time_ms


# ──────────────────────────────────────────────
#  Основная логика
# ──────────────────────────────────────────────

def main():
    exec_name = compile_cpp()

    fieldnames = [
        "size_N",
        "threads",
        "time_cpp_ms",
    ]

    rows = []

    total = len(MATRIX_SIZES) * len(THREAD_COUNTS)
    done  = 0

    for N in MATRIX_SIZES:
        print(f"\n{'='*55}")
        print(f"  Размер матрицы: {N}x{N}")
        print(f"{'='*55}")

        A = np.random.uniform(-10, 10, (N, N))
        B = np.random.uniform(-10, 10, (N, N))
        save_matrix(A_FILE, A)
        save_matrix(B_FILE, B)


        for T in THREAD_COUNTS:
            done += 1
            print(f"\n  [{done}/{total}] Потоки: {T}")

            try:
                time_cpp = run_cpp(exec_name, T)

                rows.append({
                    "size_N":            N,
                    "threads":           T,
                    "time_cpp_ms":       round(time_cpp,   7),
                })

            except Exception as e:
                print(f"  ⚠️  Пропуск (ошибка): {e}")
                rows.append({
                    "size_N":            N,
                    "threads":           T,
                    "time_cpp_ms":       None,
                })

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*55}")
    print(f"  Все эксперименты завершены.")
    print(f"  Результаты сохранены в: {RESULTS_CSV}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()