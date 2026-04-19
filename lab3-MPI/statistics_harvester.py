import numpy as np
import subprocess
import os
import csv

# ──────────────────────────────────────────────
#  Параметры экспериментов
# ──────────────────────────────────────────────
MATRIX_SIZES  = [100, 250, 500, 1000, 1500, 2000, 3000, 5000]
PROCESS_COUNTS = [1, 2, 4, 8, 12, 16, 24, 32, 48]

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

    compiler = "mpic++"

    flags = [
        "-O3",            # (как в первой лабе)
        "-std=c++17",
    ]

    cmd = [compiler] + flags + [src, "-o", exec_name]

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
        np.savetxt(f, matrix, fmt="%.5f")


def read_cpp_result(filename: str):
    """Возвращает (N, time_ms, matrix)."""
    with open(filename, "r") as f:
        N       = int(f.readline().strip())
        time_ms = float(f.readline().strip())
        mat     = np.loadtxt(f)
    return N, time_ms, mat

def run_mpi(exec_name: str, processes: int) -> float:
    """
    Запускает MPI-исполняемый файл с заданным числом процессов.
    Возвращает время выполнения (мс) из файла результата.
    """
    cmd = [
        "mpiexec",
        "--oversubscribe", # нужен, если процессов больше физических ядер
        "-np", str(processes),
        f"./{exec_name}" if os.name != "nt" else exec_name,
    ]
 
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MPI завершился с ошибкой:\n{result.stderr}"
        )
 
    _, time_ms, _ = read_cpp_result(C_FILE)
    return time_ms


# ──────────────────────────────────────────────
#  Основная логика
# ──────────────────────────────────────────────

def main():
    exec_name = compile_cpp()
 
    fieldnames = ["size_N", "processes", "time_mpi_ms"]
    rows = []
 
    total = len(MATRIX_SIZES) * len(PROCESS_COUNTS)
    done  = 0
 
    for N in MATRIX_SIZES:
        print(f"\n{'='*55}")
        print(f"  Размер матрицы: {N}x{N}")
        print(f"{'='*55}")
 
        A = np.random.uniform(-10, 10, (N, N))
        B = np.random.uniform(-10, 10, (N, N))
        save_matrix(A_FILE, A)
        save_matrix(B_FILE, B)
 
        for P in PROCESS_COUNTS:
            done += 1
            print(f"\n  [{done}/{total}] Процессов MPI: {P}")
 
            # Нельзя запустить больше процессов, чем строк в матрице
            if P > N:
                print(f"  ⚠️  Пропуск: процессов ({P}) > строк матрицы ({N})")
                rows.append({
                    "size_N":       N,
                    "processes":    P,
                    "time_mpi_ms":  None,
                })
                continue
 
            try:
                time_mpi = run_mpi(exec_name, P)
 
                rows.append({
                    "size_N":       N,
                    "processes":    P,
                    "time_mpi_ms":  round(time_mpi, 7),
                })
                print(f"  Время: {time_mpi:.5f} мс")
 
            except Exception as e:
                print(f"  ⚠️  Пропуск (ошибка): {e}")
                rows.append({
                    "size_N":       N,
                    "processes":    P,
                    "time_mpi_ms":  None,
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