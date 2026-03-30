import numpy as np
import subprocess
import os
import sys
import time

# Файлы для обмена данными
A_FILE = "matrixA.txt"
B_FILE = "matrixB.txt"
C_FILE = "matrixC.txt"
RESULTS_FILE = "results.csv" # Файл для накопления статистики

def save_matrix_to_file(filename, matrix):
    with open(filename, 'w') as f:
        f.write(f"{matrix.shape[0]}\n")
        np.savetxt(f, matrix, fmt='%.4f')

def read_result_from_file(filename):
    with open(filename, 'r') as f:
        N = int(f.readline().strip())
        time_ms = float(f.readline().strip())
        matrix_data = np.loadtxt(f)
    return N, time_ms, matrix_data

def main():
    # 1. Читаем размер матрицы из аргументов командной строки
    # Если аргумент не передан, берем размер 500 по умолчанию
    if len(sys.argv) > 1:
        try:
            MATRIX_SIZE = int(sys.argv[1])
        except ValueError:
            print("Ошибка: Размер матрицы должен быть целым числом.")
            sys.exit(1)
    else:
        MATRIX_SIZE = 500

    print(f"--- ЗАПУСК: N = {MATRIX_SIZE} ---")
    
    A = np.random.uniform(-10, 10, (MATRIX_SIZE, MATRIX_SIZE))
    B = np.random.uniform(-10, 10, (MATRIX_SIZE, MATRIX_SIZE))

    save_matrix_to_file(A_FILE, A)
    save_matrix_to_file(B_FILE, B)

    # Компиляция (выполняется только если файла matmul еще нет)
    exec_name = "matmul.exe" if os.name == "nt" else "matmul"
    if not os.path.exists(exec_name):
        print("Компиляция C++ кода...")
        subprocess.run(["g++", "-O3", "main.cpp", "-o", "matmul"], check=True)

    # Запуск C++
    print("Выполнение C++ программы...")
    exec_cmd = f"./{exec_name}" if os.name != "nt" else exec_name
    subprocess.run([exec_cmd], check=True)

    # Верификация и замер времени в Python
    print("Верификация в NumPy...")
    start_py = time.time()
    C_expected = np.dot(A, B)
    time_py_ms = (time.time() - start_py) * 1000  # переводим в миллисекунды

    # Чтение результата C++
    N_cpp, time_cpp, C_cpp = read_result_from_file(C_FILE)

    # Проверка
    is_correct = np.allclose(C_expected, C_cpp, atol=0.1)

    if is_correct:
        print(f"✅ УСПЕХ! Размер: {N_cpp}x{N_cpp} | Время C++: {time_cpp:.2f} мс | Время NumPy: {time_py_ms:.2f} мс\n")
        
        # ДОПИСЫВАЕМ результат в CSV файл
        # Флаг 'a' (append) означает "добавить в конец файла, не удаляя старое"
        file_exists = os.path.isfile(RESULTS_FILE)
        with open(RESULTS_FILE, "a") as f:
            # Если файла не было, пишем заголовок таблицы
            if not file_exists:
                f.write("Size_N,Time_CPP_ms,Time_Python_ms\n")
            # Пишем сами данные
            f.write(f"{N_cpp},{time_cpp:.2f},{time_py_ms:.2f}\n")
    else:
        print("❌ ОШИБКА: Результаты не совпадают!\n")

if __name__ == "__main__":
    main()