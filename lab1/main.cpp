#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <iomanip>

using namespace std;

// Функция для чтения матрицы из файла
vector<double> readMatrix(const string& filename, int& N) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }
    file >> N;
    vector<double> matrix(N * N);
    for (int i = 0; i < N * N; ++i) {
        file >> matrix[i];
    }
    file.close();
    return matrix;
}

// Функция для записи результата в файл
void writeMatrix(const string& filename, const vector<double>& matrix, int N, double time_ms) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось создать файл: " + filename);
    }
    // Выходные данные: объем задачи, время выполнения, значения
    file << N << "\n";
    file << time_ms << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file << fixed << setprecision(4) << matrix[i * N + j] << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    int N_A, N_B;
    vector<double> A, B, C;

    try {
        // Чтение исходных данных
        A = readMatrix("matrixA.txt", N_A);
        B = readMatrix("matrixB.txt", N_B);

        if (N_A != N_B) {
            throw runtime_error("Матрицы должны быть одинакового размера!");
        }
        
        int N = N_A;
        C.assign(N * N, 0.0);

        // Начало отсчета времени
        auto start = chrono::high_resolution_clock::now();

        // Перемножение матриц (порядок i-k-j для оптимизации кэша)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                double temp = A[i * N + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += temp * B[k * N + j];
                }
            }
        }

        // Конец отсчета времени
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;

        // Запись результатов
        writeMatrix("matrixC.txt", C, N, duration.count());

        cout << "Успешно. Время выполнения: " << duration.count() << " мс. Объем: " << N << "x" << N << endl;

    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }

    return 0;
}