// mpi.h ОБЯЗАТЕЛЬНО первым — иначе конфликт SEEK_SET с <fstream>
#include <mpi.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iomanip>

using namespace std;

// ──────────────────────────────────────────────────────────
//  Генерация случайной матрицы N×N, [-10, 10]
//  Используем rand() — совместимо с любым стандартом
// ──────────────────────────────────────────────────────────
vector<double> generateMatrix(int N, unsigned int seed) {
    srand(seed);
    vector<double> mat(N * N);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = -10.0 + 20.0 * (double)rand() / RAND_MAX;
    }
    return mat;
}

// ──────────────────────────────────────────────────────────
//  Дописать строку в CSV (только rank == 0)
// ──────────────────────────────────────────────────────────
void appendResult(int N, int processes, double time_ms) {
    const char* filename = "results_mpi.csv";

    // Проверяем, нужен ли заголовок
    bool write_header = false;
    {
        ifstream check(filename);
        write_header = !check.good();
    }

    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Не удалось открыть файл: " << filename << "\n";
        return;
    }

    if (write_header)
        file << "size_N,processes,time_mpi_ms\n";

    file << N << "," << processes << ","
         << fixed << setprecision(6) << time_ms << "\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ── Аргумент: размер матрицы N ────────────────────────────────
    if (argc < 2) {
        if (rank == 0)
            cerr << "Использование: mpirun -np <P> ./matmulMPI <N>\n";
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0)
            cerr << "Ошибка: N должен быть положительным\n";
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
        cout << "Запуск: N=" << N << ", процессов=" << size << "\n";

    // ── Генерация матриц на root, рассылка B ──────────────────────
    vector<double> A, B(N * N);

    if (rank == 0) {
        A = generateMatrix(N, 42);
        B = generateMatrix(N, 137);
    }

    MPI_Bcast(&B[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ── Распределение строк A с учётом остатка ────────────────────
    int base_rows = N / size;
    int remainder = N % size;

    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        int rows_p    = base_rows + (p < remainder ? 1 : 0);
        sendcounts[p] = rows_p * N;
        displs[p]     = offset;
        offset       += rows_p * N;
    }

    int my_rows = base_rows + (rank < remainder ? 1 : 0);
    vector<double> local_A(my_rows * N);

    MPI_Scatterv(
        rank == 0 ? &A[0] : NULL,
        &sendcounts[0], &displs[0], MPI_DOUBLE,
        &local_A[0], my_rows * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // ── Вычисление (порядок i-k-j, оптимизация кэша) ─────────────
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    vector<double> local_C(my_rows * N, 0.0);
    for (int i = 0; i < my_rows; ++i) {
        for (int k = 0; k < N; ++k) {
            double temp = local_A[i * N + k];
            for (int j = 0; j < N; ++j)
                local_C[i * N + j] += temp * B[k * N + j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    // ── Сбор на root (матрицу не сохраняем, только время) ─────────
    vector<double> C;
    if (rank == 0) C.resize(N * N);

    MPI_Gatherv(
        &local_C[0], my_rows * N, MPI_DOUBLE,
        rank == 0 ? &C[0] : NULL,
        &sendcounts[0], &displs[0], MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // ── Запись результата ─────────────────────────────────────────
    if (rank == 0) {
        double time_ms = (t_end - t_start) * 1000.0;
        cout << fixed << setprecision(4)
             << "Время: " << time_ms << " мс"
             << " | N=" << N
             << " | Процессов=" << size << "\n";
        appendResult(N, size, time_ms);
    }

    MPI_Finalize();
    return 0;
}