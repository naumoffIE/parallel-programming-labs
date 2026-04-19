#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <mpi.h>

using namespace std;

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

// Запись результата в файл (вызывается только на root-процессе)
void writeMatrix(const string& filename, const vector<double>& matrix, int N, double time_ms) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось создать файл: " + filename);
    }
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    vector<double> A, B, C;

    if (rank == 0) {
        try {
            int N_A, N_B;
            A = readMatrix("matrixA.txt", N_A);
            B = readMatrix("matrixB.txt", N_B);
            if (N_A != N_B) {
                throw runtime_error("Матрицы должны быть одинакового размера!");
            }
            N = N_A;
        } catch (const exception& e) {
            cerr << "Ошибка: " << e.what() << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_rows  = N / size;
    int remainder  = N % size;

    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        int rows_p = base_rows + (p < remainder ? 1 : 0);
        sendcounts[p] = rows_p * N;
        displs[p]     = offset;
        offset       += rows_p * N;
    }

    int my_rows = base_rows + (rank < remainder ? 1 : 0);

    vector<double> local_A(my_rows * N);
    B.resize(N * N);
    vector<double> local_C(my_rows * N, 0.0);

    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_A.data(), my_rows * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int i = 0; i < my_rows; ++i) {
        for (int k = 0; k < N; ++k) {
            double temp = local_A[i * N + k];
            for (int j = 0; j < N; ++j) {
                local_C[i * N + j] += temp * B[k * N + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(
        local_C.data(), my_rows * N, MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        double time_ms = (t_end - t_start) * 1000.0;
        try {
            writeMatrix("matrixC.txt", C, N, time_ms);
        } catch (const exception& e) {
            cerr << "Ошибка записи: " << e.what() << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cout << "Успешно. Время выполнения: " << time_ms
             << " мс. Объем: " << N << "x" << N
             << ". Процессов: " << size << endl;
    }

    MPI_Finalize();
    return 0;
}