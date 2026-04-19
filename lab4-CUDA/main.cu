#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <cuda_runtime.h>
#include <windows.h>

using namespace std;

// ──────────────────────────────────────────────
//  Вспомогательный макрос проверки ошибок CUDA
// ──────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " — "   \
                 << cudaGetErrorString(err) << endl;                            \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ──────────────────────────────────────────────
//  CUDA-ядро: базовое глобальное перемножение
// ──────────────────────────────────────────────
__global__ void matMulKernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double*       __restrict__ C,
                              int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ──────────────────────────────────────────────
//  CUDA-ядро: с разделяемой памятью (тайловое)
// ──────────────────────────────────────────────
template <int TILE>
__global__ void matMulTiledKernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double*       __restrict__ C,
                                   int N)
{
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// ──────────────────────────────────────────────
//  Последовательное перемножение (CPU-базовая линия)
// ──────────────────────────────────────────────
void matMulCPU(const vector<double>& A,
               const vector<double>& B,
               vector<double>&       C,
               int N)
{
    fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            double tmp = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += tmp * B[k * N + j];
        }
}

// ──────────────────────────────────────────────
//  Чтение / запись
// ──────────────────────────────────────────────
vector<double> readMatrix(const string& filename, int& N) {
    ifstream f(filename);
    if (!f.is_open()) throw runtime_error("Не удалось открыть файл: " + filename);
    f >> N;
    vector<double> m(N * N);
    for (auto& v : m) f >> v;
    return m;
}

void writeResult(const string& filename,
                 const vector<double>& C,
                 int N,
                 double time_ms,
                 const string& mode,
                 int blockSize)
{
    ofstream f(filename);
    if (!f.is_open()) throw runtime_error("Не удалось создать файл: " + filename);
    f << N << "\n" << time_ms << "\n" << mode << "\n" << blockSize << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            f << fixed << setprecision(4) << C[i * N + j] << " ";
        f << "\n";
    }
}

// ──────────────────────────────────────────────
//  Запуск CUDA с нужным размером блока
// ──────────────────────────────────────────────
double runCUDA(const vector<double>& A,
               const vector<double>& B,
               vector<double>&       C,
               int N, int blockSize, bool tiled)
{
    size_t bytes = (size_t)N * N * sizeof(double);
    double *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    dim3 block(blockSize, blockSize);
    dim3 grid((N + blockSize - 1) / blockSize,
              (N + blockSize - 1) / blockSize);

    // Измерение через CUDA Events (точнее chrono для GPU)
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));

    if (tiled) {
        // Выбираем шаблонный инстанс по blockSize
        switch (blockSize) {
            case  8: matMulTiledKernel< 8><<<grid, block>>>(dA, dB, dC, N); break;
            case 16: matMulTiledKernel<16><<<grid, block>>>(dA, dB, dC, N); break;
            case 32: matMulTiledKernel<32><<<grid, block>>>(dA, dB, dC, N); break;
            default: matMulKernel<<<grid, block>>>(dA, dB, dC, N); break;
        }
    } else {
        matMulKernel<<<grid, block>>>(dA, dB, dC, N);
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    return static_cast<double>(ms);
}

// ──────────────────────────────────────────────
//  main
//  Аргументы: <block_size> [tiled=0|1]
//  Пример:    ./matmul_cuda 16 1
// ──────────────────────────────────────────────
int main(int argc, char* argv[]) {

    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);


    int  blockSize = 16;
    bool tiled     = true;

    if (argc > 1) blockSize = atoi(argv[1]);
    if (argc > 2) tiled     = (atoi(argv[2]) != 0);

    // Проверка: blockSize должен быть кратным 32 или степенью 2 ≤ 32
    if (blockSize <= 0 || blockSize > 32) {
        cerr << "Допустимые размеры блока: 8, 16, 32\n";
        return 1;
    }

    try {
        int N_A, N_B;
        auto A = readMatrix("matrixA.txt", N_A);
        auto B = readMatrix("matrixB.txt", N_B);

        if (N_A != N_B) throw runtime_error("Матрицы должны быть квадратными и одинакового размера!");
        int N = N_A;

        vector<double> C(N * N, 0.0);

        double time_ms = runCUDA(A, B, C, N, blockSize, tiled);

        string mode = tiled ? "tiled" : "global";
        writeResult("matrixC.txt", C, N, time_ms, mode, blockSize);

        cout << "OK | N=" << N
             << " | block=" << blockSize << "x" << blockSize
             << " | mode=" << mode
             << " | time=" << fixed << setprecision(4) << time_ms << " ms\n";

    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }
    return 0;
}
