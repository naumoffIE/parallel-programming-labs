#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <cuda_runtime.h>

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

#define CUDA_KERNEL_CHECK()                                                     \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            cerr << "Kernel launch error in " << __FILE__                       \
                 << ":" << __LINE__ << " — " << cudaGetErrorString(err) << endl;\
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
        err = cudaDeviceSynchronize();                                          \
        if (err != cudaSuccess) {                                               \
            cerr << "Kernel exec error in " << __FILE__                         \
                 << ":" << __LINE__ << " — " << cudaGetErrorString(err) << endl;\
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
                 double time_kernel_ms,
                 double time_total_ms,
                 const string& mode,
                 int blockSize)
{
    ofstream f(filename);
    if (!f.is_open()) throw runtime_error("Не удалось создать файл: " + filename);
    // Строки 1-4 читает Python-харвестер
    f << N << "\n"
      << time_kernel_ms << "\n"
      << time_total_ms  << "\n"
      << mode           << "\n"
      << blockSize      << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            f << fixed << setprecision(4) << C[i * N + j] << " ";
        f << "\n";
    }
}

// ──────────────────────────────────────────────
//  Вспомогательная функция: запуск ядра
// ──────────────────────────────────────────────
void launchKernel(double* dA, double* dB, double* dC,
                  int N, int blockSize, bool tiled)
{
    dim3 block(blockSize, blockSize);
    dim3 grid((N + blockSize - 1) / blockSize,
              (N + blockSize - 1) / blockSize);

    if (tiled) {
        switch (blockSize) {
            case  8: matMulTiledKernel< 8><<<grid, block>>>(dA, dB, dC, N); break;
            case 16: matMulTiledKernel<16><<<grid, block>>>(dA, dB, dC, N); break;
            case 32: matMulTiledKernel<32><<<grid, block>>>(dA, dB, dC, N); break;
            default: matMulKernel<<<grid, block>>>(dA, dB, dC, N); break;
        }
    } else {
        matMulKernel<<<grid, block>>>(dA, dB, dC, N);
    }
    // Проверяем ошибку запуска и выполнения ядра.
    // Без этого программа молча продолжает работу, а C остаётся нулями.
    CUDA_KERNEL_CHECK();
}

// ──────────────────────────────────────────────
//  Запуск CUDA: возвращает {time_kernel_ms, time_total_ms}
//  time_kernel — только ядро (CUDA Events)
//  time_total  — H→D + ядро + D→H (chrono на хосте)
// ──────────────────────────────────────────────
pair<double,double> runCUDA(const vector<double>& A,
                             const vector<double>& B,
                             vector<double>&       C,
                             int N, int blockSize, bool tiled)
{
    size_t bytes = (size_t)N * N * sizeof(double);
    double *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // ── Прогревочный запуск (инициализация контекста CUDA) ──────────
    // Запускаем ядро один раз с минимальными данными, чтобы первый
    // "настоящий" замер не был загрязнён инициализацией драйвера.
    {
        double *wA, *wB, *wC;
        CUDA_CHECK(cudaMalloc(&wA, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&wB, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&wC, sizeof(double)));
        dim3 wb(1,1), wg(1,1);
        matMulKernel<<<wg,wb>>>(wA, wB, wC, 1);
        CUDA_KERNEL_CHECK();
        cudaFree(wA); cudaFree(wB); cudaFree(wC);
    }

    // ── Полный замер: H→D + ядро + D→H ─────────────────────────────
    auto t_total_start = chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // ── Замер только ядра (CUDA Events) ─────────────────────────────
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));
    launchKernel(dA, dB, dC, N, blockSize, tiled); // внутри: CUDA_KERNEL_CHECK()
    CUDA_CHECK(cudaEventRecord(ev1));
    // CUDA_KERNEL_CHECK внутри launchKernel уже сделал DeviceSync,
    // поэтому ядро гарантированно завершено к этому моменту.
    // EventSynchronize нужен только чтобы ev1 был зафиксирован на хосте.
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));

    // D→H — тоже часть полного времени
    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    auto t_total_end = chrono::high_resolution_clock::now();
    double total_ms = chrono::duration<double, milli>(t_total_end - t_total_start).count();

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    return { static_cast<double>(kernel_ms), total_ms };
}

// ──────────────────────────────────────────────
//  main
//  Аргументы: <block_size> [tiled=0|1]
//  Пример:    ./matmul_cuda 16 1
// ──────────────────────────────────────────────
int main(int argc, char* argv[]) {
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

        auto [time_kernel_ms, time_total_ms] = runCUDA(A, B, C, N, blockSize, tiled);

        string mode = tiled ? "tiled" : "global";
        writeResult("matrixC.txt", C, N, time_kernel_ms, time_total_ms, mode, blockSize);

        cout << "OK | N=" << N
             << " | block=" << blockSize << "x" << blockSize
             << " | mode=" << mode
             << " | kernel=" << fixed << setprecision(4) << time_kernel_ms << " ms"
             << " | total="  << fixed << setprecision(4) << time_total_ms  << " ms\n";

    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }
    return 0;
}
