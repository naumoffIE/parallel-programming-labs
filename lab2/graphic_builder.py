import pandas as pd
import matplotlib.pyplot as plt

# 1. Читаем ваш файл (убедитесь, что название совпадает)
file_name = "results.csv" # или full_statistics.csv
df = pd.read_csv(file_name)
print(f"✓ Загружено {len(df)} записей из {file_name}")
print(f"  Размеры матриц (N): {sorted(df['size_N'].unique())}")
print(f"  Потоки: {sorted(df['threads'].unique())}")

# 2. Считаем ускорение S = T(1) / T(p)
# Мы группируем по размеру N и делим время первого замера (где 1 поток) на все остальные
df['speedup'] = df.groupby('size_N')['time_cpp_ms'].transform(lambda x: x.iloc[0] / x)

# Создаем холст для двух графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- ГРАФИК 1: Время выполнения ---
# Берем только крупные матрицы, чтобы масштаб не "сплющился"
large_matrices = df[df['size_N'] >= 1000]
for n in large_matrices['size_N'].unique():
    subset = large_matrices[large_matrices['size_N'] == n]
    ax1.plot(subset['threads'], subset['time_cpp_ms'], marker='o', label=f'N={n}')

ax1.set_title('Падение времени при распараллеливании', fontsize=14)
ax1.set_xlabel('Количество потоков')
ax1.set_ylabel('Время (мс)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- ГРАФИК 2: Ускорение (Speedup) ---
for n in df['size_N'].unique():
    if n < 500: continue # Маленькие матрицы только портят график ускорения шумом
    subset = df[df['size_N'] == n]
    ax2.plot(subset['threads'], subset['speedup'], marker='s', label=f'N={n}')

# Рисуем линию идеального ускорения (y = x)
max_threads = df['threads'].max()
ax2.plot([1, max_threads], [1, max_threads], color='red', linestyle='--', label='Идеал (S=p)')

ax2.set_title('Коэффициент ускорения (Speedup)', fontsize=14)
ax2.set_xlabel('Количество потоков')
ax2.set_ylabel('Во сколько раз быстрее')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Сохраняем график в файл
output_file = "analysis_plot.png"
plt.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"✓ График сохранён: {output_file}")

# Показываем график в окне
plt.show()
print("✓ Скрипт выполнен успешно!")