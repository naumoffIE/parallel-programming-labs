import pandas as pd
import matplotlib.pyplot as plt

file_name = "results.csv" 
df = pd.read_csv(file_name)
print(f"✓ Загружено {len(df)} записей из {file_name}")
print(f"  Размеры матриц (N): {sorted(df['size_N'].unique())}")
print(f"  Потоки: {sorted(df['threads'].unique())}")

df['speedup'] = df.groupby('size_N')['time_cpp_ms'].transform(lambda x: x.iloc[0] / x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

large_matrices = df[df['size_N'] >= 1000]
for n in large_matrices['size_N'].unique():
    subset = large_matrices[large_matrices['size_N'] == n]
    ax1.plot(subset['threads'], subset['time_cpp_ms'], marker='o', label=f'N={n}')

ax1.set_title('Падение времени при распараллеливании', fontsize=14)
ax1.set_xlabel('Количество потоков')
ax1.set_ylabel('Время (мс)')
ax1.legend()
ax1.grid(True, alpha=0.3)

for n in df['size_N'].unique():
    if n < 500: continue 
    subset = df[df['size_N'] == n]
    ax2.plot(subset['threads'], subset['speedup'], marker='s', label=f'N={n}')

max_threads = df['threads'].max()
# ax2.plot([1, max_threads], [1, max_threads], color='red', linestyle='--', label='Идеал (S=p)')

ax2.set_title('Коэффициент ускорения (Speedup)', fontsize=14)
ax2.set_xlabel('Количество потоков')
ax2.set_ylabel('Во сколько раз быстрее')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

output_file = "analysis_plot.png"
plt.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"✓ График сохранён: {output_file}")

plt.show()
print("✓ Скрипт выполнен успешно!")