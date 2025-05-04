# plot_metrics.py

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

INPUT_METRICS = "output/metrics.json"
OUTPUT_DIR = "output/plots"


def main():
    # 1. Загрузим метрики
    with open(INPUT_METRICS, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 2. Сгруппируем по «базовому» имени
    groups = defaultdict(list)
    for entry in metrics:
        base = entry["name"]
        # соберём tuple (quality, compressed_size)
        groups[base].append((entry["quality"], entry["compressed_size"]))

    # 3. Для каждой группы построим график
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for base, points in groups.items():
        # Сортируем по quality
        points.sort(key=lambda x: x[0])
        qualities = [q for q, sz in points]
        sizes = [sz for q, sz in points]

        plt.figure()
        plt.plot(qualities, sizes, marker="o")
        plt.title(f"Compression size vs Quality for '{base}'")
        plt.xlabel("Quality")
        plt.ylabel("Compressed size (bytes)")
        plt.grid(True)

        out_path = os.path.join(OUTPUT_DIR, f"{base}_compression_curve.png")
        plt.savefig(out_path)
        print(f"Saved plot: {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
