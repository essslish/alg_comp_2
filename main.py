"""
Главный скрипт:
 - загружает тестовые изображения,
 - для каждого формата (оригинал, grayscale, bw, dithered) и для каждого уровня качества
   выполняет сжатие/декомпрессию,
 - сохраняет исходные, сжатые и восстановленные файлы,
 - выводит метрики сжатия и сохраняет их в JSON.
"""

import json
import os
from pathlib import Path

from PIL import Image

from compressor.jpeg_compressor import JPEGCompressor
from utils.ImageConverter import ImageConverter
from utils.file_utils import load_test_files
from utils.metrics import calculate_metrics, save_raw_data_to_file

# Папки
INPUT_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Уровни качества для тестов
QUALITY_LEVELS = [0, 20, 40, 60, 80, 100]


def process_variant(img: Image.Image, name: str, q: int) -> dict:
    """
    Выполняет compress/decompress для одного Image и сохраняет результаты.
    Возвращает словарь с метриками, включая имя, формат и качество.
    """
    compressor = JPEGCompressor(quality=q)

    # Сохраняем исходный файл (PNG)
    orig_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    img.save(orig_path)

    # Сжатие
    comp_bytes = compressor.compress(img)
    comp_path = os.path.join(OUTPUT_DIR, f"{name}_q{q}.jpgl")
    save_raw_data_to_file(comp_bytes, comp_path)

    # Декомпрессия
    decomp_img = compressor.decompress(comp_bytes)
    decomp_path = os.path.join(OUTPUT_DIR, f"{name}_restored_q{q}.png")
    decomp_img.save(decomp_path)

    # Метрики
    raw_bytes = img.tobytes()
    decompressed_bs = decomp_img.tobytes()
    metrics = calculate_metrics(raw_bytes, comp_bytes, decompressed_bs)
    metrics.update({
        "name": name,
        "quality": compressor.quality
    })
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    # Загружаем все файлы из data
    files = load_test_files(INPUT_DIR)
    all_metrics = []

    for q in QUALITY_LEVELS:
        for filename in files.keys():
            img_path = Path(INPUT_DIR) / filename
            img = Image.open(img_path).convert("RGB")
            base, _ = os.path.splitext(filename)

            # Оригинал
            variant = f"{base}_orig"
            all_metrics.append(process_variant(img, variant, q))

            # Grayscale
            gray = ImageConverter.to_grayscale(img)
            variant = f"{base}_gray"
            all_metrics.append(process_variant(gray, variant, q))

            # BW без дизеринга (переводим в RGB, чтобы compressor принимал)
            bw = ImageConverter.to_bw(img)
            bw_rgb = bw.convert("RGB")
            variant = f"{base}_bw"
            all_metrics.append(process_variant(bw_rgb, variant, q))

            # BW с дизерингом
            dithered = ImageConverter.to_dithered_bw(img)
            dithered_rgb = dithered.convert("RGB")
            variant = f"{base}_dithered"
            all_metrics.append(process_variant(dithered_rgb, variant, q))

    # Сохраняем все метрики в JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
