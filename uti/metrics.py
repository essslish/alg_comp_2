import numpy as np

from utils.logger import logger


def calculate_metrics(original: bytes, compressed: bytes, decompressed: bytes) -> dict:
    original_size = len(original)
    compressed_size = len(compressed)
    compression_ratio = round(original_size / compressed_size, 6) if original_size else 0
    decompressed_size = len(decompressed)
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "decompressed_size": decompressed_size,
        "compression_ratio": compression_ratio  
    }


def entropy(data: bytes) -> float:
    """
    Вычисляет энтропию байтовой строки (Shannon entropy, бит/символ).
    :param data: входные данные в виде байтов
    :return: энтропия (float)
    """
    if not data:
        return 0.0

    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))


def save_raw_data_to_file(data: bytes, path: str):
    with open(path, "wb") as f:
        f.write(data)
    logger.info(f"Данные сохранены в файл: {path}")
