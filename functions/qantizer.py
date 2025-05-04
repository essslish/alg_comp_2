"""
Модуль для работы с матрицами квантования JPEG:
- опорные (базовые) таблицы ITU-T81 для яркостного (luminance) и цветоразностного (chrominance) каналов;
- масштабирование таблиц в зависимости от уровня качества;
- квантование и обратное деквантование DCT-коэффициентов.
"""

from typing import Tuple

import numpy as np


class Quantizer:
    """
    Класс для генерации и работы с матрицами квантования JPEG.

    Методы позволяют:
    - получать масштабированные таблицы квантования для заданного quality ∈ [0…100];
    - квантовать DCT-коэффициенты;
    - обратно деквантовать.
    """

    # Базовые таблицы квантования ITU-T81, стр. 143
    _BASE_LUMINANCE = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=np.int32)

    _BASE_CHROMINANCE = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ], dtype=np.int32)

    @staticmethod
    def _scale_factor(quality: int) -> int:
        """
        Вычисляет фактор масштабирования таблицы по правилу JPEG:
            if quality < 50: scale = 5000 / quality
            else:            scale = 200 - 2 * quality

        :param quality: уровень качества ∈ [1…100] (качество=0 интерпретировать как 1).
        :return: целочисленный scale.
        """
        q = max(1, min(quality, 100))
        if q < 50:
            return 5000 // q
        else:
            return 200 - 2 * q

    @classmethod
    def get_quant_tables(cls, quality: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает масштабированные матрицы квантования для luminance и chrominance.

        :param quality: уровень качества ∈ [0…100].
        :return: кортеж (table_y, table_cbcr), оба размера 8×8, dtype int32.
        """
        scale = cls._scale_factor(quality)

        def scale_table(base: np.ndarray) -> np.ndarray:
            # Применяем формулу: T' = floor((base * scale + 50) / 100), затем обрезаем в [1…255]
            tmp = (base.astype(np.int32) * scale + 50) // 100
            tmp = np.clip(tmp, 1, 255)
            return tmp

        return scale_table(cls._BASE_LUMINANCE), scale_table(cls._BASE_CHROMINANCE)

    @staticmethod
    def quantize(
            coeffs: np.ndarray,
            table: np.ndarray
    ) -> np.ndarray:
        """
        Квантует матрицу DCT-коэффициентов по заданной таблице.

        :param coeffs: DCT-коэффициенты (N×N), dtype float или int.
        :param table:  таблица квантования того же размера (N×N), dtype int.
        :return: целочисленная матрица квантованных коэффициентов, dtype int32.
        """
        if coeffs.shape != table.shape:
            raise ValueError("Размер coeffs и table должен совпадать")
        # Делим и округляем к ближайшему целому
        return np.round(coeffs / table).astype(np.int32)

    @staticmethod
    def dequantize(
            qcoeffs: np.ndarray,
            table: np.ndarray
    ) -> np.ndarray:
        """
        Обратное деквантование: восстанавливает DCT-коэффициенты из квантованных.

        :param qcoeffs: квантованные коэффициенты (N×N), dtype int.
        :param table:   таблица квантования того же размера (N×N), dtype int.
        :return: матрица DCT-коэффициентов, dtype float64.
        """
        if qcoeffs.shape != table.shape:
            raise ValueError("Размер qcoeffs и table должен совпадать")
        return (qcoeffs.astype(np.float64) * table.astype(np.float64))


if __name__ == "__main__":
    # Пример использования
    quality = 75
    qt_y, qt_c = Quantizer.get_quant_tables(quality)
    print(f"Scaled luminance table (Q={quality}):\n{qt_y}\n")
    print(f"Scaled chrominance table (Q={quality}):\n{qt_c}\n")

    # Тест квантования/деквантования случайных DCT-коэфф.
    import numpy as np

    block = np.random.randn(8, 8) * 100
    q = Quantizer.quantize(block, qt_y)
    restored = Quantizer.dequantize(q, qt_y)
    # Проверка обратимости масштаба (±0.5 при квантовании)
    diff = np.abs(restored - block)
    print(f"Максимальная погрешность квантования ⟷ деквантования: {diff.max():.2f}")
