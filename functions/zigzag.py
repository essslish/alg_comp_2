"""
Модуль для выполнения зигзаг-сканирования (zig-zag) и обратной сборки квадратных блоков N×N.

Класс ZigZag позволяет:
- получить одномерный массив коэффициентов длины N² в порядке зигзаг-обхода;
- восстановить N×N-блок из одномерного массива коэффициентов.
"""

from typing import List, Tuple

import numpy as np


class ZigZag:
    """
    Класс для зигзаг-сканирования и обратного преобразования квадратных матриц размера N×N.

    Атрибуты:
        block_size (int): размер блока (N).
        indices (List[Tuple[int,int]]): упорядоченный список индексов (i,j) для обхода.
    """

    def __init__(self, block_size: int):
        """
        Инициализирует таблицу индексов для зигзаг-обхода.

        :param block_size: размер квадрата N. Должен быть положительным целым.
        """
        if block_size <= 0:
            raise ValueError("block_size должен быть положительным целым числом")
        self.block_size = block_size
        self.indices = self._generate_indices(block_size)

    @staticmethod
    def _generate_indices(n: int) -> List[Tuple[int, int]]:
        """
        Генерирует список индексов для зигзаг-обхода матрицы n×n.

        :param n: размер матрицы.
        :return: список (i, j) длины n*n в порядке обхода.
        """
        indices: List[Tuple[int, int]] = []
        for s in range(2 * n - 1):
            if s % 2 == 0:
                # чётная диагональ: идём снизу вверх
                i_start = min(s, n - 1)
                j_start = s - i_start
                while i_start >= 0 and j_start < n:
                    indices.append((i_start, j_start))
                    i_start -= 1
                    j_start += 1
            else:
                # нечётная диагональ: идём сверху вниз
                j_start = min(s, n - 1)
                i_start = s - j_start
                while j_start >= 0 and i_start < n:
                    indices.append((i_start, j_start))
                    i_start += 1
                    j_start -= 1
        return indices

    def encode(self, block: np.ndarray) -> np.ndarray:
        """
        Преобразует N×N-блок в одномерный массив длины N² по зигзаг-обходу.

        :param block: входная матрица размера (N, N).
        :return: одномерный numpy-массив длины N*N, dtype совпадает с block.dtype.
        """
        arr = np.asarray(block)
        if arr.ndim != 2 or arr.shape != (self.block_size, self.block_size):
            raise ValueError(f"Входной блок должен иметь форму ({self.block_size}, {self.block_size})")
        # Считываем элементы в порядке индексов
        flat = [arr[i, j] for (i, j) in self.indices]
        return np.array(flat, dtype=arr.dtype)

    def decode(self, data: np.ndarray) -> np.ndarray:
        """
        Восстанавливает N×N-блок из одномерного массива длины N², заданного в зигзаг-порядке.

        :param data: одномерный numpy-массив длины N*N.
        :return: восстановленная матрица размера (N, N), dtype совпадает с data.dtype.
        """
        arr = np.asarray(data)
        expected_len = self.block_size * self.block_size
        if arr.ndim != 1 or arr.size != expected_len:
            raise ValueError(f"Входной массив должен быть одномерным длины {expected_len}")
        # Заполняем блок нулями нужного dtype
        block = np.zeros((self.block_size, self.block_size), dtype=arr.dtype)
        for idx, (i, j) in enumerate(self.indices):
            block[i, j] = arr[idx]
        return block


if __name__ == "__main__":
    # Пример использования и тестирование
    N = 4
    zz = ZigZag(block_size=N)

    # Тестовый блок 4×4
    test_block = np.array([
        [1, 2, 6, 7],
        [3, 5, 8, 13],
        [4, 9, 12, 14],
        [10, 11, 15, 16]
    ], dtype=np.int16)

    # Прямое зигзаг-сканирование
    flat = zz.encode(test_block)
    print("ZigZag последовательность:", flat.tolist())

    # Обратное преобразование
    restored = zz.decode(flat)
    print("Восстановленный блок:\n", restored)
