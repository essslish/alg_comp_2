"""
Модуль для выполнения прямого и обратного 2D-DCT (DCT-II) на блоках размера N×N.

Класс DCT2D позволяет:
- вычислять прямое DCT-II 2D;
- вычислять обратное IDCT-II 2D;
- проверять точность восстановления (максимальная ошибка);
- оценивать энергоёмкость (доля энергии в DC-компоненте).
"""

import numpy as np


class DCT2D:
    """
    Класс для вычисления двумерного дискретного косинусного преобразования (DCT-II) и обратного преобразования (IDCT-II)
    для квадратных блоков размера N×N.

    Атрибуты:
        block_size (int): размер блока (N).
        D (np.ndarray): матрица преобразования размера (N×N).
    """

    def __init__(self, block_size: int):
        """
        Инициализация матрицы DCT-II.

        :param block_size: размер блока N (должно быть положительным).
        """
        if block_size <= 0:
            raise ValueError("block_size должен быть положительным целым числом")
        self.block_size = block_size
        self.D = self._create_dct_matrix(block_size)

    @staticmethod
    def _create_dct_matrix(N: int) -> np.ndarray:
        """
        Создаёт матрицу D для DCT-II инициализацией коэффициентов α и косинусных функций.

        :param N: размер блока.
        :return: DCT-матрица размера (N×N).
        """
        # Вектор нормировок α[k]
        alpha = np.array([np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N) for k in range(N)])
        # Индексы
        n = np.arange(N)
        k = n.reshape((N, 1))
        # Косинусный аргумент
        phi = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
        # Матрица D: α[k] * cos(...)
        return (alpha.reshape((N, 1)) * phi).astype(np.float64)

    def forward(self, block: np.ndarray) -> np.ndarray:
        """
        Вычисляет прямое DCT-II 2D для блока.

        :param block: входной блок (N×N), любой числовой тип.
        :return: массив коэффициентов DCT того же размера, dtype float64.
        """
        arr = np.asarray(block, dtype=np.float64)
        if arr.shape != (self.block_size, self.block_size):
            raise ValueError(f"Входной блок должен иметь форму ({self.block_size}, {self.block_size})")
        # C = D · f · D^T
        return self.D.dot(arr).dot(self.D.T)

    def inverse(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Вычисляет обратное DCT (IDCT-II) для матрицы коэффициентов.

        :param coeffs: DCT-коэффициенты (N×N), dtype float.
        :return: восстановленный блок (N×N), dtype float64.
        """
        c = np.asarray(coeffs, dtype=np.float64)
        if c.shape != (self.block_size, self.block_size):
            raise ValueError(f"Матрица коэффициентов должна иметь форму ({self.block_size}, {self.block_size})")
        # f = D^T · C · D
        return self.D.T.dot(c).dot(self.D)

    def reconstruct(self, block: np.ndarray) -> np.ndarray:
        """
        Прямое и обратное преобразование: f → C → f'.

        :param block: исходный блок (N×N).
        :return: восстановленный блок (N×N).
        """
        coeffs = self.forward(block)
        return self.inverse(coeffs)

    def reconstruction_error(self, block: np.ndarray) -> float:
        """
        Оценивает максимальную абсолютную ошибку при реконструкции блока.

        :param block: исходный блок (N×N).
        :return: максимальная |f' - f|.
        """
        original = np.asarray(block, dtype=np.float64)
        reconstructed = self.reconstruct(original)
        # Абсолютная погрешность
        return float(np.max(np.abs(reconstructed - original)))

    def dc_energy_ratio(self, coeffs: np.ndarray) -> float:
        """
        Вычисляет долю энергии (суммы квадратов) в DC-коэффициенте C[0,0]
        относительно полной энергии всех коэффициентов.

        :param coeffs: DCT-коэффициенты (N×N).
        :return: значение в диапазоне [0, 1].
        """
        c = np.asarray(coeffs, dtype=np.float64)
        total_energy = np.sum(c ** 2)
        if total_energy == 0:
            return 0.0
        dc_energy = c[0, 0] ** 2
        return float(dc_energy / total_energy)


if __name__ == "__main__":
    # Пример использования и тестирование точности для случайного блока
    N = 8
    dct = DCT2D(block_size=N)

    # Генерируем тестовый блок (например, значение пикселя от 0 до 255)
    block = np.random.randint(0, 256, size=(N, N)).astype(np.float64)
    coeffs = dct.forward(block)
    restored = dct.inverse(coeffs)

    err = dct.reconstruction_error(block)
    ratio = dct.dc_energy_ratio(coeffs)

    print(f"Размер блока: {N}×{N}")
    print(f"Максимальная ошибка восстановления: {err:.6f}")
    print(f"Доля энергии в DC-коэффициенте: {ratio * 100:.2f}%")
