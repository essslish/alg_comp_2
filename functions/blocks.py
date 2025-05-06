from typing import List

import numpy as np


class BlockSplitter:
    """
    Атрибуты:
        block_size (int): размер блока по высоте и ширине (N).
        fill_value (int): значение (0–255) для заполнения паддинга.
    """

    def __init__(self, block_size: int = 8, fill_value: int = 0):
        """
        :param block_size: размер блока (N). Должен быть положительным целым.
        :param fill_value: значение для заполнения краевых пикселов (0–255).
        """
        if block_size <= 0:
            raise ValueError("block_size должен быть положительным целым")
        if not (0 <= fill_value <= 255):
            raise ValueError("fill_value должен быть в диапазоне [0, 255]")
        self.block_size = block_size
        self.fill_value = fill_value

    def convert(
            self,
            data: bytes,
            width: int,
            height: int
    ) -> List[bytes]:
        """
        Разбивает изображение на блоки N×N.

        :param data: байтовый буфер размером width*height, одноканальный (uint8).
        :param width: ширина исходного изображения в пикселях.
        :param height: высота исходного изображения в пикселях.
        :return: список блоков, каждый блок — байтовая строка длины block_size*block_size.
                 Порядок обхода: сначала по строкам блоков, затем по столбцам.
        """
        expected_len = width * height
        if len(data) != expected_len:
            raise ValueError(
                f"Длина data ({len(data)}) не равна width*height ({expected_len})"
            )

        # Преобразуем в 2D-массив
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)

        # Вычисляем паддинг, чтобы размер был кратен block_size
        pad_h = (self.block_size - (height % self.block_size)) % self.block_size
        pad_w = (self.block_size - (width % self.block_size)) % self.block_size

        if pad_h or pad_w:
            # Добавляем справа и снизу строки/столбцы fill_value
            arr = np.pad(
                arr,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=self.fill_value
            )

        h_padded, w_padded = arr.shape
        blocks: List[bytes] = []

        # Перебираем все блоки по N×N
        for y in range(0, h_padded, self.block_size):
            for x in range(0, w_padded, self.block_size):
                block = arr[y:y + self.block_size, x:x + self.block_size]
                blocks.append(block.tobytes())

        return blocks

    def inverse(
            self,
            blocks: List[bytes],
            width: int,
            height: int
    ) -> bytes:
        """
        Собирает изображение обратно из списка блоков N×N.

        :param blocks: список блоков, полученных методом convert.
        :param width: ширина исходного изображения в пикселях.
        :param height: высота исходного изображения в пикселях.
        :return: байтовый буфер собранного изображения (без паддинга), размер width*height.
        """
        # Считаем, сколько блоков по каждой оси
        blocks_per_row = (width + self.block_size - 1) // self.block_size
        blocks_per_col = (height + self.block_size - 1) // self.block_size

        # Создаём массив, заполненный fill_value, размера полностью заблоченного изображения
        full_h = blocks_per_col * self.block_size
        full_w = blocks_per_row * self.block_size
        arr = np.full((full_h, full_w), fill_value=self.fill_value, dtype=np.uint8)

        if len(blocks) != blocks_per_row * blocks_per_col:
            raise ValueError(
                f"Ожидается {blocks_per_row * blocks_per_col} блоков, "
                f"получено {len(blocks)}"
            )

        # Заполняем массив блоками
        for idx, block_bytes in enumerate(blocks):
            # Индексы блока в сетке
            row = idx // blocks_per_row
            col = idx % blocks_per_row

            # Преобразуем байты в квадратный блок
            block_arr = np.frombuffer(block_bytes, dtype=np.uint8).reshape(
                self.block_size, self.block_size
            )

            # Вставляем блок в нужное место
            y0 = row * self.block_size
            x0 = col * self.block_size
            arr[y0:y0 + self.block_size, x0:x0 + self.block_size] = block_arr

        # Обрезаем до исходных размеров
        result = arr[:height, :width]
        return result.tobytes()


if __name__ == "__main__":
    # Пример использования и самотестирование
    width, height = 10, 7
    # Генерируем тестовый буфер: 0,1,2,...,width*height-1
    original_data = bytes(range(width * height))

    splitter = BlockSplitter(block_size=4, fill_value=0)
    blocks = splitter.convert(original_data, width, height)
    reconstructed = splitter.inverse(blocks, width, height)

    print(f"Исходных байт:       {len(original_data)}")
    print(f"Блоков (4×4):        {len(blocks)} по {len(blocks[0])} байт")
    print("Тест сборки:", "OK" if reconstructed == original_data else "FAIL")
