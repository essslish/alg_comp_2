from typing import Tuple

import numpy as np
from PIL import Image


class Downsampler:
   
    def __init__(self, factor_y: int = 2, factor_x: int = 2, fill_value: float = 0.0):
        """
        :param factor_y: сколько строк сгруппировать в одну (шаг по вертикали).
        :param factor_x: сколько столбцов сгруппировать в одну (шаг по горизонтали).
        :param fill_value: значение для заполнения краевых пикселов при паддинге.
        """
        if factor_y <= 0 or factor_x <= 0:
            raise ValueError("Коэффициенты даунсемплинга должны быть положительными целыми")
        self.factor_y = factor_y
        self.factor_x = factor_x
        self.fill_value = fill_value

    def downsample(
            self,
            channel: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        :param channel: 2D numpy-массив (любой числовой dtype).
        :return: кортеж (downsampled, original_shape), где
                 downsampled — результат даунсемплинга,
                 original_shape — исходная (height, width) для последующего апсемплинга.
        """
        if channel.ndim != 2:
            raise ValueError("Ожидается 2D-массив")

        h, w = channel.shape
        # Вычисляем размеры с паддингом
        pad_h = (-h) % self.factor_y
        pad_w = (-w) % self.factor_x

        # Паддинг по правому и нижнему краю
        if pad_h or pad_w:
            channel = np.pad(
                channel,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=self.fill_value
            )

        # Новые размеры после паддинга
        h_p, w_p = channel.shape
        new_h = h_p // self.factor_y
        new_w = w_p // self.factor_x

        # Ресайз с усреднением блоков
        # Сначала разбиваем на блоки
        reshaped = channel.reshape(
            new_h, self.factor_y,
            new_w, self.factor_x
        )
        # Усредняем внутри каждого блока
        down = reshaped.mean(axis=(1, 3))

        return down, (h, w)

    def upsample(
            self,
            downsampled: np.ndarray,
            original_shape: Tuple[int, int]
    ) -> np.ndarray:
       
        if downsampled.ndim != 2:
            raise ValueError("Ожидается 2D-массив")

        h_orig, w_orig = original_shape
        # Создаём «L»-изображение из массива
        img = Image.fromarray(downsampled.astype(np.uint8), mode="L")
        # Пересаживаем к нужному размеру
        up_img = img.resize((w_orig, h_orig), resample=Image.BILINEAR)
        # Конвертим обратно в numpy-массив

        return np.array(up_img, dtype=downsampled.dtype)


# Пример использования
if __name__ == "__main__":
    # Создадим тестовый канал 5x7
    channel = np.arange(35).reshape(5, 7).astype(float)
    ds = Downsampler(factor_y=2, factor_x=3, fill_value=0.0)

    down, orig_shape = ds.downsample(channel)
    up = ds.upsample(down, orig_shape)

    print("Original:\n", channel)
    print("Downsampled:\n", down)
    print("Restored:\n", up)
