"""
Модуль для преобразования изображений с помощью библиотеки Pillow:
- конвертация в оттенки серого (Grayscale);
- преобразование в чёрно-белое без дизеринга;
- преобразование в чёрно-белое с дизерингом (Floyd–Steinberg).
"""

import sys
from io import BytesIO

from PIL import Image


class ImageConverter:
    """
    Класс для выполнения основных преобразований цветного изображения:
      - в градации серого (L);
      - в чёрно-белое без дизеринга (1-бит);
      - в чёрно-белое с дизерингом (1-бит, Floyd–Steinberg).
    """

    @staticmethod
    def to_grayscale(img: Image.Image) -> Image.Image:
        """
        Преобразует изображение в оттенки серого.

        :param img: объект PIL.Image (любой режим).
        :return: новый объект PIL.Image в режиме "L" (8-битных серых оттенков).
        """
        return img.convert("L")

    @staticmethod
    def to_bw(img: Image.Image, threshold: int = 128) -> Image.Image:
        """
        Преобразует изображение в чёрно-белое без дизеринга.

        Используется простое пороговое значение: пиксели со значением >= threshold
        станут белыми (255), остальные — чёрными (0).

        :param img: объект PIL.Image.
        :param threshold: порог в диапазоне [0..255].
        :return: новый объект PIL.Image в режиме "1" (1-бит на пиксель).
        """
        # Сначала приводим к оттенкам серого
        gray = img.convert("L")
        # Применяем порог и переводим в режим "1"
        bw = gray.point(lambda p: 255 if p >= threshold else 0, mode="1")
        return bw

    @staticmethod
    def to_dithered_bw(img: Image.Image) -> Image.Image:
        """
        Преобразует изображение в чёрно-белое с дизерингом Флойда–Стейнберга.

        :param img: объект PIL.Image.
        :return: новый объект PIL.Image в режиме "1" (1-бит на пиксель) с дизерингом.
        """
        # Pillow поддерживает дизеринг при конвертации в 1-битный режим
        return img.convert("1", dither=Image.FLOYDSTEINBERG)

    @staticmethod
    def save_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
        """
        Сохраняет объект PIL.Image в байтовый буфер.

        :param img: объект PIL.Image.
        :param fmt: формат сохранения (например, "PNG", "JPEG").
        :return: байтовая строка с данными изображения.
        """
        buffer = BytesIO()
        img.save(buffer, format=fmt)
        return buffer.getvalue()


if __name__ == "__main__":
    path = sys.argv[1]
    img = Image.open(path)

    # Конвертация в оттенки серого
    gray = ImageConverter.to_grayscale(img)
    gray.save("../output/grayscale.png")

    # Ч/б без дизеринга
    bw = ImageConverter.to_bw(img, threshold=128)
    bw.save("../output/bw.png")

    # Ч/б с дизерингом
    dithered = ImageConverter.to_dithered_bw(img)
    dithered.save("../output/dithered.png")

    print("Готово: output/grayscale.png, output/bw.png, output/dithered.png")
