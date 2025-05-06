from typing import ByteString


class RGBToYCbCr:
    """
    Класс для преобразования изображений между цветовыми пространствами RGB и YCbCr.

    Формулы преобразования (ITU-R BT.601):
        Y  =  0.299  · R + 0.587  · G + 0.114  · B
        Cb = -0.1687 · R - 0.3313  · G + 0.5     · B + 128
        Cr =  0.5     · R - 0.4187  · G - 0.0813  · B + 128

    Обратное преобразование:
        R = Y + 1.402   · (Cr - 128)
        G = Y - 0.34414 · (Cb - 128) - 0.71414 · (Cr - 128)
        B = Y + 1.772   · (Cb - 128)
    """

    @staticmethod
    def convert(rgb_data: ByteString) -> bytes:
    
        if len(rgb_data) % 3 != 0:
            raise ValueError("Длина rgb_data должна быть кратна 3")

        ycbcr = bytearray(len(rgb_data))
        for i in range(0, len(rgb_data), 3):
            r = rgb_data[i]
            g = rgb_data[i + 1]
            b = rgb_data[i + 2]

            # Вычисляем компоненты Y, Cb, Cr
            y = round(0.299 * r + 0.587 * g + 0.114 * b)
            cb = round(-0.1687 * r - 0.3313 * g + 0.5 * b + 128)
            cr = round(0.5 * r - 0.4187 * g - 0.0813 * b + 128)

            # Записываем результат, обрезая значения в диапазон [0, 255]
            ycbcr[i] = max(0, min(255, y))
            ycbcr[i + 1] = max(0, min(255, cb))
            ycbcr[i + 2] = max(0, min(255, cr))

        return bytes(ycbcr)

    @staticmethod
    def inverse(ycbcr_data: ByteString) -> bytes:
       
        if len(ycbcr_data) % 3 != 0:
            raise ValueError("Длина ycbcr_data должна быть кратна 3")

        rgb = bytearray(len(ycbcr_data))
        for i in range(0, len(ycbcr_data), 3):
            y = ycbcr_data[i]
            cb = ycbcr_data[i + 1]
            cr = ycbcr_data[i + 2]

            # Обратное преобразование в RGB
            r = int(y + 1.402 * (cr - 128))
            g = int(y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128))
            b = int(y + 1.772 * (cb - 128))

            # Обрезаем в диапазон [0, 255]
            rgb[i] = max(0, min(255, r))
            rgb[i + 1] = max(0, min(255, g))
            rgb[i + 2] = max(0, min(255, b))

        return bytes(rgb)


if __name__ == "__main__":
    # Пример использования и простейший тест "туда-обратно"
    original_rgb = bytes([255, 0, 0,  # Красный
                          0, 255, 0,  # Зелёный
                          0, 0, 255])  # Синий

    # Преобразование в YCbCr и обратно
    ycbcr = RGBToYCbCr.convert(original_rgb)
    restored_rgb = RGBToYCbCr.inverse(ycbcr)

    print("Оригинальный RGB:", list(original_rgb))
    print("YCbCr данные:      ", list(ycbcr))
    print("Восстановленный RGB:", list(restored_rgb))
