import numpy as np

class Downsampler:
    @staticmethod
    def convert(data: bytes, width: int, height: int) -> bytes:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        downsampled = (arr[::2, ::2] + arr[1::2, ::2] + arr[::2, 1::2] + arr[1::2, 1::2]) // 4
        return downsampled.astype(np.uint8).tobytes()

    @staticmethod
    def inverse(data: bytes, width: int, height: int) -> bytes:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height // 2, width // 2)
        upsampled = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        return upsampled.astype(np.uint8).tobytes()
    
if __name__ == '__main__':
    # Пример использования
    width = 8
    height = 12

    # Создадим тестовое изображение (градации серого)
    original_data = bytes(range(width * height))  # 8x8 = 64 байта

    # Уменьшаем разрешение
    downsampled_data = Downsampler.convert(original_data, width, height)

    # Увеличиваем разрешение обратно
    reconstructed_data = Downsampler.inverse(downsampled_data, width, height)

    # Проверка размерностей
    original_arr = np.frombuffer(original_data, dtype=np.uint8).reshape(height, width)
    reconstructed_arr = np.frombuffer(reconstructed_data, dtype=np.uint8).reshape(height, width)

    print(f"Размер исходного изображения: {original_arr.shape}")
    print(f"Размер уменьшенного изображения: {(height // 2, width // 2)}")
    print(f"Размер восстановленного изображения: {reconstructed_arr.shape}")

    # Проверка данных (приблизительное соответствие, т.к. есть потеря информации)
    # Сравниваем только размеры массивов, так как полное совпадение данных маловероятно
    if original_arr.shape == reconstructed_arr.shape:
        print("Размеры исходного и восстановленного изображения совпадают.")
    else:
        print("Размеры исходного и восстановленного изображения не совпадают.")

    # Визуальная проверка (для более тщательного анализа)
    print("\nПервые 10 байт исходного изображения:", original_data[:10])
    print("Первые 10 байт уменьшенного изображения:", downsampled_data[:10])
    print("Первые 10 байт восстановленного изображения:", reconstructed_data[:10])
    
"""
class Downsampler:
    @staticmethod
    def convert(data: bytes, width: int, height: int) -> bytes:
        downsampled = bytearray()
        for i in range(0, height, 2):
            for j in range(0, width, 2):
                # Усреднение для Cb и Cr (каналы 1 и 2)
                cb = (data[i*width + j*3 + 1] + data[i*width + (j+1)*3 + 1] + 
                      data[(i+1)*width + j*3 + 1] + data[(i+1)*width + (j+1)*3 + 1]) // 4
                cr = (data[i*width + j*3 + 2] + data[i*width + (j+1)*3 + 2] + 
                      data[(i+1)*width + j*3 + 2] + data[(i+1)*width + (j+1)*3 + 2]) // 4
                downsampled.extend([data[i*width + j*3], cb, cr])
        return bytes(downsampled)

    @staticmethod
    def inverse(data: bytes, original_width: int, original_height: int) -> bytes:
        restored = bytearray()
        for i in range(0, original_height * original_width // 4):
            Y = data[i*3]
            Cb = data[i*3 + 1]
            Cr = data[i*3 + 2]
            # Повторяем Cb и Cr для 4 пикселей
            for _ in range(4):
                restored.extend([Y, Cb, Cr])
        return bytes(restored)
"""
