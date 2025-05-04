import numpy as np

class BlockSplitter:
    @staticmethod
    def convert(data: bytes, width: int, height: int, block_size: int = 8) -> list[bytes]:
        """
            Returns:
            Список блоков, каждый блок - байтовая строка.
        """
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)  
        h_pad = (block_size - height % block_size) % block_size  # Вычисляем, сколько строк нужно добавить для padding по высоте, чтобы высота была кратна `block_size`. 
        w_pad = (block_size - width % block_size) % block_size  # Вычисляем, сколько столбцов нужно добавить для padding по ширине, чтобы ширина была кратна `block_size`.
        padded = np.pad(arr, ((0, h_pad), (0, w_pad)), mode='constant') 
        blocks = [padded[y:y+block_size, x:x+block_size].tobytes()  
                  for y in range(0, padded.shape[0], block_size)  # Перебираем строки блоков с шагом `block_size`.
                  for x in range(0, padded.shape[1], block_size)]  
        return blocks  

    @staticmethod
    def inverse(blocks: list[bytes], width: int, height: int, block_size: int = 8) -> bytes:
        """
        Returns:
            Байтовое представление восстановленного изображения.
        """
        h_blocks = (height + block_size - 1) // block_size  # Вычисляем количество блоков по высоте. Целочисленное деление с округлением вверх.
        w_blocks = (width + block_size - 1) // block_size  # Вычисляем количество блоков по ширине. Целочисленное деление с округлением вверх.
        padded = np.zeros((h_blocks * block_size, w_blocks * block_size), dtype=np.uint8)  
        for i, block in enumerate(blocks): 
            y = (i // w_blocks) * block_size  # Вычисляем y-координату верхнего левого угла текущего блока в восстановленном изображении.  `i // w_blocks` - номер строки блока.
            x = (i % w_blocks) * block_size  # Вычисляем x-координату верхнего левого угла текущего блока в восстановленном изображении. `i % w_blocks` - номер столбца блока.
            padded[y:y+block_size, x:x+block_size] = np.frombuffer(block, dtype=np.uint8).reshape(block_size, block_size)  
        return padded[:height, :width].tobytes() 

if __name__ == '__main__':
    # Пример использования
    width = 10
    height = 7

    # Создадим тестовое изображение (просто последовательность байтов)
    original_data = bytes(range(width * height))

    # Разбиваем изображение на блоки
    blocks = BlockSplitter.convert(original_data, width, height)

    print(f"Количество блоков: {len(blocks)}")
    print(f"Размер первого блока: {len(blocks[0])}")


    # Собираем изображение из блоков
    reconstructed_data = BlockSplitter.inverse(blocks, width, height)

    # Проверяем, что восстановленное изображение совпадает с исходным
    if original_data == reconstructed_data:
        print("Тест пройден: Восстановленное изображение совпадает с исходным.")
    else:
        print("Тест не пройден: Восстановленное изображение отличается от исходного.")
        print(f"Original data: {original_data}")
        print(f"Reconstructed data: {reconstructed_data}")