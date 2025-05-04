import numpy as np

class ZigZag:
    @staticmethod
    def convert(matrix_bytes: bytes, width: int, height: int) -> bytes:
        """
        Прямое преобразование: зигзаг-обход матрицы
        Returns:
            байтовая строка с элементами в порядке зигзаг-обхода (int16).
        """
        matrix = np.frombuffer(matrix_bytes, dtype=np.int16).reshape(height, width).tolist()  # Преобразуем байтовую строку в NumPy массив типа int16 и преобразуем его в обычный список
        
        def walk(mat):
            """
            Внутренняя функция для выполнения зигзаг-обхода матрицы.
            """
            zigzag = [] 
            n = len(mat) 
            for index in range(1, 2*n):  # Перебираем диагонали матрицы.
                if index <= n:  # Если номер диагонали меньше или равен размеру матрицы:
                    slice = [row[:index] for row in mat[:index]]  # Выделяем часть матрицы до текущей диагонали.
                else:  # Если номер диагонали больше размера матрицы:
                    slice = [row[index-n:] for row in mat[index-n:]]  # Выделяем часть матрицы после текущей диагонали.
                
                diag = [slice[i][len(slice)-i-1] for i in range(len(slice))]  # Извлекаем элементы текущей диагонали.
                if len(diag) % 2:  
                    diag.reverse() 
                zigzag += diag  
            return zigzag 
        
        zigzag = walk(matrix) 
        return np.array(zigzag, dtype=np.int16).tobytes() 

    @staticmethod
    def inverse(zigzag_bytes: bytes, width: int, height: int) -> bytes:
        """
        :return: байтовая строка с восстановленной матрицей
        """
        zigzag = np.frombuffer(zigzag_bytes, dtype=np.int16).tolist()
        n = height
        matrix = [[0]*width for _ in range(height)]
        index = 0
        
        # заполняем матрицу как раньше
        for d in range(1, 2*n):
            if d <= n:
                slice_size = d
                for i in range(slice_size):
                    if d % 2:
                        row = i
                        col = slice_size - 1 - i
                    else:
                        row = slice_size - 1 - i
                        col = i
                    if row < height and col < width and index < len(zigzag):
                        matrix[row][col] = zigzag[index]
                        index += 1
            else:
                slice_size = 2*n - d
                for i in range(slice_size):
                    if d % 2:
                        row = n - slice_size + i
                        col = n - 1 - i
                    else:
                        row = n - 1 - i
                        col = n - slice_size + i
                    if row < height and col < width and index < len(zigzag):
                        matrix[row][col] = zigzag[index]
                        index += 1
        
        # отражаем элементы относительно главной диагонали
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        return np.array(matrix, dtype=np.int16).tobytes()
    
# Создаем тестовую матрицу 4x4
test_matrix = np.array([
    [1, 2, 6, 7],
    [3, 5, 8, 13],
    [4, 9, 12, 14],
    [10, 11, 15, 16]
], dtype=np.int16)

# Преобразуем в байтовую строку
matrix_bytes = test_matrix.tobytes()

# Прямое преобразование
zigzag_bytes = ZigZag.convert(matrix_bytes, 4, 4)
print("Зигзаг-последовательность:", np.frombuffer(zigzag_bytes, dtype=np.int16))
print(zigzag_bytes)

# Обратное преобразование
restored_bytes = ZigZag.inverse(zigzag_bytes, 4, 4)
restored_matrix = np.frombuffer(restored_bytes, dtype=np.int16).reshape(4, 4)
print("Восстановленная матрица:")
print(restored_bytes)
print(restored_matrix)