import numpy as np

class DC_coeff:
    @staticmethod
    def convert(dc_coeffs: bytes) -> bytes:
        """
        :return: байтовая строка с разностями (первый элемент сохраняется)
        """
        arr = np.frombuffer(dc_coeffs, dtype=np.int16)
        if len(arr) == 0:
            return b""
        
        # Создаем массив для разностей
        diffs = np.empty_like(arr)
        diffs[0] = arr[0]  # Первый элемент сохраняем как есть
        
        for i in range(1, len(arr)):
            diffs[i] = arr[i] - arr[i-1]
            
        return diffs.tobytes()

    @staticmethod
    def inverse(diff_bytes: bytes) -> bytes:
        """
        :return: байтовая строка с исходными DC коэффициентами
        """
        diffs = np.frombuffer(diff_bytes, dtype=np.int16)
        if len(diffs) == 0:
            return b""
        
        dc_coeffs = np.empty_like(diffs)
        dc_coeffs[0] = diffs[0]  # Первый элемент остается
        
        for i in range(1, len(diffs)):
            dc_coeffs[i] = dc_coeffs[i-1] + diffs[i]
            
        return dc_coeffs.tobytes()
    

# Тестовые данные
original_dc = np.array([10, 12, 15, 14, 18], dtype=np.int16)
print("Исходные DC:", original_dc)

# Прямое преобразование
encoded = DC_coeff.convert(original_dc.tobytes())
decoded = DC_coeff.inverse(encoded)

# Проверка
print("Закодированные разности:", np.frombuffer(encoded, dtype=np.int16))
print("Восстановленные DC:", np.frombuffer(decoded, dtype=np.int16))
print("Совпадение:", np.array_equal(original_dc, np.frombuffer(decoded, dtype=np.int16)))