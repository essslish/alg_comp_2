"""
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.1687*R - 0.3313*G + 0.5*B + 128
    Cr = 0.5*R - 0.4187*G - 0.0813*B + 128
"""
import numpy as np
from typing import Tuple


"""class RGBToYCbCr:
    @staticmethod
    def convert(rgb_bytes: bytes, width: int, height: int) -> Tuple[bytes, bytes, bytes]:
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
        
        Y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        Cb = -0.1687 * rgb[:,:,0] - 0.3313 * rgb[:,:,1] + 0.5 * rgb[:,:,2] + 128
        Cr = 0.5 * rgb[:,:,0] - 0.4187 * rgb[:,:,1] - 0.0813 * rgb[:,:,2] + 128
        
        Y = np.clip(Y, 0, 255).astype(np.uint8)
        Cb = np.clip(Cb, 0, 255).astype(np.uint8)
        Cr = np.clip(Cr, 0, 255).astype(np.uint8)
        
        return Y.tobytes(), Cb.tobytes(), Cr.tobytes()
    
    @staticmethod
    def inverse(y_bytes: bytes, cb_bytes: bytes, cr_bytes: bytes, width: int, height: int) -> bytes:
        Y = np.frombuffer(y_bytes, dtype=np.uint8).reshape(height, width)
        Cb = np.frombuffer(cb_bytes, dtype=np.uint8).reshape(height, width) - 128
        Cr = np.frombuffer(cr_bytes, dtype=np.uint8).reshape(height, width) - 128
        
        R = Y + 1.402 * Cr
        G = Y - 0.34414 * Cb - 0.71414 * Cr
        B = Y + 1.772 * Cb
        
        rgb = np.stack([R, G, B], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8).tobytes()
    

test_rgb = bytes([
    255, 0, 0,    # Пиксель 0
    0, 255, 0,    # Пиксель 1
    0, 0, 255,    # Пиксель 2
    255, 255, 255 # Пиксель 3
])

width, height = 2, 2

# Прямое преобразование RGB -> YCbCr
Y, Cb, Cr = RGBToYCbCr.convert(test_rgb, width, height)

# Обратное преобразование YCbCr -> RGB
restored_rgb = RGBToYCbCr.inverse(Y, Cb, Cr, width, height)

# Вывод результатов
print("Исходные RGB данные:")
print(np.frombuffer(test_rgb, dtype=np.uint8).reshape(height, width, 3))
print("\nРезультаты преобразования RGB->YCbCr:")
print("Y канал:", np.frombuffer(Y, dtype=np.uint8).reshape(height, width))
print("Cb канал:", np.frombuffer(Cb, dtype=np.uint8).reshape(height, width))
print("Cr канал:", np.frombuffer(Cr, dtype=np.uint8).reshape(height, width))
print("\nВосстановленные RGB данные:")
print(np.frombuffer(restored_rgb, dtype=np.uint8).reshape(height, width, 3))
print("\nСовпадают ли исходные и восстановленные данные?",
      np.array_equal(np.frombuffer(test_rgb, dtype=np.uint8),
                    np.frombuffer(restored_rgb, dtype=np.uint8)))"""

    
class RGBToYCbCr:
    @staticmethod
    def convert(rgb_data: bytes) -> bytes:
        ycbcr_data = bytearray()
        for i in range(0, len(rgb_data), 3):
            R, G, B = rgb_data[i], rgb_data[i+1], rgb_data[i+2]
            Y = int(0.299 * R + 0.587 * G + 0.114 * B)
            Cb = int(-0.1687 * R - 0.3313 * G + 0.5 * B + 128)
            Cr = int(0.5 * R - 0.4187 * G - 0.0813 * B + 128)
            ycbcr_data.extend([Y, Cb, Cr])
        return bytes(ycbcr_data)

    @staticmethod
    def inverse(ycbcr_data: bytes) -> bytes:
        rgb_data = bytearray()
        for i in range(0, len(ycbcr_data), 3):
            Y, Cb, Cr = ycbcr_data[i], ycbcr_data[i+1], ycbcr_data[i+2]
            R = int(Y + 1.402 * (Cr - 128))
            G = int(Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128))
            B = int(Y + 1.772 * (Cb - 128))
            rgb_data.extend([max(0, min(255, R)), max(0, min(255, G)), max(0, min(255, B))])
        return bytes(rgb_data)
        

# Тест: преобразование туда-обратно
rgb_test = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])  # Красный, Зелёный, Синий
ycbcr = RGBToYCbCr.convert(rgb_test)
rgb_back = RGBToYCbCr.inverse(ycbcr)
print("Original RGB:", list(rgb_test))
print(ycbcr)
print("Restored RGB:", list(rgb_back))  


