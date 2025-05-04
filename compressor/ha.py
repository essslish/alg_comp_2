import heapq
import struct

from compressor.base import Compressor


class HACompressor(Compressor):
    class Node:
        """
        Узел дерева Хаффмана.
        """

        def __init__(self, freq, symbol=None, left=None, right=None):
            self.freq = freq
            self.symbol = symbol
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.freq < other.freq

    def compress(self, data: bytes) -> bytes:
        """
        Сжимает данные алгоритмом Хаффмана.
        Шаги:
          1. Подсчёт частот символов.
          2. Построение дерева Хаффмана.
          3. Формирование таблицы кодов.
          4. Кодирование данных с помощью полученной таблицы.
          5. Сохранение заголовка с информацией о частотах для декомпрессии.
        """
        if not data:
            return b""

        # 1. Подсчет частот для каждого символа
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1

        # 2. Построение дерева Хаффмана
        root = self._build_ha_tree(freq)

        # 3. Формирование таблицы кодов: рекурсивный обход дерева
        codes = {}

        def build_codes(node, code=""):
            if node is None:
                return
            if node.symbol is not None:
                codes[node.symbol] = code or "0"  # для случая одного уникального символа
            build_codes(node.left, code + "0")
            build_codes(node.right, code + "1")

        build_codes(root)

        # 4. Кодирование данных
        encoded_bits = "".join(codes[byte] for byte in data)

        # Добавляем паддинг, чтобы длина была кратна 8
        pad_len = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += "0" * pad_len

        # Упаковка битовой строки в байты
        compressed_data = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte = int(encoded_bits[i:i + 8], 2)
            compressed_data.append(byte)

        # 5. Формирование заголовка:
        # Сохраним число уникальных символов (2 байта),
        # для каждого символа сохраним: 1 байт (symbol) + 4 байта (частота)
        header = bytearray()
        header += struct.pack(">H", len(freq))  # число уникальных символов (big-endian)
        for symbol, f in freq.items():
            header += struct.pack(">BI", symbol, f)

        # Сохраним информацию о паддинге (1 байт)
        header += struct.pack("B", pad_len)

        # Итог: заголовок + сжатые данные
        return bytes(header) + bytes(compressed_data)

    def decompress(self, data: bytes) -> bytes:
        """
        Восстанавливает исходные данные, декодируя поток,
        сгенерированный алгоритмом Хаффмана.
        Шаги:
          1. Считывание заголовка и восстановление частотной таблицы.
          2. Построение дерева Хаффмана.
          3. Декодирование битового потока.
        """
        if not data:
            return b""

        pos = 0
        # 1. Читаем число уникальных символов (2 байта)
        num_symbols = struct.unpack(">H", data[pos:pos + 2])[0]
        pos += 2

        # Восстанавливаем частотную таблицу
        freq = {}
        for _ in range(num_symbols):
            symbol, f = struct.unpack(">BI", data[pos:pos + 5])
            freq[symbol] = f
            pos += 5

        # Чтение информации о паддинге (1 байт)
        pad_len = struct.unpack("B", data[pos:pos + 1])[0]
        pos += 1

        # Остальные данные – это сжатый поток
        compressed_bits = ""
        for byte in data[pos:]:
            compressed_bits += f"{byte:08b}"
        # Удаляем паддинг в конце
        if pad_len:
            compressed_bits = compressed_bits[:-pad_len]

        # 2. Построение дерева Хаффмана на основе частот
        root = self._build_ha_tree(freq)

        # 3. Декодирование битового потока
        decoded_bytes = bytearray()
        node = root
        for bit in compressed_bits:
            if bit == "0":
                node = node.left
            else:
                node = node.right
            if node.symbol is not None:  # достигли листа
                decoded_bytes.append(node.symbol)
                node = root

        return bytes(decoded_bytes)

    def _build_ha_tree(self, freq):
        heap = []
        for symbol, f in freq.items():
            heapq.heappush(heap, self.Node(f, symbol))
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = self.Node(left.freq + right.freq, None, left, right)
            heapq.heappush(heap, merged)
        return heap[0]
