from abc import ABC, abstractmethod


class Compressor(ABC):
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """
        Метод для сжатия данных.
        :param data: исходные данные в виде байтов
        :return: сжатые данные в виде байтов
        """
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """
        Метод для восстановления исходных данных из сжатых.
        :param data: сжатые данные в виде байтов
        :return: восстановленные исходные данные в виде байтов
        """
        pass
