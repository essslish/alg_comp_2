import os
import imghdr
from PIL import Image
from uti.logger import logger


def load_test_files(directory) -> dict:
    """
    Загружает все файлы из указанной папки.
    :param directory: путь к папке с тестовыми файлами
    :return: словарь, где ключ – имя файла, значение – данные в байтах
    """
    test_files = {}
    if not os.path.exists(directory):
        logger.error(f"Папка {directory} не найдена.")
        raise FileNotFoundError(f"Папка {directory} не найдена.")

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        test_files[filename] = load_test_file(path)
    return test_files


def load_test_file(path: str) -> bytes:
    """
    Загружает файл по указанному пути.
    Если файл является изображением JPEG/PNG, извлекает и возвращает его сырые пиксельные данные.
    В противном случае — возвращает содержимое файла как есть.
    """
    if not os.path.isfile(path):
        logger.error(f"Файл {path} не найден.")
        raise FileNotFoundError(f"Файл {path} не найден.")

    file_type = imghdr.what(path)

    if file_type in ['jpeg', 'png', 'tiff']:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")  # Преобразуем в RGB, если не в этом формате
                raw_data = img.tobytes()
                logger.debug(f"Изображение {path} загружено как сырые данные (RGB, {img.size}).")
                return raw_data
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {path}: {e}")
            raise
    else:
        with open(path, "rb") as f:
            file_data = f.read()
            logger.debug(f"Файл {path} загружен как бинарные данные ({len(file_data)} байт).")
            return file_data
