from typing import Sequence, List


class DCDifferentialCodec:

    @staticmethod
    def encode(dc_coeffs: Sequence[int]) -> List[int]:
        """
        Прямое дифференциальное кодирование DC‐коэффициентов.

        Первый элемент списка разностей равен первому DC‐коэффициенту (разность с «нулевым» предыдущим значением).
        Каждый последующий элемент — разность текущего DC и предыдущего.

        :param dc_coeffs: последовательность DC‐коэффициентов (целые значения).
        :return: список целых разностей той же длины, что и входной список.
        :raises ValueError: если входная последовательность пуста.
        """
        if not dc_coeffs:
            raise ValueError("Последовательность dc_coeffs не должна быть пустой")

        diffs: List[int] = [dc_coeffs[0]]
        # Для каждого последующего коэффициента сохраняем разницу с предыдущим
        for prev, curr in zip(dc_coeffs, dc_coeffs[1:]):
            diffs.append(curr - prev)
        return diffs

    @staticmethod
    def decode(diffs: Sequence[int]) -> List[int]:
        """
        Обратное дифференциальное декодирование DC‐коэффициентов.

        Восстанавливает исходную последовательность DC‐коэффициентов из списка разностей.

        :param diffs: последовательность разностей (целые значения), длина ≥ 1.
        :return: список восстановленных DC‐коэффициентов.
        :raises ValueError: если входная последовательность пуста.
        """
        if diffs is None or len(diffs) == 0:
            raise ValueError("Последовательность diffs не должна быть пустой")

        dc_coeffs: List[int] = [diffs[0]]
        # Для каждого последующего diff восстанавливаем следующий DC как сумму с предыдущим
        for d in diffs[1:]:
            dc_coeffs.append(dc_coeffs[-1] + d)
        return dc_coeffs


if __name__ == "__main__":
    # Пример использования и самотестирование
    # Задаём тестовую последовательность DC‐коэффициентов для серии блоков
    original_dc = [100, 102, 98, 105, 105, 110]

    # Кодируем разности
    diffs = DCDifferentialCodec.encode(original_dc)
    print("Исходные DC:", original_dc)
    print("Разности DC:", diffs)

    # Декодируем обратно
    restored_dc = DCDifferentialCodec.decode(diffs)
    print("Восстановленные DC:", restored_dc)

    # Проверяем корректность
    assert restored_dc == original_dc, "Ошибка: восстановленные DC не совпадают с исходными"
    print("Тест пройден: восстановление DC корректно.")
