import numpy as np
import pandas as pd



def find_outliers_iqr(data: pd.DataFrame, feature: str, left: float = 1.3, right: float = 1.5, log_scale: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finds outliers in the data using the interquartile range (IQR) method. 
    The classic method is modified by adding:
    * the ability to log-scale the distribution
    * manual control over the number of IQRs on both sides of the distribution
    
    Args:
        data (pandas.DataFrame): The dataset.
        feature (str): The name of the feature used for outlier detection.
        left (float, optional): The number of IQRs on the left side of the distribution. Default is 1.5.
        right (float, optional): The number of IQRs on the right side of the distribution. Default is 1.5.
        log_scale (bool, optional): Log-scale mode. Default is False, which means no log-scale is applied.
    
    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: 
        - The outliers that fall outside the defined bounds.
        - The cleaned data, excluding the outliers.
    """
    if log_scale:
        x = np.log(data[feature] + 1)
    else:
        x = data[feature]
    
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75)
    iqr = quartile_3 - quartile_1
    
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    
    return outliers, cleaned


def find_outliers_z_score(data: pd.DataFrame, feature: str, left: float = 3, right: float = 3, log_scale: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Находит выбросы в данных, используя метод z-отклонений.
    Классический метод модифицирован путем добавления:
    * возможности логарифмирования распределения
    * ручного управления количеством стандартных отклонений в обе стороны распределения

     Аргументы:
        data (pandas.DataFrame): Набор данных.
        feature (str): Имя признака, на основе которого происходит поиск выбросов.
        left (float, опционально): Количество стандартных отклонений в левую сторону распределения. По умолчанию 3.
        right (float, опционально): Количество стандартных отклонений в правую сторону распределения. По умолчанию 3.
        log_scale (bool, опционально): Режим логарифмирования. По умолчанию False - логарифмирование не применяется.

     Возвращает:
        tuple[pd.DataFrame, pd.DataFrame]: Наблюдения, попавшие в разряд выбросов, и очищенные данные, из которых исключены выбросы.
    """
    # Применяем логарифмирование, если задано
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]

    # Вычисляем среднее и стандартное отклонение
    mu = x.mean()
    sigma = x.std()

    # Вычисляем нижнюю и верхнюю границы для выбросов
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma

    # Находим выбросы и очищенные данные
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]

    return outliers, cleaned


 def find_outliers_quantile(data, feature, left=0.01, right=0.99):
    """
    Находит выбросы в заданном признаке набора данных с использованием квантилей.

    Аргументы:
        data (DataFrame): Набор данных.
        feature (str): Название признака.
        left (float, optional): Левый квантиль. По умолчанию 0.01.
        right (float, optional): Правый квантиль. По умолчанию 0.99.

    Возвращает:
        DataFrame: Выбросы.
        DataFrame: Очищенный набор данных.
    """
    # Извлечение столбца признака из набора данных
    x = data[feature]

    # Вычисление нижней и верхней границы квантилей
    lower_bound = x.quantile(left)
    upper_bound = x.quantile(right)

    # Поиск выбросов на основе границ квантилей
    outliers = data[(x < lower_bound) | (x > upper_bound)]

    # Удаление выбросов из набора данных
    cleaned = data[(x > lower_bound) & (x < upper_bound)]

    return outliers, cleaned