import pandas as pd
import numpy as np


def calc_weighted_avg_price(price, weight, grouper):
    grouper.rename('grouper', inplace=True)
    price_weight = price.multiply(weight)
    df = pd.concat([price_weight, weight], axis=1)
    grouped = df.groupby(grouper).sum()
    res = (
        grouped
        .iloc[:, 0]
        .divide(grouped.iloc[:, 1])
    )
    return res


def calc_weighted_anchored_price(price, weight, grouper):
    grouper.rename('grouper', inplace=True)
    price_weight = price.multiply(weight)
    df = pd.concat([price_weight, weight], axis=1)
    grouped = df.groupby(grouper).cumsum()
    return grouped.iloc[:, 0].divide(grouped.iloc[:, 1])
