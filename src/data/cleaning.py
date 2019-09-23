import pandas as pd, numpy as np

def clean_ohlc(ohlc_df):
    ohlc_df.loc[ohlc_df['Open'] > ohlc_df['High'], 'High'] = ohlc_df['Open']
    ohlc_df.loc[ohlc_df['Open'] < ohlc_df['Low'], 'Low'] = ohlc_df['Open']
    ohlc_df.loc[ohlc_df['Close'] > ohlc_df['High'], 'High'] = ohlc_df['Close']
    ohlc_df.loc[ohlc_df['Close'] < ohlc_df['Low'], 'Low'] = ohlc_df['Close']
    return ohlc_df

def resample_ohlc(ohlc_df, freq='D'):
    resample_df = pd.DataFrame()
    resample_df['Open'] = ohlc_df['Open'].resample(freq).first().copy()
    resample_df['High'] = ohlc_df['High'].resample(freq).max().copy()
    resample_df['Low'] = ohlc_df['Low'].resample(freq).min().copy()
    resample_df['Close'] = ohlc_df['Close'].resample(freq).last().copy()
    resample_df['Volume'] = ohlc_df['Volume'].resample(freq).sum().copy()
    return resample_df

def split_ohlcv(ohlcv):
    return ohlcv['Open'], ohlcv['High'], ohlcv['Low'], ohlcv['Close'], ohlcv['Volume']

def get_tick_range(window, tick_size=50):
    window_high = int(window['High'].max())
    window_low = int(window['Low'].min())
    tick_range = range(window_low, window_high, tick_size)
    return np.array(tick_range)

def pd_tickround(df, tick_size=1):
    return np.round(df.astype(float).divide(tick_size)).multiply(tick_size).astype(int)

def tickround(x, tick_size=5):
    return int(tick_size * round(float(x)/tick_size))

def calc_num_contracts(symbol, price, pos_size):
    if symbol[:3] == 'XBT':
        contracts = pos_size * (price / 100)
    elif symbol[-3:] == 'USD':
        contracts = pos_size / (price / 100 * 1e-06)
    else:
        contracts = pos_size / (price * 1e-08)
    return int(contracts)