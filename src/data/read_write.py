from cleaning import get_aggregated_trades
import arctic
from arctic.date import DateRange
import requests
import time
import zlib
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class BitmexScraper:

    S3_ENDPOINT = 'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/{}/{}.csv.gz'

    def __init__(self, proxy=False):
        self.session = requests.Session()
        if proxy:
            self.session.trust_env = False
            HTTPS_PROXY = os.getenv('HTTPS_PROXY')
            self.proxies = {
                'https': HTTPS_PROXY}
            self.get_kwargs = {'proxies': self.proxies}
        else:
            self.get_kwargs = {}

    @staticmethod
    def format_trade_df(df, aggregate=False):
        df = (
            df
            .set_index('timestamp')
            .filter(['symbol', 'side', 'size', 'price'])
            .dropna(how='all')
            .astype({'symbol': str, 'side': str, 'size': int, 'price': float})
        )
        if aggregate:
            df = get_aggregated_trades(df)
        return df

    @staticmethod
    def format_quote_df(df):
        df = (
            df
            .set_index('timestamp')
            .dropna(how='all')
            .replace('', {'bidSize': 0, 'bidPrice': np.nan, 'askSize': 0, 'askPrice': np.nan})
            .astype({'symbol': str, 'bidSize': int, 'bidPrice': float, 'askSize': int, 'askPrice': float})
        )
        return df

    def data_to_df(self, data, channel='trade'):
        """
        Takes a list of strings and formats in to trades df

        Args:
            data (str): list of strings
            channel (str): trade or quote

        Returns
            df (pd.DataFrame): formatted DataFrame
        """
        df = pd.Series(data).str.split(',', expand=True)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('D', ' '))
        format_dict = {'trade': self.format_trade_df,
                       'quote': self.format_quote_df}
        return format_dict[channel](df)

    def get_df(self, date, channel='trade'):
        """
        Scrapes file from Bitmex S3 bucket and formats as pandas df

        Args
            date: a datetime-like object

        Returns
            trade_df: pandas DataFrame
        """
        re = requests.exceptions
        exc_tup = tuple([getattr(re, i) for i in dir(re)[:24]]) + (OSError,)

        date_str = date.strftime('%Y%m%d')
        url = self.S3_ENDPOINT.format(channel, date_str)
        retry_count = 0
        while True:
            try:
                r = self.session.get(url, **self.get_kwargs)
            except exc_tup:
                time.sleep(1)
                retry_count += 1
                continue
            break
        if retry_count:
            print(f'retried {retry_count} times')

        data = (
            zlib
            .decompress(r.content, zlib.MAX_WBITS | 32)
            .decode()
            .split("\n")
        )
        df = self.data_to_df(data, channel)
        df.index.name = 'date'
        return df

    def scrape_to_arctic(self, library, start=None, end=None, host='localhost', channel='trade', **kwargs):
        if not start:
            start = '11/22/2014'
        if not end:
            end = (pd.datetime.utcnow() - pd.Timedelta(days=1, hours=6)).date()

        date_range = pd.date_range(start, end)

        con = arctic.Arctic(host)
        if library not in con.list_libraries():
            lib_type = kwargs.get('lib_type', arctic.CHUNK_STORE)
            con.initialize_library(library, lib_type=lib_type)
        lib = con[library]

        for date in tqdm(date_range):
            df = self.get_df(date, channel)

            lib.update(channel, df, upsert=True)


class ArcticHandler:
    """
    Reads in data from an arctic data store

    param hostname: where arctic data store is hosted
    param library: arctic library where data is stored
    """

    def __init__(self, library:str, hostname='localhost'):
        self.arctic = arctic.Arctic(hostname)
        self.library = self.arctic[library]

    def calc_averages(self, symbol:str, pair: str, start, end, chunk_size: str, rs_freq: str, save_csv=False):
        """
        Pulls data between specified date range from arctic and writes to .csv
        in chunks

        :param symbol: name of the arctic library
        :param pair: name of the trading pair
        :param start: start date for sample
        :param end: end date for sample
        :param chunk_size: size of chunk to read from arctic on each iteration (7D)
        :param rs_freq: the frequency to gather statistics on (30T, 4h, 1D)
        :param save_csv: where to store final .csv file
        """
        dates = pd.date_range(start, end, freq=chunk_size)
        dates_rs = pd.date_range(start, end, freq=rs_freq)
        agg_dict = {'price': pd.Series.count, 'size': sum}

        df = pd.DataFrame(index=dates_rs, columns=['price', 'size'])
        for i in trange(len(dates[:-1])):
            chunk_start = dates[i]
            chunk_end = dates[i+1]
            chunk = self.get_arctic_chunk(
                symbol, chunk_start, chunk_end)
            chunk = self.format_chunk(chunk, pair)
            chunk_avg = chunk.resample(rs_freq).agg(agg_dict)
            df.loc[chunk_avg.index] = chunk_avg

        if save_csv:
            str_format = '%m%b%Y'
            start_str = pd.to_datetime(start).strftime(str_format)
            end_str = pd.to_datetime(end).strftime(str_format)
            df.to_csv(
                f'../data/interim/{symbol} averages {start_str}_{end_str}.csv')
        return df

    def write_csv(self, symbol: str, pair: str, start:str, end:str, chunk_size: str, path=None, filename=None):
        """
        Pulls data between specified date range from arctic and writes to .csv
        in chunks

        :param symbol: name of the arctic library
        :param pair: name of the trading pair
        :param start: start date for sample
        :param end: end date for sample
        :param chunk_size: size of chunk to read from arctic on each iteration
        :param path: where to store final .csv file
        """
        dates = pd.date_range(start, end, freq=chunk_size)
        str_format = '%m%b%Y'
        start_str = pd.to_datetime(start).strftime(str_format).upper()
        end_str = pd.to_datetime(end).strftime(str_format).upper()

        path = '../data/raw' if path is None else path
        filename = f'{symbol} mlfinlab format {start_str}_{end_str}.csv' if filename is None else filename
        savepath = os.path.join(path, filename)

        # Delete existing file if there is one
        if os.path.exists(savepath):
            os.remove(savepath)

        header = True
        for i in trange(len(dates[:-1])):
            chunk_start = dates[i]
            chunk_end = dates[i+1]
            chunk = self.get_arctic_chunk(
                symbol, chunk_start, chunk_end)
            chunk = self.format_chunk(chunk, pair)
            chunk.to_csv(savepath, mode='a', header=header, index=False)
            header = False

    def get_arctic_chunk(self, symbol: str, start: str, end: str):
        chunk_range = make_date_range(start, end)
        chunk = self.library.read(symbol, chunk_range=chunk_range)
        return chunk

    @staticmethod
    def format_chunk(chunk, symbol: str, symbol_col='symbol'):
        """
        Takes chunk (with multiple instruments) and formats to play nice with mlfinlab

        Args
            symbol (str): the pair/symbol to be filtered out
            symbol_col (str): the column in the chunk to filter on
        """
        chunk = chunk[chunk['symbol'] == symbol]
        formatted = (
            chunk
            .drop(['symbol', 'side'], axis='columns')
            [['price', 'size']]
            .reset_index()
        )
        formatted.columns = ['date_time', 'price', 'volume']
        return formatted


def make_date_range(start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return DateRange(start, end)


def write_test():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df.to_csv('../data/test.csv')


if __name__ == '__main__':
    pass

#     library = 'bitmex-s3-chunk'
#     item = 'trade'
#     symbol = 'XBTUSD'
#     start = '11/22/2014'
#     end = '9/21/2017'
#     chunk_size = '7d'
#     rs_freq = '1T'

#     arctic_reader = ArcticReader()
#     avg = arctic_reader.calc_averages(
#         library=library, item=item, symbol=symbol, start=start, end=end,
#         chunk_size=chunk_size, rs_freq=rs_freq, save_csv=True
#     )
#     avg = None  # clear memory
    # arctic_reader.write_csv(library=library, item=item, symbol=symbol,
    #                         start=start, end=end, chunk_size=chunk_size)
