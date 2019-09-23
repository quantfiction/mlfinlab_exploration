from tqdm import tqdm
import time
from dotenv import find_dotenv, load_dotenv
import os


def func():
    print('GM say it back')


def tqdm_test():
    for i in tqdm(range(50)):
        print(i)
        time.sleep(0.02)
        pass


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    MSG = os.getenv('MSG')
    print(MSG)
