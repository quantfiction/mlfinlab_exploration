from tqdm import tqdm
import time


def func():
    print('GM say it back')


def tqdm_test():
    for i in tqdm(range(50)):
        time.sleep(0.02)
        pass
