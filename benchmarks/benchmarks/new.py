import random
from time import sleep


class Me:
    def tracemalloc_test(self):
        a = [1] * random.randint(1000000, 10000000)

    def time_test(self):
        sleep(1)
