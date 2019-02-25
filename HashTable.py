from sklearn.utils import murmurhash3_32
import random


def hashfunc(m):
    rand = random.randint(0, 2 ** 32 - 1)
    def func(key):
        return (murmurhash3_32(key, seed=rand, positive=True) % m)
    return func


#####

class BloomFilter:
    def __init__(self, r, n):
        self.n = n
        self.r = r
        self.k = 1
        self.functions = []
        for i in range(self.k):
            self.functions.append(hashfunc(self.r))
        self.table = [0 for dummy in range(self.r)]

    def insert(self, key):
        for func in self.functions:
            bit = func(key)
            self.table[bit] = 1

    def test(self, key):
        for func in self.functions:
            bit = func(key)
            if self.table[bit] == 0:
                return False
        return True
