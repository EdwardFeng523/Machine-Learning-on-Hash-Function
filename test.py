import tensorflow
import random
import numpy as np


TABLE_SIZE = 1000

def h(x):
    return (25 * x + 30) % TABLE_SIZE

table = [0 for i in range(TABLE_SIZE)]

def insert(x):
    idx = h(x)
    table[idx] = 1

def test(x):
    idx = h(x)
    if table[idx] == 1:
        return True
    else:
        return False


def load_table(percentage):
    for i in range(int(TABLE_SIZE * percentage)):
        item = random.randrange(0, TABLE_SIZE * 5)
        insert(item)

load_table(0.6)

