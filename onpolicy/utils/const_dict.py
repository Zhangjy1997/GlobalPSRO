import time
import numpy as np
from pympler import asizeof

"""
Implementation of ConstDict from https://github.com/Grente/ConstDict
"""


class ConstDict(object):
    __slots__ = []

    def __init__(self):
        for key in self.__slots__:
            setattr(self, key, None)

    def __iter__(self):
        return iter(self.__slots__)

    def __getitem__(self, item):
        if not isinstance(item, str):
            return None
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, item):
        return hasattr(self, item)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return str(dict(self.items()))

    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def update(self, dic):
        for key, val in dic.items():
            if key in self.__slots__:
                setattr(self, key, val)
    
    def clear(self):
        for key in self.__slots__:
            if hasattr(self, key):
                delattr(self, key)
    
    def setdefault(self, key, val):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            setattr(self, key, val)
            return val
    
    def pop(self, key, default=None):
        if hasattr(self, key):
            val = getattr(self, key)
            delattr(self, key)
            return val
        return default
        
    def items(self):
        return [(key, getattr(self, key, None)) for key in self.__slots__ if hasattr(self, key)]

    def keys(self):
        return [key for key in self.__slots__ if hasattr(self, key)]

    def values(self):
        return [getattr(self, key, None) for key in self.__slots__ if hasattr(self, key)]

    

if __name__ == "__main__":
    max_n = 200000
    n_sample = 150000
    random_keys = np.random.choice(max_n, n_sample, replace=False)
    random_values = np.random.choice(max_n, n_sample, replace=False)

    # test_condict = ConstDict(dict())
    base_dict = dict()

    for i in range(len(random_keys)):
        # test_condict[random_keys[i]] = random_values[i]
        base_dict["A" + str(random_keys[i])] = random_values[i]

    class M_dict(ConstDict):
        __slots__ = base_dict.keys()

    test_condict = M_dict()

    for key in base_dict.keys():
        test_condict[key] = base_dict[key]

    test_flag1 = True
    for key in base_dict.keys():
        v1 = base_dict[key]
        v2 = test_condict[key]
        if v1 != v2:
            test_flag1 = False
            break

    print("flag of test_1 = ", test_flag1)

    test_flag2 = True
    for key in test_condict.keys():
        v1 = base_dict[key]
        v2 = test_condict[key]
        if v1 != v2:
            test_flag2 = False
            break

    print("flag of test_2 = ", test_flag2)

    test_flag3 = True
    for key, v1 in test_condict.items():
        v2 = base_dict[key]
        if v1 != v2:
            test_flag3 = False
            break

    print("flag of test_3 = ", test_flag3)

    print("Memory usage of raw dict = ", asizeof.asizeof(base_dict)/1048576)
    print("Memory usage of reduced dict = ", asizeof.asizeof(test_condict)/1048576)
