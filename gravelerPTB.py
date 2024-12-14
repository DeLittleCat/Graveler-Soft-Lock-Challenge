import os
import numpy as np
import numba as nb
from timeit import timeit


@nb.njit(nb.void(nb.f8[:]))
def main(cdArray: list) -> None:
    print("Most Paralyzations:", np.searchsorted(cdArray, np.random.uniform(0, 1)) + 90, "\nNumber of Executions: 1000000000")


if __name__ == "__main__": print("Done in:", timeit(globals=globals(), stmt=r"main(np.load(os.path.join(os.path.dirname(__file__), 'cdArrayPTB.npy')))", number=1), "s")
