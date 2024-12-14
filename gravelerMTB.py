import os
import numpy as np
import numba as nb
from timeit import timeit


@nb.njit(nb.void(nb.f8[:]))
def main(cdArray: list) -> None:
    threadMaxes = np.zeros(nb.get_num_threads(), dtype=np.float64)
    for _ in nb.prange(1000000000):
        rand, thread = np.random.uniform(), nb.get_thread_id()
        if rand > threadMaxes[thread]: threadMaxes[thread] = rand
    ind = np.searchsorted(cdArray, np.max(threadMaxes))
    print("Most Paralyzations:", ind - 1 if ind != 0 else ind, "\nNumber of Executions: 1000000000")


if __name__ == "__main__": print("Done in:", timeit(globals=globals(), stmt=r"main(np.load(os.path.join(os.path.dirname(__file__), 'cdArrayMTB.npy')))", number=1), "s")
