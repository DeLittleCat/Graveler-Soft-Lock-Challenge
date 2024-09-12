from time import time
import numpy as np
import numba as nb


@nb.njit(nb.void(nb.u4), parallel=True)
def main(executions):
    threadBests = np.ones(nb.get_num_threads(), dtype=np.uint8)
    for _ in nb.prange(executions):
        ones, thread = np.random.binomial(n=231, p=0.25), nb.get_thread_id()
        if ones > threadBests[thread]: threadBests[thread] = ones
    print("Most Ones Rolled:", min(np.max(threadBests), 177), "\nNumber of Executions:", executions)

if __name__ == "__main__": startTime = time(); main(1E9); print("Done in:", time() - startTime, "s")
