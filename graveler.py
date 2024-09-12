from time import time
import numpy as np
import numba as nb


@nb.njit(nb.void(nb.u4), parallel=True)
def main(executions):
    threadBests = np.ones(nb.get_num_threads(), dtype=np.uint8)
    for _ in nb.prange(executions):
        ones, thread = np.sum(np.random.randint(1, 5, size=231) == 1), nb.get_thread_id()
        if ones > threadBests[thread]: threadBests[thread] = ones
    print("Most Ones Rolled:", min(np.max(threadBests), 177), "\nNumber of Executions:", executions)


if __name__ == "__main__": startTime=time(); main(1000000000); print("Done in:",time()-startTime,"s")
