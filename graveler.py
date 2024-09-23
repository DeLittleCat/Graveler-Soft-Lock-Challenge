from timeit import timeit
import numpy as np
import numba as nb


@nb.njit(nb.void(nb.u4), parallel=True)
def main(executions: int) -> None:
    """A function that generates an arbitrary
       amount of random binomials and prints
       the largest or 177, whichever is
       smaller, alongside the arbitrary amount."""
    
    threadMaxes = np.zeros(nb.get_num_threads(), dtype=np.uint8)
    for _ in nb.prange(executions):
        paras, thread = np.random.binomial(n=231, p=0.25), nb.get_thread_id()
        if paras > threadMaxes[thread]: threadMaxes[thread] = paras
    print("Most Paralyzations:", min(np.max(threadMaxes), 177), "\nNumber of Executions:", executions)
    

if __name__ == "__main__": print("Done in:", timeit(globals=globals(), stmt="main(1E9)", number=1), "s")
