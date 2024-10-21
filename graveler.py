# Please read this before the code, or at least the basic flow at the bottom of this wall of text
# Probability Mass Function (PMF):
# The probability mass function (PMF) models the likelihood of observing exactly k paralyzations (0 <= k <= 231) 
# in 231 trials. This was used in the video. P.S. It helped a lot with this idea :widepeepoHappy:
#
#     PM(k) = (0.25^k * 0.75^(231-k)) * (231! / (k! * (231-k)!))
#
# This formula calculates the exact probability of getting exactly k paralyzations out of 231 moves.
# For numerical stability, it can be rewritten as:
#
#     PM(k) = (3^(231-k) * 231!) / (k! * (231-k)! * 4^231)
#
# Cumulative Density Function (CDF):
# The cumulative density function (CDF) is the running sum of the PMF values. It represents the probability 
# of observing k or fewer paralyzations. For example, CD(177) gives the cumulative probability of getting 
# up to 177 paralyzations.
#
# Proving Random Number Comparison with CDF:
# To determine how many paralyzations occur in a given escape attempt, we compare a random number (between 0 and 1) 
# to the CDF values. Since the CDF is a cumulative probability, each CDF(k) gives the probability of getting 
# up to k paralyzations. By finding the first CDF value greater than or equal to the random number, we can identify 
# the corresponding number of paralyzations.
#
# This works because the CDF partitions the [0, 1] range into intervals, each corresponding to a specific number 
# of paralyzations. For instance:
# - If a random number falls between CD(k-1) and CD(k), this means the number of paralyzations is k-1.
# - The search for the random number in the CDF directly maps the random probability to the number of paralyzations.
#
# Floating Point Precision Considerations:
# Floating point precision errors can arise both in the calculation of the PMF and in the generation of random 
# numbers (due to limitations of floating point in general). These errors can affect the fairness of the simulation:
# - When computing the PMF, particularly for very small probabilities, floating point errors may slightly 
#   distort the actual probability values.
# - Random numbers generated using typical PRNGs are also subject to floating point precision issues, meaning the
#   simulation is not perfectly fair, as it is granular, unlike their pure math counterpart.
#
# In practice, these floating point errors are small, but they are present and unavoidable. The impact is minimal 
# for most cases, but at extremely high or low probabilities, they can lead to slight inaccuracies in how the 
# paralyzation counts are determined.
#
# Basic Flow:
# - Make array of factorials
# - Make array of probability masses using the array of factorials
# - Find the largest of 1 billion random numbers 0-1
# - Make array of cumulative probabilities using the array of probability masses
# - Find the index of that number in the array of cumulative probabilities, or the closest below it
#   (searchsorted goes up, so I couldn't one line it :sadge:)
# - The final result is capped at 177 paralyzations, as that is the required threshold to escape.
#
# Written majoritively by ChatGPT.


import numpy as np
import numba as nb
from timeit import timeit


@nb.njit(nb.f8(nb.u4))
def maxRand(rolls: int) -> float:
    threadMaxes = np.zeros(nb.get_num_threads(), dtype=np.float64)
    for _ in nb.prange(rolls):
        rand, thread = np.random.uniform(), nb.get_thread_id()
        if rand > threadMaxes[thread]: threadMaxes[thread] = rand
    return np.max(threadMaxes)

@nb.njit(nb.void(nb.f8[:], nb.f8, nb.u4))
def printSearch(pdArray: list, flo: float, executions: int) -> None:
    cdArray = np.cumsum(pdArray)
    ind = np.searchsorted(cdArray, flo)
    if cdArray[ind] != flo: ind -= 1
    print("Most Paralyzations:", min(ind, 177), "\nNumber of Executions:", executions)

def main(executions: int) -> None:
    fac = np.arange(0, 232, dtype=object)
    fac[0] = 1
    fac = np.cumprod(fac)
    printSearch(np.array(3 ** np.arange(0, 232, dtype=object)[::-1] * fac[231] / (fac * fac[::-1] * 4 ** 231), dtype=np.float64), maxRand(executions), executions)


if __name__ == "__main__": print("Done in:", timeit(globals=globals(), stmt="main(1E9)", number=1), "s")
