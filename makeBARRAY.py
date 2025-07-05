from decimal import Decimal, getcontext
from time import time
import numpy as np
import os


def findZeroAndOne(arr: np.array) -> np.array:
    zo = np.zeros(2, dtype=np.uint8)
    for i in range(len(arr)):
        if arr[i] == 0:
            zo[0] += 1
        elif arr[i] == 1:
            zo[1] += 1
    return zo

def printApprox(arr: np.array, arrName: str) -> None:
    for i in range(len(arr)):
        print(f"{arrName}[{i}]: ~{arr[i]:.{1024}f}".rstrip("0").rstrip("."))
    print("\n")

    
st = time()
getcontext().prec = 2**24
PRINT = True
p = Decimal(1)/4
q = 1 - p
nCk = Decimal(1)
sum = q**231

cdf = np.zeros(232, dtype=Decimal)
cdf[0] = sum
for i in range(1, 232):
    nCk = nCk * (232 - i) // i
    sum += nCk * p**i * q**(231 - i)
    cdf[i] = sum
pcdf = [x**1000000000 for x in cdf]

if PRINT:
    print("Decimal Values: ")
    printApprox(cdf, "cdf")
    printApprox(pcdf, "pcdf")

cdf = [float(x) for x in cdf]
pcdf = [float(x) for x in pcdf]
cdfzo = findZeroAndOne(cdf)
pcdfzo = findZeroAndOne(pcdf)
cdf = cdf[cdfzo[0]:232 - cdfzo[1]]
pcdf = pcdf[pcdfzo[0]:232 - pcdfzo[1]]      

if PRINT:
    print("Stored Values: ")
    printApprox(cdf, "cdf")
    printApprox(pcdf, "pcdf")

np.save(os.path.join(os.path.dirname(__file__), "cdArrayMTB.npy"), cdf)
np.save(os.path.join(os.path.dirname(__file__), "cdArrayPTB.npy"), pcdf)
print("Change the offset of PTB(after np.searchsorted in main) to", pcdfzo[0], "if it's not already.\nDone in:", time() - st, "s")
