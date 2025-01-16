import numpy as np
import os
from decimal import Decimal, getcontext
from timeit import timeit


#Increase accuracy ▼                       ▼ Print out certain containers/values
def main(prec: int=2**20, printExact: bool=False) -> None:
    def decList(length: int, fract: tuple) -> list:
        def facList(length: int, p: int=1) -> list:
            return [1] + [p := p * i for i in range(1, length)]

        def cumsum(lis: list, s: int=0) -> list: # For Decimal objects
            return [s := s + x for x in lis]
    
        qn, fac, length = fract[1] - fract[0], facList(length), length - 1
        dlf, pdPowLen = Decimal(fac[-1]), fract[1]**length
        return cumsum([qn**i * dlf / (pdPowLen * fac[i] * fac[length - i]) for i in range(length, -1, -1)])
    def fixcd(cdArray: list) -> list:
        return np.unique(cdArray[(cdArray > 0)])
    getcontext().prec = prec
    decLis = decList(232, (1,4))
    floArrMTB, floArrPTB = np.array([float(x) for x in decLis], dtype=np.float64), np.array([float(x**1000000000) for x in decLis], dtype=np.float64)
    cdArrMTB, cdArrPTB = fixcd(floArrMTB), fixcd(floArrPTB)
    np.save(os.path.join(os.path.dirname(__file__), "cdArrayMTB.npy"), cdArrMTB)
    np.save(os.path.join(os.path.dirname(__file__), "cdArrayPTB.npy"), cdArrPTB)
    if printExact:
        print("Exact Decimals:")
        for x in decLis: print(f"{x:.{prec}f}".rstrip("0").rstrip("."))
        print("Exact Floats(for MTB):")
        for x in floArrMTB: print(f"{x:.{1074}f}".rstrip("0").rstrip("."))
        print("Exact Floats(for PTB):")
        for x in floArrPTB: print(f"{x:.{1074}f}".rstrip("0").rstrip("."))
        print("End results:\n   MTB:", cdArrMTB, "\n   PTB:", cdArrPTB)
    print("For PTB, change the added number to:", np.nonzero(floArrPTB)[0][0])


if __name__ == "__main__": print("Done in:", timeit(globals=globals(), stmt="main()", number=1), "s")
    

