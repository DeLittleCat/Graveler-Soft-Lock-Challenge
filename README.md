# The Unlikely Escape

Source code (and data) of my implementations for this challenge.

# How to use

Put the algorithm you want to run in your python folder, along with its respective npy file. Now run the chosen python file in console or even IDLE. If you want to make your own npy files, run makeBARRAY in the python folder; it will make both. Both are dependent on numba, numpy, and timeit, so pip install them if you haven't already. Yes, that's all you need to do!

# Algorithms

Compilation time is not included in Runtime. Runtime is measured for 4x 1.1GHz
| Algorithm | Runtime | Time Complexity |
| :-------- | :------ | :-------------- |
| MultiThreaded BARRAY(MTB) | $14.91959649999626 s$ | $O(\frac{N}{T})$ |
| Power Transformed BARRAY(PTB) | $0.029008161963973 s$ | $O(\log_2N)$ |

## MultiThreaded BARRAY

A multithreaded max uniform finder. The N for time complexity is the amount of trials($1000000000$); T is the amount of threads your CPU has.

## Power Transformed BARRAY

A transformed BALIAS for one uniform. The N for time complexity is the amount of elements in the array, due to searchsorted. For most purposes, it is $O(1)$, since there will only be $27$ elements in the array, possibly more if you increase precision.
<br>
$$u^{\frac{1}{n}} = \text{Max}(\{u_0, u_1,..., u_999999999\})$$
<br>
$$\text{Find}(\text{BCD}^{n}, u) = \text{Find}(\text{BCD}, u^{\frac{1}{n}})$$
<br>
   where:
      <br>
      $\text{Max}(S) \text{ returns the greatest element of }S\text{.}$
      <br>
      $\text{Find}(S, U) \text{ returns the position of the first element of }S\text{ greater than or equal to }U\text{.}$

## Building BARRAY

Makes both BARRAYs by the following.
<br>
$$\text{BCD}(\text{length}, (a, b)) = \{S_k \ | \ S_k = \sum_{i=0}^k \frac{q^i \cdot n!}{b^n \cdot i! \cdot (n-i)!} \ \}, \ k \in \{0, 1, \dots, n\}$$
<br>
   where:
      <br>
      $q = b - a\newline$
      <br>
      $n = \text{length} - 1\newline$
      <br>
$$\text{MTB} = \{ x \ | \ x \in \text{Round}(\text{BCD}) \ \wedge \ 0 < x < 1 \}$$
<br>
$$\text{PTB}(\text{trials}) = \{ x \ | \ x \in \text{Round}(\text{BCD}^{\text{trials}}) \ \wedge \ 0 < x < 1 \}$$
<br>
   where:
      <br>
      $\text{Round}(S) \text{ returns the set of the elements of }S\text{ as the nearest IEEE 754 double.}$
