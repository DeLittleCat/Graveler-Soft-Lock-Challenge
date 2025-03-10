# The Unlikely Escape

Source code (and data) of my implementations for this challenge. I moved the math to Desmos, since GitHub's $\LaTeX$ rendering is horrific to work with and so I don't have to describe what I'm doing with sets.

# How to use

Put the algorithm you want to run in your python folder, along with its respective npy file. Now run the chosen python file in console or even IDLE. If you want to make your own npy files, run makeBARRAY in the python folder; it will make both. Both are dependent on numba, numpy, and timeit, so pip install them if you haven't already. Yes, that's all you need to do!

# Algorithms

Compilation time is not included in Runtime. Runtime is measured for 4x 1.1GHz
| Algorithm | Runtime | Time Complexity |
| :-------- | :------ | :-------------- |
| MultiThreaded BARRAY(MTB) | $14.91959649999626 s$ | $O(\frac{N}{T})$ |
| Power Transformed BARRAY(PTB) | $0.012722500134259 s$ | $O(\log_2N)$ |

## MultiThreaded BARRAY

A multithreaded max uniform finder. The N for time complexity is the amount of trials($1000000000$); T is the amount of threads your CPU has.

## Power Transformed BARRAY

A transformed BALIAS for one uniform. The N for time complexity is the amount of elements in the array, due to searchsorted. For most purposes, it is $O(1)$, since there will only be $27$ elements in the array, possibly more if you increase precision.

## Building BARRAY

You can access a desmos version of the buidling process here.
<br>
https://www.desmos.com/calculator/cn6syebnbv
