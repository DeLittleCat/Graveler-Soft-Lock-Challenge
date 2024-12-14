# The Unlikely Escape

Source code (and data) of my implementations for this challenge.

# How to use

Put the algorithm you want to run in your python folder, along with its respective npy file. Now run the chosen python file in console or even IDLE. If you want to make your own, run makeBARRAY in the python folder; it will make both. Yes, that's all you need to do!

# Algorithms

Compilation time is not included in Runtime. Runtime is measured for 4x 1.1GHz
| Algorithm | Runtime | Time Complexity |
| :-------- | :------ | :-------------- |
| MultiThreaded BARRAY(MTB) | 14.91959649999626 s | O(N/T) |
| Power Transformed BARRAY(PTB) | 0.029008161963973 s | O(log_2(N)) |

## MultiThreaded BARRAY

A multithreaded max uniform finder that relies on a BALIAS.

## Power Transformed BARRAY

A transformed BALIAS for one uniform.
