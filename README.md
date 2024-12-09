# The Unlikely Escape

Source code (and data) of my implementations for this challenge.

# How to use

Put the branch you want to run in your python folder, along with its respective npy file. Now run the chosen python file in console or even IDLE. Yes, that's all you need to do!

# Algorithms

Compilation time is not included in Runtime. Runtime is measured for 4x 1.1GHz
| Algorithm | Runtime | Time Complexity |
| :-------- | :------ | :-------------- |
| Multithread | 14.91959649999626 s | O(N/T) |
| Inverse Transform Sampling | 0.029008161963973 s | O(log_2(N)) |

## Multithread

The main branch, multithreaded max uniform generator.

## Inverse Transform Sampling

The ITS branch, single max uniform generator.
