# gridworld-localization

This project was completed for the course Artificial Intelligence at Columbia University.

The goal of the project is to implement a grid world localization using Hidden Markov Model (HMM) inference algorithms. The agent moves in a grid environment where each cell can be passable or blocked (e.g. if the cell is a wall). Because the agent's exact position is unknown, a belief state must be maintained over each cell, including blocked cells which always have zero probability.

The agent's movement is modeled such that it can stay in the current cell or move to any adjacent free cell with uniform probability. Adjacent cells include both cardinal and diagonal directions.

Observations are made in each of the four cardinal directions, indicating whether each direction is blocked or free. These observations are noisy with a probability ϵ that each bit may be independently wrong. The probability of an observation given the true state depends on the number of discrepancies between the observed and correct observations.

The main functions I wrote for this project can be found in gridworld_hmm.py. In short, these functions:
- **initT() and init0()**: generates 2 key matrices for grid world localization using HMMs
- **forward()**: computes the forward probability message (alpha) given the previous alpha message and a single observation using matrix-vector multiplication
- **backward()**: computes the backward probability message (beta) given the next beta message and a single observation using matrix-vector multiplication
- **filtering()**: uses forward() to generate a list of observations starting from an initial belief state and returns an N x T NumPy array, where N = number of cells in grid and T = number of observations
- **smoothing()**: similar to filtering() but instead uses backward()

