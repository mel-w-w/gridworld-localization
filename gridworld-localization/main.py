import numpy as np
import matplotlib.pyplot as plt
from gridworld.gridworld_hmm import Gridworld_HMM


def run(env, epsilons, T, N):
    cells = np.nonzero(env.grid == 0)
    indices = cells[0] * env.grid.shape[1] + cells[1]
    init = np.zeros(env.grid.size)
    init[indices] = 1 / indices.size

    FE = np.zeros((len(epsilons), T))
    SE = np.zeros((len(epsilons), T))

    for e in range(len(epsilons)):
        env.epsilon = epsilons[e]
        env.obs = env.initO()
        for n in range(N):
            trajectory = []
            readings = []
            curr = np.random.choice(indices)
            for t in range(T):
                trajectory.append(
                    np.random.choice(env.trans.shape[0], p=env.trans[:, curr])
                )
                curr = trajectory[-1]
                readings.append(np.random.choice(env.obs.shape[0], p=env.obs[:, curr]))
            FE[e] += env.loc_error(env.filtering(init, readings), trajectory)
            SE[e] += env.loc_error(env.smoothing(init, readings), trajectory)

    return FE / N, SE / N


def aima_world():
    walls = [
        (0, 4),
        (0, 10),
        (0, 14),
        (1, 0),
        (1, 1),
        (1, 4),
        (1, 6),
        (1, 7),
        (1, 9),
        (1, 11),
        (1, 13),
        (1, 14),
        (1, 15),
        (2, 0),
        (2, 4),
        (2, 6),
        (2, 7),
        (2, 13),
        (2, 14),
        (3, 2),
        (3, 6),
        (3, 11),
    ]
    env = Gridworld_HMM((4, 16), 0, walls)
    epsilons = [0.4, 0.2, 0.1, 0.05, 0.02, 0]
    FE, SE = run(env, epsilons, 40, 400)

    for e in range(len(epsilons)):
        plt.plot(FE[e], label="e=%.2f" % epsilons[e])
    plt.legend()
    plt.title("Filtering localization error")
    plt.show()

    for e in range(len(epsilons)):
        plt.plot(SE[e], label="e=%.2f" % epsilons[e])
    plt.legend()
    plt.title("Smoothing localization error")
    plt.show()


def main():
    aima_world()


if __name__ == "__main__":
    main()
