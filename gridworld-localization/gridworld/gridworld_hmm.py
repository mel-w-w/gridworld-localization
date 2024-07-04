import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: bool = False):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1),
            (i, j), (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1),
        ]
        neighbors = []
        for a in adjacent:
            if a[0] >= 0 and a[0] < M and a[1] >= 0 and a[1] < N and self.grid[a] == 0:
                neighbors.append(a)
        return neighbors

    """
    4.1 Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        M, N = self.grid.shape
        T = np.zeros((M * N, M * N))

        # Iterate through T
        T_j = 0
        for i in range(M):
            for j in range(N):
                # Find neighbors at (i,j)
                neighbors = self.neighbors((i, j))
                # Find number of neighbors s
                s = len(neighbors)
                if s == 0:  # in case of no neighbors
                    probability = 0
                else:
                    probability = 1/s
                # Iterate through neighbors
                for neighbor in neighbors:
                    # insert probability at T[T_i, T_j]
                    T_i = neighbor[0] * N + neighbor[1]
                    T[T_i, T_j] = probability
                # Update T_j
                T_j += 1

        # make sure to specify type in T
        T = T.astype(np.float64)
        # normalize T
        T_normalized = T / np.sum(T, axis=0)
        return T_normalized

    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        M, N = self.grid.shape
        O = np.zeros((16, M * N))

        # interate through O
        for i in range(M):
          for j in range(N):
            # initialize 'correct' observation bitstring to all 0
            bitstring = [0,0,0,0]
            # find neighbors at (i,j)
            neighbors = self.neighbors((i,j))

            # update bitstring depending on neighbors (NESW)
            if (i-1, j) not in neighbors:
              bitstring[0] = 1 # N
            if (i, j+1) not in neighbors:
              bitstring[1] = 1 # E
            if (i+1, j) not in neighbors:
              bitstring[2] = 1 # S
            if (i, j-1) not in neighbors:
              bitstring[3] = 1 # W

            # solve for observation probabilities
            probabilities = []
            # iterate through a range of 0-15
            for val in range(16):
              # format val 
              possible_observation_bit = format(val, '04b')
              # create a list of possible observations
              possible_observations = []
              for char in possible_observation_bit:
                possible_observations.append(int(char))
              
              # compare bitstring and possible_observation
              # by taking the 2 lists n turning values into tuples
              correct_possible = zip(bitstring, possible_observations)
              d = 0
              # go through each tuple
              for correct, possible in correct_possible:
                # compute xor value using ^ operator
                xor = correct ^ possible
                # update d
                if xor == 1:
                  d += 1
              # solve for probability
              probability = ((1-self.epsilon)**(4-d))*((self.epsilon)**(d))
              # append probability to list of probabilities
              probabilities.append(probability)

            # turn list of probabilities into an nd array
            probabilities = np.array(probabilities, dtype = np.float64)
            # update O
            O[:, (i*N+j)] = np.reshape(probabilities, (16,))
        
        return O

    """
    4.2 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO:
        alpha_prime = self.trans@alpha
        alpha_next = self.obs[observation,:] * alpha_prime
        
        return alpha_next

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current "message" of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated message.
        """
        # TODO:
        beta_prime = self.obs[observation,:] * beta
        beta_next = self.trans.T@beta_prime

        return beta_next

    def filtering(self, init: npt.ArrayLike, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO:
        # dimensions for belief_states
        N = self.grid.size
        T = len(observations)
        # belief_states is our nd array of all belief states
        belief_states = np.empty((N, T))

        # call forward() first
        belief_states[:, 0] = self.forward(init, observations[0])

        # iterate through number of observations
        for i in range(1, len(observations)):
          # iteratively call forward() and update belief_states
          belief_states[:, i] = self.forward(belief_states[:, i-1], observations[i])

        # normalize belief_states
        belief_states = belief_states/np.sum(belief_states, axis = 0)

        return belief_states

    def smoothing(self, init: npt.ArrayLike, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Smoothed belief state at each timestep.
        """
        # TODO:
        # dimensions for beta
        T = len(observations)
        N = self.grid.size

        # use filtering() to get alpha
        alpha = self.filtering(init, observations)
        # initialize beta
        beta = np.empty((N, T))

        # saw this on Ed-we need to make sure initial values in init are all 1s
        for val in range(len(init)):
          init[val] = 1

        # Call backward() to update beta
        beta[:, len(observations) - 1] = self.backward(init, observations[T-1])

        # keep calling backward() to update beta
        for val in range(len(observations) - 2, -1, -1):
          beta[:, val] = self.backward(beta[:, val+1], observations[val])
        
        # multiply alpha and beta
        smoothed_belief_state = alpha * beta

        # normalize smoothed_belief_state
        smoothed_belief_state_normalized = smoothed_belief_state / np.sum(smoothed_belief_state, axis = 0)

        return smoothed_belief_state_normalized

    """
    4.3 Localization error
    """

    def loc_error(self, beliefs: npt.ArrayLike, trajectory: list[int]):
        """Compute localization error at each timestep.
        Args:
          beliefs (np.ndarray): Belief state at each timestep.
          trajectory (list[int]): List of states visited.
        Returns:
          list[int]: Localization error at each timestep.
        """
        # TODO:
        all_loc_errors = []
        N = self.grid.shape[1]
        number_of_states_visited = len(trajectory)

        # iterate through # of states visited
        for i in range(number_of_states_visited):
          # actual state - use trajectory[i] and N to solve
          actual_state = np.array([trajectory[i] / N, trajectory[i] % N])
          # solve for predicted_state using argmax of corresponding belief state (given)
          predicted_state = np.array([np.argmax(beliefs[:,i]) / N, np.argmax(beliefs[:,i]) % N])
          # compute sum of absolute x displacement and absolute y distance (basically actual_state & predicted_state)
          # append to list
          all_loc_errors.append(np.sum(np.abs(actual_state-predicted_state)))
      
        return all_loc_errors
