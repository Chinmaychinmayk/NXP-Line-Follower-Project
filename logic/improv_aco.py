import numpy as np

class ImprovedACO:
    def __init__(self, grid, start, goal, num_ants, num_iterations, alpha=1, beta=2, gamma=1, rho=0.5, Q=100):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.Q = Q
        self.pheromone = np.ones(grid.shape)
        self.heuristic = 1 / (np.array(grid.shape) - 1)
        self.best_path = None
        self.best_length = float('inf')

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def compute_transition_probabilities(self, ant_path):
        probs = np.zeros(self.grid.shape)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 0 and (i, j) not in ant_path:
                    tau = self.pheromone[i, j] ** self.alpha
                    eta = self.heuristic[i, j] ** self.beta
                    nu = (1 / (1 + len(ant_path))) ** self.gamma
                    probs[i, j] = tau * eta * nu
        return probs / np.sum(probs)

    def update_pheromone(self, paths, lengths):
        self.pheromone *= (1 - self.rho)
        for path, length in zip(paths, lengths):
            for (i, j) in path:
                self.pheromone[i, j] += self.Q / length

    def run(self):
        for iteration in range(self.num_iterations):
            paths = []
            lengths = []
            for ant in range(self.num_ants):
                ant_path = [self.start]
                current_pos = self.start
                while current_pos != self.goal:
                    probs = self.compute_transition_probabilities(ant_path)
                    next_pos = tuple(np.unravel_index(np.argmax(probs), probs.shape))
                    ant_path.append(next_pos)
                    current_pos = next_pos
                paths.append(ant_path)
                lengths.append(self.path_length(ant_path))
            self.update_pheromone(paths, lengths)
            min_length = min(lengths)
            if min_length < self.best_length:
                self.best_length = min_length
                self.best_path = paths[lengths.index(min_length)]
            print(f"Iteration {iteration + 1}/{self.num_iterations}, Best Length: {self.best_length}")

    def path_length(self, path):
        return sum(self.distance(path[i], path[i + 1]) for i in range(len(path) - 1))

# Example usage
grid = np.zeros((10, 10))
start = (0, 0)
goal = (9, 9)
aco = ImprovedACO(grid, start, goal, num_ants=10, num_iterations=50)
aco.run()
print(f"Best path found: {aco.best_path}")


## WE NEED TO KEEP MODIFYING GOAL SO THAT THE BUGGY MOVES FORWARD. ELSE IT WILL STOP IN BETWEEN
# WE ALSO NEED TO CHANGE VALUES OF ALPHA BETA Q AND OTHER CONSTANTS DEPENDING ON OUR REQUIREMENTS. BETA > ALPHA (SHORTER PATH MUST BE OUR PRIORITY)
