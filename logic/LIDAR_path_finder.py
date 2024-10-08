import numpy as np

# Constants
GRID_SIZE = 50
OBSTACLE_PROBABILITY = 0.3
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
ROBOT_START = (0, 0)
ALPHA = 1.0
BETA = 2.0
OMEGA = 1.0

# Generate a grid with random obstacles
def generate_grid(size, obstacle_prob):
    grid = np.zeros((size, size))
    obstacles = np.random.rand(size, size) < obstacle_prob
    grid[obstacles] = 1
    return grid

# Convert LiDAR data to grid coordinates
def lidar_to_grid(lidar_data, robot_pose, grid_size):
    grid = np.zeros((grid_size, grid_size))
    for distance, angle in lidar_data:
        x = int(robot_pose[0] + distance * np.cos(angle))
        y = int(robot_pose[1] + distance * np.sin(angle))
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[x, y] = 1
    return grid

# Define the cost function
def cost_function(grid, goal, alpha, beta, omega):
    rows, cols = grid.shape
    cost = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                cost[i, j] = np.inf
            else:
                dist = np.sqrt((goal[0] - i)*2 + (goal[1] - j)*2)
                angle1 = np.arctan2(goal[1] - j, goal[0] - i)
                angle2 = np.arctan2(goal[1] - i, goal[0] - j)
                cost[i, j] = alpha * dist + beta * np.abs(angle1) + omega * np.abs(angle2)
    return cost

# Pathfinding using a simplified Ant Colony Optimization (ACO)
def aco_pathfinding(grid, start, goal, alpha, beta, omega, n_ants=10, n_iterations=50):
    rows, cols = grid.shape
    pheromones = np.ones((rows, cols))
    best_path = []
    best_cost = np.inf
    
    for _ in range(n_iterations):
        paths = []
        costs = []
        
        for _ in range(n_ants):
            path = [start]
            current = start
            while current != goal:
                x, y = current
                next_step = None
                min_cost = np.inf
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                        step_cost = pheromones[nx, ny] * (alpha * np.sqrt((goal[0] - nx)*2 + (goal[1] - ny)*2))
                        if step_cost < min_cost:
                            min_cost = step_cost
                            next_step = (nx, ny)
                
                if next_step is None:
                    break
                
                path.append(next_step)
                current = next_step
            
            path_cost = sum(cost_function(grid, goal, alpha, beta, omega)[x, y] for x, y in path)
            paths.append(path)
            costs.append(path_cost)
            
            if path_cost < best_cost:
                best_cost = path_cost
                best_path = path
        
        # Update pheromones
        pheromones *= 0.9
        for path in paths:
            for x, y in path:
                pheromones[x, y] += 1 / (1 + costs[paths.index(path)])
    
    return best_path

# Define car movement commands
def move_forward(speed):
    print(f"Move forward with speed {speed}")

def move_backward(speed):
    print(f"Move backward with speed {speed}")

def turn_left(angle):
    print(f"Turn left with angle {angle}")

def turn_right(angle):
    print(f"Turn right with angle {angle}")

# Command generator function
def generate_commands(path):
    commands = []
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        
        if x1 == x2 and y1 == y2:
            continue  # Skip if no movement

        if x1 == x2:
            if y2 > y1:
                commands.append(("move_forward", 1))
            else:
                commands.append(("move_backward", 1))
        elif y1 == y2:
            if x2 > x1:
                commands.append(("move_forward", 1))
            else:
                commands.append(("move_backward", 1))
        
        # Turn left or right based on direction change
        if i > 1:
            prev_x, prev_y = path[i - 2]
            if (x1 != prev_x and y1 == prev_y) or (x1 == prev_x and y1 != prev_y):
                if (x2 > x1) and (y2 > y1):
                    commands.append(("turn_right", 90))
                elif (x2 < x1) and (y2 < y1):
                    commands.append(("turn_left", 90))
                elif (x2 > x1) and (y2 < y1):
                    commands.append(("turn_left", 90))
                elif (x2 < x1) and (y2 > y1):
                    commands.append(("turn_right", 90))
    
    return commands

# Main
if __name__ == "_main_":
    grid = generate_grid(GRID_SIZE, OBSTACLE_PROBABILITY)
    print("Grid with obstacles:")
    print(grid)
    
    # Simulate LiDAR data (for example purposes)
    lidar_data = [(5, np.pi/4), (10, np.pi/2), (15, 3*np.pi/4), (20, np.pi)]  # (distance, angle)
    robot_pose = ROBOT_START
    
    # Update grid based on LiDAR data
    lidar_grid = lidar_to_grid(lidar_data, robot_pose, GRID_SIZE)
    updated_grid = np.maximum(grid, lidar_grid)
    
    print("Updated Grid with LiDAR data:")
    print(updated_grid)
    
    # Pathfinding
    path = aco_pathfinding(updated_grid, ROBOT_START, GOAL, ALPHA, BETA, OMEGA)
    print("Path found:")
    print(path)
    
    # Generate and print commands
    commands = generate_commands(path)
    for command in commands:
        action, value = command
        if action == "move_forward":
            move_forward(value)
        elif action == "move_backward":
            move_backward(value)
        elif action == "turn_left":
            turn_left(value)
        elif action == "turn_right":
            turn_right(value)
