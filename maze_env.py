import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class MazeEnv(gym.Env):
    """
    A single-maze environment for reinforcement learning agents.
    
    The observation is a 3x3 local view around the agent:
      - 0: free space
      - 1: wall
      - 2: target (only marked if within the 3x3 view)
      - 3: out-of-bounds (or "outside" the maze)
    
    The agent receives a reward of -1 for each step to encourage reaching the target quickly.
    Reaching the target yields a reward of +10.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, maze_file, max_steps=100):
        """        
        Args:
            maze_file (str): Path to the exported maze file.
            max_steps (int): Maximum number of steps per episode.
        """
        super(MazeEnv, self).__init__()

        # Load maze from file
        self.load_maze_from_file(maze_file)

        # Define gym spaces
        self.action_space = spaces.Discrete(4) # 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.observation_space = spaces.MultiDiscrete([4] * 9) # Observation space: 3x3 local view => 9 cells, each cell can be [0..3]

        # Initialize state variables
        self.agent_pos = self.start_pos
        self.current_steps = 0
        self.max_steps = max_steps

    def reset(self):
        """
        Resets the environment to the initial state.
        
        Returns:
            np.ndarray: The initial observation (flattened 3x3 grid).
        """
        self.agent_pos = self.start_pos
        self.current_steps = 0
        return self._get_observation()

    def step(self, action):
        """
        Executes an action in the environment.
        
        Args:
            action (int): An integer representing the action to take.
                          0 - Up, 1 - Down, 2 - Left, 3 - Right.
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_steps += 1
        row, col = self.agent_pos

        # Define movement based on action
        movement = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

        if action not in movement:
            raise ValueError("Invalid Action! Action must be one of [0, 1, 2, 3].")

        delta_row, delta_col = movement[action]
        new_row = row + delta_row
        new_col = col + delta_col

        # Validate new position
        if self._is_valid_position((new_row, new_col)):
            self.agent_pos = (new_row, new_col)
        else:
            # Invalid move; agent stays in place
            pass

        # Initialize reward and done flag
        reward = -1  # Step penalty to encourage efficiency
        done = False
        info = {}

        # Check if the agent has reached the target
        if self.agent_pos == self.target_pos:
            reward = 10
            done = True

        # Check if maximum steps have been exceeded
        if self.current_steps >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """
        Constructs the 3x3 local view around the agent.
        
        Returns:
            np.ndarray: Flattened array representing the local view.
        """
        row, col = self.agent_pos
        obs_grid = np.zeros((3,3), dtype=int)

        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                r = row + r_offset
                c = col + c_offset
                rr = r_offset + 1  # Row index in 3x3 grid
                cc = c_offset + 1  # Column index in 3x3 grid

                if not self._is_within_bounds(r, c):
                    obs_grid[rr, cc] = 3  # Out-of-bounds
                elif self.maze[r, c] == 1:
                    obs_grid[rr, cc] = 1  # Wall
                elif (r, c) == self.target_pos:
                    obs_grid[rr, cc] = 2  # Target
                else:
                    obs_grid[rr, cc] = 0  # Free space

        return obs_grid.flatten()

    def render(self, mode='human'):
        """
        Renders the current state of the maze using matplotlib.
        
        Args:
            mode (str): The mode in which to render. Currently only 'human' is supported.
        """
        if mode != 'human':
            raise NotImplementedError("Only 'human' render mode is supported.")

        # Define color mapping
        cmap = colors.ListedColormap(['white', 'black', 'green', 'blue'])
        # 0: Free space (white)
        # 1: Wall (black)
        # 2: Target (green)
        # 3: Agent (blue)

        # Create a copy of the maze for visualization
        maze_visual = np.copy(self.maze)

        # Mark the target
        tgt_r, tgt_c = self.target_pos
        maze_visual[tgt_r, tgt_c] = 2  # Target as 2

        # Mark the agent
        agent_r, agent_c = self.agent_pos
        maze_visual = np.where(
            (np.indices(maze_visual.shape)[0] == agent_r) & 
            (np.indices(maze_visual.shape)[1] == agent_c),
            3,  # Agent as 3
            maze_visual
        )

        plt.figure(figsize=(5,5))
        plt.imshow(maze_visual, cmap=cmap, vmin=0, vmax=3)
        plt.title("Single Maze Environment")
        plt.xticks([]), plt.yticks([])

        # Create a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Free Space'),
            Patch(facecolor='black', edgecolor='black', label='Wall'),
            Patch(facecolor='green', edgecolor='black', label='Target'),
            Patch(facecolor='blue', edgecolor='black', label='Agent')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.show()

    def close(self):
        """Closes the rendering window."""
        plt.close()

    def _is_within_bounds(self, row, col):
        """
        Checks if a position is within the maze boundaries.
        
        Args:
            row (int): Row index.
            col (int): Column index.
        
        Returns:
            bool: True if within bounds, False otherwise.
        """
        return 0 <= row < self.height and 0 <= col < self.width

    def _is_valid_position(self, pos):
        """
        Validates if a position is within bounds and not a wall.
        
        Args:
            pos (tuple): (row, col) position.
        
        Returns:
            bool: True if position is valid, False otherwise.
        """
        row, col = pos
        return self._is_within_bounds(row, col) and self.maze[row, col] == 0

    def load_maze_from_file(self, file_path):
        """
        Loads a maze from a text file and sets up the environment.

        Args:
            file_path (str): Path to the exported maze file.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Read metadata (size and start direction)
        size_line = lines[0].strip().split(": ")[1]
        rows, cols = map(int, size_line.split(", "))

        direction_line = lines[1].strip().split(": ")[1]
        self.start_direction = direction_line  # Store start direction (optional for RL)

        # Read the maze grid
        maze_grid = []
        self.start_pos = None
        self.target_pos = None

        for r, line in enumerate(lines[3:]):  # Skip first two metadata lines and one blank line
            row_data = line.strip().split()
            maze_row = []

            for c, cell in enumerate(row_data):
                if cell == "W":
                    maze_row.append(1)  # Wall
                elif cell == "0":
                    maze_row.append(0)  # Free space
                elif cell == "S":
                    maze_row.append(0)  # Start position (considered free)
                    self.start_pos = (r, c)
                elif cell == "T":
                    maze_row.append(0)  # Target position (considered free)
                    self.target_pos = (r, c)
                elif cell == "R":
                    maze_row.append(0)  # Reward (can be handled separately)
                elif cell == "D":
                    maze_row.append(0)  # Danger (can be handled separately)

            maze_grid.append(maze_row)

        self.maze = np.array(maze_grid)
        self.height, self.width = self.maze.shape

        # Default target if not found
        if self.target_pos is None:
            self.target_pos = (self.height - 2, self.width - 2)  # Set default target (inside border)

if __name__ == "__main__":
    maze_file = "demo_maze.txt"  # Replace with your own maze

    env = MazeEnv(maze_file=maze_file, max_steps=20)

    obs = env.reset()
    env.render()
    
    # Actions: 0 - Up, 1 - Down, 2 - Left, 3 - Right
    action_mapping = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

    # Example sequence of actions.
    demo_actions_set_1 = [1, 1, 3, 1, 1, 3, 3, 3]  # Fails
    demo_actions_set_1 = [1, 1, 1, 1, 3, 3, 3, 3]  # Succeeds

    # Set your own action sequence here
    agent_action_sequence = demo_actions_set_1
    
    for action in agent_action_sequence:
        obs, reward, done, info = env.step(action)
        print(f"Action: {action_mapping[action]} ({action}), Position: {env.agent_pos}, Reward: {reward}, Done: {done}")
        env.render()
        if done:
            if env.agent_pos == env.target_pos:
                print("Target reached!")
            else:
                print("Maximum steps exceeded!")
            break
    
    env.close()
