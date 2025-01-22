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

    def __init__(self, 
                 maze, 
                 start_pos=(0,0), 
                 target_pos=None, 
                 max_steps=100):
        """        
        Args:
            maze (array-like): 2D array with 0s for empty cells and 1s for walls.
            start_pos (tuple): Starting position of the agent (row, col). Default is (0, 0).
            target_pos (tuple): Target position in the maze (row, col). 
                                If None, defaults to the bottom-right corner.
            max_steps (int): Maximum number of steps per episode.
        """
        super(MazeEnv, self).__init__()
        
        self.maze = np.array(maze)
        self.height, self.width = self.maze.shape
        
        self.start_pos = start_pos
        if target_pos is not None:
            self.target_pos = target_pos
        else:
            self.target_pos = (self.height - 1, self.width - 1)

        
        # Actions: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 3x3 local view => 9 cells, each cell can be [0..3]
        self.observation_space = spaces.MultiDiscrete([4]*9)
        
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

if __name__ == "__main__":
    # 0 - empty space
    # 1 - wall
    maze_array = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    target = (4, 4)
    
    env = MazeEnv(maze=maze_array, start_pos=start, target_pos=target, max_steps=20)
    
    obs = env.reset()
    env.render()
    
    # Example sequence of actions
    # Actions: 0 - Up, 1 - Down, 2 - Left, 3 - Right
    actions = [1, 1, 3, 1, 1, 3, 3, 3]
    action_mapping = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    
    for action in actions:
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
