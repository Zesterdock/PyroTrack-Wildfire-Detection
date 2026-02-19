"""
Wildfire Simulation Environment for RL-based UAV Monitoring
Implements a POMDP with partial observability and belief state tracking.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to reduce memory usage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class WildfireEnv(gym.Env):
    """
    Custom Gymnasium environment for wildfire monitoring.
    
    Grid: 32x32
    Cell states:
        0 = empty
        1 = tree
        2 = burning
        3 = burned
        
    UAV agent observes 5x5 local patch and maintains belief map.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, grid_size=32, observation_size=9, max_steps=500, 
                 fire_spread_prob=0.15, wind_direction='E', wind_strength=0.2):
        super().__init__()
        
        self.grid_size = grid_size
        self.observation_size = observation_size
        self.max_steps = max_steps
        self.fire_spread_prob = fire_spread_prob
        self.wind_direction = wind_direction
        self.wind_strength = wind_strength
        
        # Cell states
        self.EMPTY = 0
        self.TREE = 1
        self.BURNING = 2
        self.BURNED = 3
        
        # Wind direction vectors
        self.wind_vectors = {
            'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1),
            'NE': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1), 'SW': (1, -1)
        }
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_space = spaces.Discrete(5)
        
        # Observation space: local patch + belief map (flattened)
        obs_patch_size = observation_size * observation_size
        belief_map_size = grid_size * grid_size
        
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(obs_patch_size + belief_map_size,),
            dtype=np.float32
        )
        
        # Initialize state
        self.grid = None
        self.uav_pos = None
        self.belief_map = None
        self.steps = 0
        self.burning_cells_detected = set()
        self.total_burning_cells = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize grid with trees
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Add some empty patches (15% empty)
        empty_mask = np.random.rand(self.grid_size, self.grid_size) < 0.15
        self.grid[empty_mask] = self.EMPTY
        
        # Start fire at random location
        fire_row = np.random.randint(5, self.grid_size - 5)
        fire_col = np.random.randint(5, self.grid_size - 5)
        self.grid[fire_row, fire_col] = self.BURNING
        
        # Initialize UAV at opposite corner
        self.uav_pos = [self.grid_size - 5, self.grid_size - 5]
        
        # Initialize belief map (uniform prior)
        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.1
        
        self.steps = 0
        self.burning_cells_detected = set()
        self.total_burning_cells = 1
        
        return self._get_observation(), {}
    
    def step(self, action):
        # Move UAV
        old_pos = self.uav_pos.copy()
        if action == 0:  # up
            self.uav_pos[0] = max(0, self.uav_pos[0] - 1)
        elif action == 1:  # down
            self.uav_pos[0] = min(self.grid_size - 1, self.uav_pos[0] + 1)
        elif action == 2:  # left
            self.uav_pos[1] = max(0, self.uav_pos[1] - 1)
        elif action == 3:  # right
            self.uav_pos[1] = min(self.grid_size - 1, self.uav_pos[1] + 1)
        # action == 4: stay
        
        # Update belief map based on observation
        self._update_belief_map()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Spread fire
        self._spread_fire()
        
        # Update step counter
        self.steps += 1
        
        # Check termination
        terminated = (self.steps >= self.max_steps)
        truncated = False
        
        # Note: Don't terminate when fires burn out - agent should learn to 
        # explore and monitor over full episode length
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self):
        """
        Returns flattened observation: [local_patch, belief_map]
        """
        # Extract local patch around UAV
        half_obs = self.observation_size // 2
        
        # Handle boundary conditions
        row_start = max(0, self.uav_pos[0] - half_obs)
        row_end = min(self.grid_size, self.uav_pos[0] + half_obs + 1)
        col_start = max(0, self.uav_pos[1] - half_obs)
        col_end = min(self.grid_size, self.uav_pos[1] + half_obs + 1)
        
        local_patch = np.zeros((self.observation_size, self.observation_size), dtype=np.float32)
        
        # Calculate padding
        pad_top = half_obs - (self.uav_pos[0] - row_start)
        pad_left = half_obs - (self.uav_pos[1] - col_start)
        
        patch = self.grid[row_start:row_end, col_start:col_end]
        
        local_patch[pad_top:pad_top+patch.shape[0], 
                   pad_left:pad_left+patch.shape[1]] = patch
        
        # Flatten and concatenate with belief map
        obs = np.concatenate([
            local_patch.flatten(),
            self.belief_map.flatten()
        ]).astype(np.float32)
        
        return obs
    
    def _update_belief_map(self):
        """
        Update belief map based on current observation.
        Uses simple Bayesian update in observed region.
        """
        half_obs = self.observation_size // 2
        
        row_start = max(0, self.uav_pos[0] - half_obs)
        row_end = min(self.grid_size, self.uav_pos[0] + half_obs + 1)
        col_start = max(0, self.uav_pos[1] - half_obs)
        col_end = min(self.grid_size, self.uav_pos[1] + half_obs + 1)
        
        # Update belief based on direct observation
        observed_region = self.grid[row_start:row_end, col_start:col_end]
        
        # Set belief to 1.0 for burning cells, 0.0 for non-burning
        for i in range(observed_region.shape[0]):
            for j in range(observed_region.shape[1]):
                grid_i = row_start + i
                grid_j = col_start + j
                
                if observed_region[i, j] == self.BURNING:
                    self.belief_map[grid_i, grid_j] = 1.0
                    self.burning_cells_detected.add((grid_i, grid_j))
                else:
                    self.belief_map[grid_i, grid_j] = 0.0
        
        # Decay belief in unobserved regions
        mask = np.ones((self.grid_size, self.grid_size), dtype=bool)
        mask[row_start:row_end, col_start:col_end] = False
        self.belief_map[mask] *= 0.95
    
    def _calculate_reward(self):
        """
        Reward function:
        +1 for detecting new burning cells
        -0.1 per step (encourage efficiency)
        -1 for undetected burning cells (proportional)
        """
        reward = -0.1  # step penalty
        
        # Count currently burning cells
        burning_positions = set(zip(*np.where(self.grid == self.BURNING)))
        self.total_burning_cells = len(burning_positions)
        
        # Reward for new detections
        new_detections = burning_positions.intersection(self.burning_cells_detected)
        reward += len(new_detections)
        
        # Penalty for undetected burning cells
        undetected = burning_positions - self.burning_cells_detected
        reward -= 0.5 * len(undetected)
        
        # Belief-guided exploration reward: incentivize checking high-belief areas
        half_obs = self.observation_size // 2
        row_start = max(0, self.uav_pos[0] - half_obs)
        row_end = min(self.grid_size, self.uav_pos[0] + half_obs + 1)
        col_start = max(0, self.uav_pos[1] - half_obs)
        col_end = min(self.grid_size, self.uav_pos[1] + half_obs + 1)
        
        belief_in_view = self.belief_map[row_start:row_end, col_start:col_end]
        high_belief_cells = np.sum(belief_in_view > 0.5)
        reward += high_belief_cells * 0.05  # Small reward for investigating suspicious areas
        
        return reward
    
    def _spread_fire(self):
        """
        Spread fire to neighboring cells with probability.
        Wind direction influences spread probability.
        """
        burning_cells = np.argwhere(self.grid == self.BURNING)
        new_burning = []
        
        for row, col in burning_cells:
            # Check all 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    new_row, new_col = row + dr, col + dc
                    
                    # Check bounds
                    if not (0 <= new_row < self.grid_size and 
                           0 <= new_col < self.grid_size):
                        continue
                    
                    # Only spread to trees
                    if self.grid[new_row, new_col] != self.TREE:
                        continue
                    
                    # Calculate spread probability with wind influence
                    spread_prob = self.fire_spread_prob
                    
                    # Check if direction aligns with wind
                    wind_vec = self.wind_vectors.get(self.wind_direction, (0, 0))
                    if (dr, dc) == wind_vec:
                        spread_prob += self.wind_strength
                    elif (dr, dc) == (-wind_vec[0], -wind_vec[1]):
                        spread_prob -= self.wind_strength * 0.5
                    
                    spread_prob = np.clip(spread_prob, 0, 1)
                    
                    if np.random.rand() < spread_prob:
                        new_burning.append((new_row, new_col))
            
            # Burning cell becomes burned
            self.grid[row, col] = self.BURNED
        
        # Update new burning cells
        for row, col in new_burning:
            self.grid[row, col] = self.BURNING
    
    def render(self, mode='rgb_array'):
        """
        Render the environment.
        Returns RGB array of size (128, 128, 3)
        """
        try:
            # Create figure without display backend
            fig = plt.figure(figsize=(4, 4), dpi=32)
            ax = fig.add_subplot(111)
            
            # Color map: empty=white, tree=green, burning=red, burned=black
            colors = ['white', 'green', 'red', 'black']
            cmap = ListedColormap(colors)
            
            # Plot grid
            ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=3)
            
            # Plot UAV position
            ax.plot(self.uav_pos[1], self.uav_pos[0], 'b^', markersize=10, 
                   markeredgecolor='blue', markerfacecolor='cyan')
            
            # Plot observation range
            half_obs = self.observation_size // 2
            rect = plt.Rectangle(
                (self.uav_pos[1] - half_obs - 0.5, self.uav_pos[0] - half_obs - 0.5),
                self.observation_size, self.observation_size,
                fill=False, edgecolor='blue', linewidth=2, linestyle='--'
            )
            ax.add_patch(rect)
            
            ax.set_title(f'Step: {self.steps}, Burning: {self.total_burning_cells}')
            ax.axis('off')
            
            # Convert to RGB array - use simple canvas draw without tight_layout
            # to avoid matplotlib backend issues
            fig.canvas.draw()
            
            # Get RGB array (compatible with newer matplotlib versions)
            # Use buffer_rgba() and convert to RGB
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            image = buf.reshape(h, w, 4)[:, :, :3]  # Drop alpha channel
            
            plt.close(fig)
            
            return image
            
        except Exception as e:
            # If rendering fails (bitmap allocation error), return a simple grid-based image
            print(f"⚠️  Render warning: {e}, using fallback rendering")
            
            # Create simple RGB image from grid state
            h, w = self.grid.shape
            image = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Map states to colors
            # empty=white, tree=green, burning=red, burned=black
            image[self.grid == 0] = [255, 255, 255]  # white
            image[self.grid == 1] = [0, 255, 0]      # green
            image[self.grid == 2] = [255, 0, 0]      # red
            image[self.grid == 3] = [0, 0, 0]        # black
            
            # Resize to expected size (128, 128)
            from scipy.ndimage import zoom
            scale = 128 / self.grid_size
            image = zoom(image, (scale, scale, 1), order=0)
            
            return image
    
    def get_belief_map(self):
        """Return current belief map for visualization."""
        return self.belief_map.copy()
    
    def get_grid(self):
        """Return current grid state."""
        return self.grid.copy()


if __name__ == "__main__":
    # Test the environment
    env = WildfireEnv()
    obs, info = env.reset()
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={terminated}")
        
        if terminated or truncated:
            break
    
    print("\nTest completed successfully!")
