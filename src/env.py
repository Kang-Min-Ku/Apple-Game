import gym
import numpy as np
from gym import spaces

class AppleGameEnv(gym.Env):
    def __init__(self, seed):
        super(AppleGameEnv, self).__init__()
        
        self.seed = seed
        np.random.seed(self.seed)

        # game table
        self.rows = 10
        self.cols = 17
        self.grid_size = (self.rows, self.cols)
        # self.grid = torch.randint(1,10,self.grid_size).to(self.device)
        self.grid = np.random.randint(1,10,self.grid_size)
        # action space
        self.action_space = spaces.MultiDiscrete([self.rows, self.cols, self.rows, self.cols])
        # self.observation_space = spaces.Box(low=1, high=9, shape=self.grid_size, dtype=torch.int32)
        self.observation_space = spaces.Box(low=1, high=9, shape=self.grid_size, dtype=np.int32)
    
    def reset(self):
        # self.grid = torch.randint(1,10,self.grid_size).to(self.device)
        self.grid = np.random.randint(1,10,self.grid_size)

        return self.grid.clone()
    
    def step(self, action):
        x1, y1, x2, y2 = action

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        selected_area = self.grid[x1:x2+1, y1:y2+1]
        masked_area = selected_area[selected_area > 0]
        # step_sum = torch.sum(masked_area)
        step_sum = np.sum(masked_area)

        reward = 0
        done = False

        # reward
        if step_sum == 10:
            prev_zero_count = np.count_nonzero(self.grid == 0)
            self.grid[x1:x2+1, y1:y2+1] = 0
            now_zero_count = np.count_nonzero(self.grid == 0)
            reward = now_zero_count - prev_zero_count

        if not self._has_remaining_moves():
            done = True

        return self.grid, reward, done, {}
    
    def _has_remaining_moves(self):
        # check if there is any possible moves
        max_width, max_height = self.grid_size
        
        # 2 ~ 170
        for size in range(2, max_width*max_height+1):
            total_sum_over = True
            # 1 ~ 170
            for w in range(1, size+1):
                if size % w != 0:
                    continue
                h = size // w

                if w > max_width or h > max_height:
                    continue

                for x in range(max_width - w + 1):
                    for y in range(max_height - h + 1):
                        selected_area = self.grid[x:x+w, y:y+h]
                        masked_area = selected_area[selected_area > 0]
                        # step_sum = torch.sum(masked_area)
                        step_sum = np.sum(masked_area)

                        if step_sum == 10:
                            return True
                        elif step_sum < 10:
                            total_sum_over = False

            if total_sum_over:
                break

        return False
        
                        
    def render(self):
        print(self.grid)

    def close(self):
        pass