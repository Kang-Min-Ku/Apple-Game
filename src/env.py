import gym
import numpy as np
from gym import spaces
from typing import Optional, Callable, Dict

class AppleGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "agent"]}

    def __init__(self, render_mode: Optional[str] = None, limit_step: int = 200):
        super(AppleGameEnv, self).__init__()

        self.render_mode = render_mode
        self.limit_step = limit_step
        self.current_step = 0    
        self.info_fn = None 

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

    
    def reset(self, seed: Optional[int] = None, options = None):
        # self.grid = torch.randint(1,10,self.grid_size).to(self.device)
        
        if seed is not None:
            np.random.seed(seed)

        self.grid = np.random.randint(1,10,self.grid_size)
        self.current_step = 0
        obs = self.grid.copy()
        info = self.get_info()

        return obs, info
    
    
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
        truncated = False

        # reward
        if step_sum == 10:
            prev_zero_count = np.count_nonzero(self.grid == 0)
            self.grid[x1:x2+1, y1:y2+1] = 0
            now_zero_count = np.count_nonzero(self.grid == 0)
            reward = now_zero_count - prev_zero_count

        self.current_step += 1
        if self.current_step > self.limit_step:
            truncated = True

        if not self._has_remaining_moves():
            done = True

        obs = self.grid.copy()
        info = self.get_info()

        return obs, reward, done, truncated, info
    
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
        print(self.render_mode)
        print(self.grid)


    def close(self):
        pass


    def set_info(self, info_fn: Callable[[], Dict]):
        self.info_fn = info_fn

    
    def get_info(self):
        return self.info_fn() if self.info_fn is not None else {}