# environments/simple_tag_env.py
import gymnasium
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import functools
# pygame import is conditional for rendering

class SimpleTagEnv(ParallelEnv):
    metadata = {
        "name": "simple_tag_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, grid_size=7, max_cycles=100, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.max_cycles = max_cycles
        self.render_mode = render_mode

        self.possible_agents = ["pursuer_0", "evader_0"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agents = []

        self.num_actions = 5
        self._action_spaces = {
            agent: Discrete(self.num_actions) for agent in self.possible_agents
        }
        # Observation: [my_x, my_y, opponent_x, opponent_y]
        self._observation_spaces = {
            agent: Box(low=0, high=self.grid_size - 1, shape=(4,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.agent_positions = {}
        self.current_cycle = 0

        self.window = None
        self.clock = None
        self.cell_size = 50

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def _get_obs(self):
        observations = {}
        pursuer_pos = self.agent_positions["pursuer_0"]
        evader_pos = self.agent_positions["evader_0"]

        observations["pursuer_0"] = np.array(
            [pursuer_pos[0], pursuer_pos[1], evader_pos[0], evader_pos[1]],
            dtype=np.float32,
        )
        observations["evader_0"] = np.array(
            [evader_pos[0], evader_pos[1], pursuer_pos[0], pursuer_pos[1]],
            dtype=np.float32,
        )
        return observations

    def reset(self, seed=None, options=None):
        if seed is not None:
            # Note: For proper seeding with PettingZoo and Gymnasium,
            # it's better to use the provided seeding mechanism in the spaces as well,
            # but for simplicity in this example, we focus on np.random.
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_cycle = 0

        self.agent_positions["pursuer_0"] = np.random.randint(
            0, self.grid_size, size=2
        )
        while True:
            self.agent_positions["evader_0"] = np.random.randint(
                0, self.grid_size, size=2
            )
            if not np.array_equal(
                self.agent_positions["pursuer_0"], self.agent_positions["evader_0"]
            ):
                break

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def _move_agent(self, agent_name, action):
        pos = self.agent_positions[agent_name].copy() # Use copy to avoid modifying original during logic
        if action == 0:  # Up
            pos[1] = min(self.grid_size - 1, pos[1] + 1)
        elif action == 1:  # Down
            pos[1] = max(0, pos[1] - 1)
        elif action == 2:  # Left
            pos[0] = max(0, pos[0] - 1)
        elif action == 3:  # Right
            pos[0] = min(self.grid_size - 1, pos[0] + 1)
        elif action == 4: # Stay
            pass
        self.agent_positions[agent_name] = pos


    def step(self, actions):
        self.current_cycle += 1

        for agent_name, action in actions.items():
            if agent_name in self.agents: # Only move active agents
                 self._move_agent(agent_name, action)

        pursuer_pos = self.agent_positions["pursuer_0"]
        evader_pos = self.agent_positions["evader_0"]

        rewards = {agent: 0.0 for agent in self.possible_agents} # Initialize for all possible
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        caught = np.array_equal(pursuer_pos, evader_pos)

        if caught:
            rewards["pursuer_0"] = 10.0
            rewards["evader_0"] = -10.0
            terminations = {agent: True for agent in self.possible_agents}
        else:
            rewards["pursuer_0"] = -0.1
            rewards["evader_0"] = 0.1

        if self.current_cycle >= self.max_cycles:
            truncations = {agent: True for agent in self.possible_agents}
            if not caught: # If truncated without catch, no explicit reward change here
                pass

        observations = self._get_obs()
        infos = {agent: {} for agent in self.possible_agents}

        # Update self.agents list based on terminations or truncations
        # For ParallelEnv, agents list is typically managed by the environment user (e.g. RLlib)
        # based on the done flags. If an agent is done, it's often not included in the next action dict.
        # However, PettingZoo's ParallelEnv expects rewards/dones for all `possible_agents`
        # if they were active at the start of the step, or if using `.agents` property correctly.
        # If any agent is terminated or truncated, the episode effectively ends for all in many MARL setups.
        if any(terminations.values()) or any(truncations.values()):
            self.agents = [] # Clears active agents, new agents list on next reset
        else:
            # Keep agents list as is if no one terminated/truncated
            # This part can be tricky depending on how RLlib handles ParallelEnv agent cycling.
            # Often, RLlib expects data for all agents listed in `possible_agents` or
            # those returned in `reset()`.
            pass


        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(to_rgb_array=True)
        elif self.render_mode == "human":
            self._render_frame() # Rendering handled in step/reset for human mode

    def _render_frame(self, to_rgb_array=False):
        try:
            import pygame
        except ImportError:
            raise ImportError(
                "pygame is not installed. Please install it to use human rendering."
            )

        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
        if self.render_mode == "human" and self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(
            (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        )
        canvas.fill((255, 255, 255))

        for x_coord in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, (200, 200, 200),
                (x_coord * self.cell_size, 0),
                (x_coord * self.cell_size, self.grid_size * self.cell_size),
            )
        for y_coord in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, (200, 200, 200),
                (0, y_coord * self.cell_size),
                (self.grid_size * self.cell_size, y_coord * self.cell_size),
            )

        pursuer_center = (
            int((self.agent_positions["pursuer_0"][0] + 0.5) * self.cell_size),
            int((self.agent_positions["pursuer_0"][1] + 0.5) * self.cell_size),
        )
        pygame.draw.circle(
            canvas, (255, 0, 0), pursuer_center, int(self.cell_size / 2 * 0.8)
        )

        evader_center = (
            int((self.agent_positions["evader_0"][0] + 0.5) * self.cell_size),
            int((self.agent_positions["evader_0"][1] + 0.5) * self.cell_size),
        )
        pygame.draw.circle(
            canvas, (0, 0, 255), evader_center, int(self.cell_size / 2 * 0.8)
        )

        if to_rgb_array:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        elif self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(10)
            return None

    def close(self):
        if self.window is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.window = None
                self.clock = None
            except ImportError:
                pass # Pygame not available or already quit