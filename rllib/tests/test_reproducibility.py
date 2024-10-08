import gymnasium as gym
import numpy as np
import unittest

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils.metrics import (
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MIN,
    ENV_RUNNER_RESULTS,
)
from ray.tune.registry import register_env


class TestReproducibility(unittest.TestCase):
    def test_reproducing_trajectory(self):
        class PickLargest(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=(4,)
                )
                self.action_space = gym.spaces.Discrete(4)

            def reset(self, *, seed=None, options=None):
                self.obs = np.random.randn(4)
                return self.obs, {}

            def step(self, action):
                reward = self.obs[action]
                return self.obs, reward, True, False, {}

        def env_creator(env_config):
            return PickLargest()

        trajs = []
        for trial in range(3):
            ray.init()
            register_env("PickLargest", env_creator)
            config = (
                DQNConfig()
                .environment("PickLargest")
                .debugging(seed=666 if trial in [0, 1] else 999)
                .reporting(
                    min_time_s_per_iteration=0,
                    min_sample_timesteps_per_iteration=100,
                )
            )
            algo = config.build()

            trajectory = list()
            for _ in range(8):
                r = algo.train()
                trajectory.append(r[ENV_RUNNER_RESULTS][EPISODE_RETURN_MAX])
                trajectory.append(r[ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN])
            trajs.append(trajectory)

            algo.stop()
            ray.shutdown()

        # trial0 and trial1 use same seed and thus
        # expect identical trajectories.
        all_same = True
        for v0, v1 in zip(trajs[0], trajs[1]):
            if v0 != v1:
                all_same = False
        self.assertTrue(all_same)

        # trial1 and trial2 use different seeds and thus
        # most rewards tend to be different.
        diff_cnt = 0
        for v1, v2 in zip(trajs[1], trajs[2]):
            if v1 != v2:
                diff_cnt += 1
        self.assertTrue(diff_cnt > 8)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
