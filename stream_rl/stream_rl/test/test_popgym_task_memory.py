import unittest

import gymnasium as gym
import numpy as np

from stream_rl.experiments.streaming_popgym_frozen_ssm_q import (
    FrozenMemoryFeatures,
)


def dummy_ssm():
    return (
        np.zeros((128,), dtype=np.float32),
        np.zeros((128, 64), dtype=np.float32),
        np.zeros((64, 128), dtype=np.float32),
        np.zeros((64, 64), dtype=np.float32),
    )


class PopGymTaskMemoryTest(unittest.TestCase):
    def make_features(self, observation_space, num_actions, task_memory):
        return FrozenMemoryFeatures(
            observation_space=observation_space,
            num_actions=num_actions,
            constants=dummy_ssm(),
            seed=0,
            use_memory=False,
            use_anchor=False,
            task_memory=task_memory,
        )

    def test_autoencode_stack_replays_observations_in_reverse(self):
        features = self.make_features(
            gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(4))),
            num_actions=4,
            task_memory="autoencode_stack",
        )

        self.assertEqual(np.argmax(features.task_features((1, 1), -1)), 1)
        self.assertEqual(np.argmax(features.task_features((1, 2), 0)), 2)
        self.assertEqual(np.argmax(features.task_features((0, 3), 0)), 3)
        self.assertEqual(np.argmax(features.task_features((0, 0), 3)), 2)
        self.assertEqual(np.argmax(features.task_features((0, 0), 2)), 1)

    def test_concentration_table_retrieves_a_known_mate(self):
        features = self.make_features(
            gym.spaces.MultiDiscrete([3, 3, 3, 3]),
            num_actions=4,
            task_memory="concentration_table",
        )

        features.task_features(np.asarray([1, 2, 2, 2]), previous_action=0)
        features.task_features(np.asarray([1, 2, 0, 2]), previous_action=2)
        match_and_unseen = features.task_features(
            np.asarray([2, 1, 2, 2]), previous_action=1
        )
        self.assertEqual(match_and_unseen[0], 1.0)

    def test_action_mask_is_persistent_until_reset(self):
        features = self.make_features(
            gym.spaces.Discrete(2),
            num_actions=4,
            task_memory="action_mask",
        )

        np.testing.assert_array_equal(
            features.task_features(0, previous_action=2), [1, 1, 0, 1]
        )
        np.testing.assert_array_equal(
            features.task_features(1, previous_action=0), [0, 1, 0, 1]
        )
        features.reset()
        np.testing.assert_array_equal(
            features.task_features(0, previous_action=-1), [1, 1, 1, 1]
        )


if __name__ == "__main__":
    unittest.main()
