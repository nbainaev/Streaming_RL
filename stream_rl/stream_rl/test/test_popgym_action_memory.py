import unittest

import gymnasium as gym
import numpy as np

from stream_rl.experiments.streaming_popgym_action_memory_q import (
    BattleshipActionMemory,
    ConcentrationActionMemory,
)


class ConcentrationActionMemoryTest(unittest.TestCase):
    def test_known_mate_is_retrieved_for_every_position(self):
        memory = ConcentrationActionMemory(
            gym.spaces.MultiDiscrete([3, 3, 3, 3]), num_actions=4
        )
        memory.observe(np.asarray([1, 2, 2, 2]), previous_action=0)
        memory.observe(
            np.asarray([1, 2, 0, 2]), previous_action=2, previous_reward=-0.1
        )
        features, valid = memory.observe(
            np.asarray([2, 1, 2, 2]), previous_action=1
        )

        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertEqual(features[0, 2], 1.0)
        self.assertEqual(features[2, 2], 0.0)

    def test_successfully_matched_cards_are_masked(self):
        memory = ConcentrationActionMemory(
            gym.spaces.MultiDiscrete([3, 3, 3, 3]), num_actions=4
        )
        memory.observe(np.asarray([1, 2, 2, 2]), previous_action=0)
        _, valid = memory.observe(
            np.asarray([1, 1, 2, 2]), previous_action=1, previous_reward=0.1
        )

        self.assertFalse(valid[0])
        self.assertFalse(valid[1])


class BattleshipActionMemoryTest(unittest.TestCase):
    def test_actions_are_never_repeated_and_hits_create_a_frontier(self):
        memory = BattleshipActionMemory(num_actions=16)
        features, valid = memory.observe(1, previous_action=5)

        self.assertFalse(valid[5])
        for neighbor in (1, 4, 6, 9):
            self.assertEqual(features[neighbor, 3], 1.0)

    def test_aligned_hits_create_line_extensions(self):
        memory = BattleshipActionMemory(num_actions=16)
        memory.observe(1, previous_action=5)
        features, _ = memory.observe(1, previous_action=6)

        self.assertEqual(features[4, 7], 1.0)
        self.assertEqual(features[7, 7], 1.0)


if __name__ == "__main__":
    unittest.main()
