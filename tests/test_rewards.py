"""Tests for rewards module."""

import pytest
import numpy as np
from mini_spice.rewards import reasoner_reward, challenger_reward


class TestReasonerReward:
    """Tests for reasoner reward function."""
    
    def test_correct_answer(self):
        reward = reasoner_reward("42", "42", "integer")
        assert reward == 1.0
    
    def test_incorrect_answer(self):
        reward = reasoner_reward("42", "43", "integer")
        assert reward == 0.0
    
    def test_mcq_correct(self):
        reward = reasoner_reward("A", "A", "mcq")
        assert reward == 1.0
    
    def test_mcq_incorrect(self):
        reward = reasoner_reward("A", "B", "mcq")
        assert reward == 0.0


class TestChallengerReward:
    """Tests for challenger reward function."""
    
    def test_peak_at_50_percent(self):
        """Reward should peak when pass-rate is 50%."""
        correctness_list = [1.0, 1.0, 0.0, 0.0]  # 50% pass-rate
        reward, p = challenger_reward(correctness_list, sigma=0.15)
        
        assert p == 0.5
        # Reward should be high (close to 1.0) at 50%
        assert reward > 0.9
    
    def test_all_correct(self):
        """Reward should decrease when all answers are correct."""
        correctness_list = [1.0, 1.0, 1.0, 1.0]  # 100% pass-rate
        reward, p = challenger_reward(correctness_list, sigma=0.15)
        
        assert p == 1.0
        # Reward should be lower than at 50%
        assert reward < 0.5
    
    def test_all_incorrect(self):
        """Reward should decrease when all answers are incorrect."""
        correctness_list = [0.0, 0.0, 0.0, 0.0]  # 0% pass-rate
        reward, p = challenger_reward(correctness_list, sigma=0.15)
        
        assert p == 0.0
        # Reward should be lower than at 50%
        assert reward < 0.5
    
    def test_reward_curve(self):
        """Test that reward decreases as we move away from 0.5."""
        # Test various pass-rates
        test_cases = [
            ([1.0, 0.0], 0.5),  # 50% - should be highest
            ([1.0, 1.0, 0.0], 0.67),  # ~67% - should be lower
            ([1.0], 1.0),  # 100% - should be lowest
            ([0.0], 0.0),  # 0% - should be lowest
        ]
        
        rewards_at_50 = None
        for correctness, expected_p in test_cases:
            reward, p = challenger_reward(correctness, sigma=0.15)
            
            if abs(p - 0.5) < 0.01:  # Close to 50%
                rewards_at_50 = reward
            
            # Verify pass-rate
            assert abs(p - expected_p) < 0.01
        
        # Verify reward at 50% is higher than at extremes
        assert rewards_at_50 is not None
        reward_100, _ = challenger_reward([1.0], sigma=0.15)
        reward_0, _ = challenger_reward([0.0], sigma=0.15)
        
        assert rewards_at_50 > reward_100
        assert rewards_at_50 > reward_0
    
    def test_reward_clipping(self):
        """Reward should be clipped to [0, 1]."""
        correctness_list = [1.0] * 100  # Very high pass-rate
        reward, _ = challenger_reward(correctness_list, sigma=0.15)
        
        assert 0.0 <= reward <= 1.0
    
    def test_empty_list(self):
        """Test handling of empty list."""
        reward, p = challenger_reward([], sigma=0.15)
        assert reward == 0.0
        assert p == 0.0
    
    def test_sigma_parameter(self):
        """Test that sigma affects reward spread."""
        correctness_list = [1.0, 1.0, 0.0, 0.0]  # 50% pass-rate
        
        reward_narrow, _ = challenger_reward(correctness_list, sigma=0.1)
        reward_wide, _ = challenger_reward(correctness_list, sigma=0.3)
        
        # At exactly 50%, both should be high, but narrow sigma should be higher
        assert reward_narrow > 0.9
        assert reward_wide > 0.5

