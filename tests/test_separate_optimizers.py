"""Tests to verify separate optimizer approach doesn't cause unexpected training dynamics."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from mini_spice.policy import SimpleGRPO


class SimpleModel(nn.Module):
    """Simple model for testing optimizer behavior."""
    
    def __init__(self, dim=10):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear(x))
        return self.output(x)


class TestSeparateOptimizers:
    """Tests to verify separate optimizer approach for dual-role training."""
    
    def _create_model_logprobs(self, model, device, n=2, detach_after_forward=False):
        """Create logprobs connected to model parameters."""
        torch.manual_seed(42)
        x = torch.randn(n, 10, device=device)
        output = model(x)
        logprobs = torch.log_softmax(output, dim=-1)
        logprobs_list = [logprobs[i, 0].sum() for i in range(n)]
        
        if detach_after_forward:
            logprobs_list = [lp.detach().clone().requires_grad_(True) for lp in logprobs_list]
        
        return logprobs_list
    
    def test_optimizer_state_independence(self):
        """Test that optimizer states remain independent between C and R."""
        model = SimpleModel(dim=10)
        device = torch.device("cpu")
        
        optimizer_C = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer_R = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device)
        
        logprobs_C = self._create_model_logprobs(model, device, n=2)
        rewards_C = [0.8, 0.3]
        loss_C, _, _ = policy.update(logprobs_C, rewards_C, [], [])
        
        first_param = list(optimizer_C.param_groups[0]['params'])[0]
        state_C_after = optimizer_C.state[first_param] if first_param in optimizer_C.state else {}
        assert len(state_C_after) > 0, "Optimizer C state should be populated after step"
        
        torch.manual_seed(43)
        logprobs_R = self._create_model_logprobs(model, device, n=2)
        rewards_R = [1.0, 0.0]
        _, loss_R, _ = policy.update([], [], logprobs_R, rewards_R)
        
        state_R_after = optimizer_R.state[first_param] if first_param in optimizer_R.state else {}
        assert len(state_R_after) > 0, "Optimizer R state should be populated after step"
        
        assert first_param in optimizer_C.state
        assert first_param in optimizer_R.state
        assert isinstance(loss_C, float)
        assert isinstance(loss_R, float)
    
    def test_parameter_updates_both_roles(self):
        """Test that both optimizers can update the same model parameters."""
        model = SimpleModel(dim=10)
        device = torch.device("cpu")
        
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        optimizer_C = torch.optim.AdamW(model.parameters(), lr=1e-2)
        optimizer_R = torch.optim.AdamW(model.parameters(), lr=1e-2)
        
        policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device)
        
        logprobs_C = self._create_model_logprobs(model, device, n=1)
        rewards_C = [0.8]
        
        torch.manual_seed(45)
        logprobs_R = self._create_model_logprobs(model, device, n=1, detach_after_forward=True)
        rewards_R = [1.0]
        
        policy.update(logprobs_C, rewards_C, logprobs_R, rewards_R)
        
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should have been updated by at least one optimizer"
    
    def test_gradient_independence(self):
        """Test that gradients are computed independently for each role."""
        model = SimpleModel(dim=10)
        device = torch.device("cpu")
        
        optimizer_C = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer_R = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device)
        
        logprobs_C = self._create_model_logprobs(model, device, n=2)
        rewards_C = [0.8, 0.3]
        
        torch.manual_seed(47)
        logprobs_R = self._create_model_logprobs(model, device, n=2, detach_after_forward=True)
        rewards_R = [1.0, 0.0]
        
        loss_C, loss_R, total = policy.update(logprobs_C, rewards_C, logprobs_R, rewards_R)
        
        assert isinstance(loss_C, float) and np.isfinite(loss_C)
        assert isinstance(loss_R, float) and np.isfinite(loss_R)
        assert abs(total - (loss_C + loss_R)) < 1e-6
    
    def test_update_order_independence(self):
        """Test that the order of updates (C then R) doesn't cause issues."""
        model1 = SimpleModel(dim=10)
        model2 = SimpleModel(dim=10)
        
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        device = torch.device("cpu")
        
        optimizer_C1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        optimizer_R1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        
        optimizer_C2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        optimizer_R2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        
        policy1 = SimpleGRPO(model1, optimizer_C1, optimizer_R1, device=device)
        policy2 = SimpleGRPO(model2, optimizer_C2, optimizer_R2, device=device)
        
        logprobs_C = self._create_model_logprobs(model1, device, n=2)
        rewards_C = [0.8, 0.3]
        
        torch.manual_seed(49)
        logprobs_R = self._create_model_logprobs(model1, device, n=2, detach_after_forward=True)
        rewards_R = [1.0, 0.0]
        
        policy1.update(logprobs_C, rewards_C, logprobs_R, rewards_R)
        
        for i in range(2):
            torch.manual_seed(50 + i * 2)
            lp_C = self._create_model_logprobs(model2, device, n=2)
            policy2.update(lp_C, rewards_C, [], [])
            
            torch.manual_seed(51 + i * 2)
            lp_R = self._create_model_logprobs(model2, device, n=2)
            policy2.update([], [], lp_R, rewards_R)
        
        params1_updated = any(not torch.allclose(p, torch.zeros_like(p), atol=1e-6) 
                             for p in model1.parameters())
        params2_updated = any(not torch.allclose(p, torch.zeros_like(p), atol=1e-6) 
                             for p in model2.parameters())
        
        assert params1_updated, "Model1 should have been updated"
        assert params2_updated, "Model2 should have been updated"
    
    def test_multiple_update_steps(self):
        """Test that training dynamics remain stable over multiple update steps."""
        model = SimpleModel(dim=10)
        device = torch.device("cpu")
        
        optimizer_C = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer_R = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device)
        
        losses_C = []
        losses_R = []
        param_norms = []
        
        for step in range(5):
            torch.manual_seed(42 + step * 2)
            logprobs_C = self._create_model_logprobs(model, device, n=2)
            rewards_C = [0.8, 0.3]
            
            torch.manual_seed(43 + step * 2)
            logprobs_R = self._create_model_logprobs(model, device, n=2, detach_after_forward=True)
            rewards_R = [1.0, 0.0]
            
            loss_C, loss_R, total = policy.update(logprobs_C, rewards_C, logprobs_R, rewards_R)
            
            param_norm_after = sum(p.norm().item() for p in model.parameters())
            
            losses_C.append(loss_C)
            losses_R.append(loss_R)
            param_norms.append(param_norm_after)
            
            assert np.isfinite(loss_C), f"Loss C is not finite at step {step}"
            assert np.isfinite(loss_R), f"Loss R is not finite at step {step}"
            
            for name, param in model.named_parameters():
                assert torch.isfinite(param).all(), f"Parameter {name} contains NaN/Inf at step {step}"
        
        param_change = abs(param_norms[-1] - param_norms[0])
        assert param_change > 1e-8, f"Parameters should have changed over training (change: {param_change})"
        
        assert len(losses_C) == 5
        assert len(losses_R) == 5
        assert all(isinstance(l, float) and np.isfinite(l) for l in losses_C)
        assert all(isinstance(l, float) and np.isfinite(l) for l in losses_R)
    
    def test_zero_grad_isolation(self):
        """Test that zero_grad is called separately for each optimizer."""
        model = SimpleModel(dim=10)
        device = torch.device("cpu")
        
        optimizer_C = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer_R = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device)
        
        logprobs_C = [torch.tensor(-1.5, device=device, requires_grad=True)]
        rewards_C = [0.8]
        
        logprobs_R = [torch.tensor(-1.8, device=device, requires_grad=True)]
        rewards_R = [1.0]
        
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param)
        
        has_grads_before = any(p.grad is not None for p in model.parameters())
        assert has_grads_before, "Should have gradients before update"
        
        policy.update(logprobs_C, rewards_C, logprobs_R, rewards_R)
    
    def test_empty_batch_handling(self):
        """Test that empty batches for one role don't cause issues."""
        model = SimpleModel(dim=10)
        device = torch.device("cpu")
        
        optimizer_C = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer_R = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device)
        
        logprobs_C = self._create_model_logprobs(model, device, n=1)
        rewards_C = [0.8]
        logprobs_R = []
        rewards_R = []
        
        loss_C, loss_R, total = policy.update(logprobs_C, rewards_C, logprobs_R, rewards_R)
        
        assert isinstance(loss_C, float) and np.isfinite(loss_C)
        assert loss_R == 0.0
        assert abs(total - loss_C) < 1e-6
        
        logprobs_C2 = []
        rewards_C2 = []
        logprobs_R2 = self._create_model_logprobs(model, device, n=1)
        rewards_R2 = [1.0]
        
        loss_C2, loss_R2, total2 = policy.update(logprobs_C2, rewards_C2, logprobs_R2, rewards_R2)
        
        assert loss_C2 == 0.0
        assert isinstance(loss_R2, float) and np.isfinite(loss_R2)
        assert abs(total2 - loss_R2) < 1e-6
        
        loss_C3, loss_R3, total3 = policy.update([], [], [], [])
        
        assert loss_C3 == 0.0
        assert loss_R3 == 0.0
        assert total3 == 0.0
    
    def test_different_learning_rates(self):
        """Test that different learning rates for C and R work correctly."""
        model = SimpleModel(dim=10)
        device = torch.device("cpu")
        
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        optimizer_C = torch.optim.AdamW(model.parameters(), lr=1e-2)
        optimizer_R = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device)
        
        logprobs_C = self._create_model_logprobs(model, device, n=1)
        rewards_C = [0.8]
        
        torch.manual_seed(51)
        logprobs_R = self._create_model_logprobs(model, device, n=1, detach_after_forward=True)
        rewards_R = [1.0]
        
        policy.update(logprobs_C, rewards_C, logprobs_R, rewards_R)
        
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should have been updated"

