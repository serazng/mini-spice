"""GRPO-style policy update with centered advantages."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


class SimpleGRPO:
    """GRPO implementation with centered advantages per role.
    
    Uses separate optimizers for challenger and reasoner roles to prevent
    policy interference. Both optimizers update the same model parameters,
    but maintain independent optimizer states (momentum, Adam moments, etc.).
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_C: torch.optim.Optimizer,
        optimizer_R: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        use_amp: bool = False
    ):
        """Initialize SimpleGRPO with separate optimizers for each role.
        
        Args:
            model: The shared model used by both challenger and reasoner.
            optimizer_C: Optimizer for challenger role updates.
            optimizer_R: Optimizer for reasoner role updates.
            device: Device to use for computations.
            use_amp: Whether to use automatic mixed precision.
        """
        self.model = model
        self.optimizer_C = optimizer_C
        self.optimizer_R = optimizer_R
        self.use_amp = use_amp
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def update(
        self,
        logprobs_roleC: List,
        rewards_roleC: List[float],
        logprobs_roleR: List,
        rewards_roleR: List[float],
        clip_grad_norm: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """Perform policy update with centered advantages.
        
        Each role's loss is backpropagated and optimized independently using
        separate optimizers to prevent policy interference.
        """
        if len(logprobs_roleC) == 0 and len(logprobs_roleR) == 0:
            return 0.0, 0.0, 0.0
        
        if len(logprobs_roleC) > 0 and isinstance(logprobs_roleC[0], torch.Tensor):
            logprobs_C = torch.stack(logprobs_roleC)
        else:
            logprobs_C = torch.tensor(logprobs_roleC, dtype=torch.float32, device=self.device)
        
        if len(logprobs_roleR) > 0 and isinstance(logprobs_roleR[0], torch.Tensor):
            logprobs_R = torch.stack(logprobs_roleR)
        else:
            logprobs_R = torch.tensor(logprobs_roleR, dtype=torch.float32, device=self.device)
        
        rewards_C = torch.tensor(rewards_roleC, dtype=torch.float32, device=self.device)
        rewards_R = torch.tensor(rewards_roleR, dtype=torch.float32, device=self.device)
        
        mean_reward_C = torch.mean(rewards_C) if len(rewards_C) > 0 else torch.tensor(0.0, device=self.device)
        mean_reward_R = torch.mean(rewards_R) if len(rewards_R) > 0 else torch.tensor(0.0, device=self.device)
        
        advantages_C = rewards_C - mean_reward_C
        advantages_R = rewards_R - mean_reward_R
        
        loss_C = -torch.mean(logprobs_C * advantages_C) if len(logprobs_C) > 0 else torch.tensor(0.0, device=self.device)
        loss_R = -torch.mean(logprobs_R * advantages_R) if len(logprobs_R) > 0 else torch.tensor(0.0, device=self.device)
        
        # Update challenger independently
        loss_C_value = 0.0
        if len(logprobs_C) > 0:
            self.optimizer_C.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_C).backward()
                if clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer_C)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                self.scaler.step(self.optimizer_C)
                self.scaler.update()
            else:
                loss_C.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                self.optimizer_C.step()
            
            loss_C_value = loss_C.item()
        
        # Update reasoner independently (gradients cleared from previous update)
        loss_R_value = 0.0
        if len(logprobs_R) > 0:
            self.optimizer_R.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_R).backward()
                if clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer_R)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                self.scaler.step(self.optimizer_R)
                self.scaler.update()
            else:
                loss_R.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                self.optimizer_R.step()
            
            loss_R_value = loss_R.item()
        
        total_loss_value = loss_C_value + loss_R_value
        
        return (
            loss_C_value,
            loss_R_value,
            total_loss_value
        )
