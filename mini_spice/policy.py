"""GRPO-style policy update with centered advantages."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


class SimpleGRPO:
    """GRPO implementation with centered advantages per role."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        use_amp: bool = False
    ):
        """Initialize SimpleGRPO."""
        self.model = model
        self.optimizer = optimizer
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
        """Perform policy update with centered advantages."""
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
        
        total_loss = loss_C + loss_R
        
        self.optimizer.zero_grad()
        
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            if clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
            self.optimizer.step()
        
        return (
            loss_C.item(),
            loss_R.item(),
            total_loss.item()
        )
