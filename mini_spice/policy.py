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
        
        # Zero gradients for both optimizers before any backward pass
        # This prevents in-place modifications from breaking the computation graph
        has_loss_C = len(logprobs_C) > 0
        has_loss_R = len(logprobs_R) > 0
        
        if has_loss_C:
            self.optimizer_C.zero_grad()
        if has_loss_R:
            self.optimizer_R.zero_grad()
        
        # Backward both losses before any optimizer step to preserve computation graph
        if has_loss_C and has_loss_R:
            # Both losses exist: backward with retain_graph=True for first
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_C).backward(retain_graph=True)
                self.scaler.scale(loss_R).backward()
            else:
                loss_C.backward(retain_graph=True)
                loss_R.backward()
        elif has_loss_C:
            # Only challenger loss
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_C).backward()
            else:
                loss_C.backward()
        elif has_loss_R:
            # Only reasoner loss
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_R).backward()
            else:
                loss_R.backward()
        
        # Apply gradient clipping if needed (only once for shared parameters)
        if clip_grad_norm is not None and (has_loss_C or has_loss_R):
            if self.use_amp and self.scaler is not None:
                # Unscale for both optimizers before clipping
                if has_loss_C:
                    self.scaler.unscale_(self.optimizer_C)
                if has_loss_R:
                    self.scaler.unscale_(self.optimizer_R)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
        
        # Apply optimizer steps after both backward passes complete
        loss_C_value = 0.0
        if has_loss_C:
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer_C)
            else:
                self.optimizer_C.step()
            loss_C_value = loss_C.item()
        
        loss_R_value = 0.0
        if has_loss_R:
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer_R)
            else:
                self.optimizer_R.step()
            loss_R_value = loss_R.item()
        
        # Update scaler once after all optimizer steps (for AMP)
        if self.use_amp and self.scaler is not None and (has_loss_C or has_loss_R):
            self.scaler.update()
        
        total_loss_value = loss_C_value + loss_R_value
        
        return (
            loss_C_value,
            loss_R_value,
            total_loss_value
        )
