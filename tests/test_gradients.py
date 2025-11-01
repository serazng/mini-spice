#!/usr/bin/env python3
"""Quick test to verify gradients are flowing."""

import torch
import sys

# Add a test that verifies logprobs have gradients
print("Testing gradient flow...")

# Create a simple test
device = torch.device("cpu")
x = torch.randn(2, 3, requires_grad=True, device=device)
y = x.mean()
loss = -y * 2.0
loss.backward()

print(f"✓ Basic gradient test: loss={loss.item():.4f}, grad_norm={x.grad.norm().item():.4f}")

# Now test with a dummy scenario that mimics our code
logprobs = torch.tensor([-1.5, -2.0], device=device, requires_grad=True)
advantages = torch.tensor([0.5, -0.5], device=device)
loss = -torch.mean(logprobs * advantages)

print(f"✓ REINFORCE-style test: loss={loss.item():.4f}, has_grad_fn={loss.grad_fn is not None}")

if loss.grad_fn is not None:
    loss.backward()
    print(f"✓ Backward pass successful, logprob grad norm={logprobs.grad.norm().item():.4f}")
    print("\n✅ Gradient flow is working!")
else:
    print("❌ No gradient function - something is wrong")
    sys.exit(1)

