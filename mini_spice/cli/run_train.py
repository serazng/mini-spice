"""Training script."""

import argparse
import os
import sys
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from mini_spice.config import Config
from mini_spice.trainer import Trainer
from mini_spice.policy import SimpleGRPO
from mini_spice.storage import create_run_log


def load_model_and_tokenizer(config: Config):
    """Load model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine dtype
    if config.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif config.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    print(f"Loading model: {config.base_model}")
    print(f"Device: {device}, dtype: {torch_dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }
    
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant
        )
        model_kwargs["quantization_config"] = quantization_config
        print("Using 4-bit quantization")
    elif config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        print("Using 8-bit quantization")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            **model_kwargs
        )
    except Exception as e:
        print(f"Failed to load {config.base_model}: {e}")
        print(f"Falling back to {config.fallback_model}")
        model = AutoModelForCausalLM.from_pretrained(
            config.fallback_model,
            **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(config.fallback_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Move to device if not using device_map
    if not torch.cuda.is_available() or model.device.type == "cpu":
        model = model.to(device)
    
    # Apply LoRA if requested
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        print(f"Applied LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    
    # Enable gradient checkpointing
    if config.enable_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
    
    # Enable flash attention if available
    if config.use_flash_attention_2:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model.config.name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2"
            )
            print("Flash Attention 2 enabled")
        except Exception:
            print("Flash Attention 2 not available, using default")
    
    return model, tokenizer, device


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train mini-SPICE model")
    
    # Training parameters
    parser.add_argument("--T", type=int, help="Total iterations (default: 100)")
    parser.add_argument("--B", type=int, help="Batch size (default: 8)")
    parser.add_argument("--G", type=int, help="Group size (default: 3)")
    
    # Sampling temperatures
    parser.add_argument("--temp-C", type=float, dest="temp_C", help="Challenger temperature (default: 1.0)")
    parser.add_argument("--temp-R", type=float, dest="temp_R", help="Reasoner temperature (default: 1.0)")
    
    # Model parameters
    parser.add_argument("--model", help="Base model name")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], help="Data type")
    
    # Memory/quantization
    parser.add_argument("--8bit", action="store_true", dest="load_in_8bit", help="Use 8-bit quantization")
    parser.add_argument("--4bit", action="store_true", dest="load_in_4bit", help="Use 4-bit quantization")
    parser.add_argument("--lora", action="store_true", dest="use_lora", help="Use LoRA adapters")
    
    # Paths
    parser.add_argument("--corpus-dir", help="Corpus directory")
    parser.add_argument("--runs-dir", help="Runs directory")
    parser.add_argument("--checkpoints-dir", help="Checkpoints directory")
    
    # Other
    parser.add_argument("--invalid-penalty", type=float, dest="invalid_penalty", help="Invalid output penalty")
    parser.add_argument("--learning-rate-C", type=float, dest="learning_rate_C", help="Challenger learning rate (default: 1e-5)")
    parser.add_argument("--learning-rate-R", type=float, dest="learning_rate_R", help="Reasoner learning rate (default: 1e-5)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--log-interval", type=int, dest="log_interval", help="Logging interval")
    parser.add_argument("--checkpoint-interval", type=int, dest="checkpoint_interval", help="Checkpoint interval")
    
    # Auto-detect quantization
    parser.add_argument("--auto-quantize", action="store_true", dest="auto_quantize", help="Auto-detect and apply quantization")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override with CLI arguments
    if args.T is not None:
        config.T = args.T
    if args.B is not None:
        config.B = args.B
    if args.G is not None:
        config.G = args.G
    if args.temp_C is not None:
        config.temp_C = args.temp_C
    if args.temp_R is not None:
        config.temp_R = args.temp_R
    if args.model:
        config.base_model = args.model
    if args.dtype:
        config.dtype = args.dtype
    if args.load_in_8bit:
        config.load_in_8bit = True
    if args.load_in_4bit:
        config.load_in_4bit = True
    if args.use_lora:
        config.use_lora = True
    if args.corpus_dir:
        config.corpus_dir = args.corpus_dir
    if args.runs_dir:
        config.runs_dir = args.runs_dir
    if args.checkpoints_dir:
        config.checkpoints_dir = args.checkpoints_dir
    if args.invalid_penalty is not None:
        config.invalid_penalty = args.invalid_penalty
    if args.learning_rate_C is not None:
        config.learning_rate_C = args.learning_rate_C
    if args.learning_rate_R is not None:
        config.learning_rate_R = args.learning_rate_R
    if args.seed is not None:
        config.seed = args.seed
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    
    # Auto-detect quantization if requested
    if args.auto_quantize:
        config.auto_detect_quantization()
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    print("=" * 60)
    print("Mini-SPICE Training")
    print("=" * 60)
    print(f"Config: T={config.T}, B={config.B}, G={config.G}")
    print(f"Temperatures: C={config.temp_C}, R={config.temp_R}")
    print(f"Learning rates: C={config.learning_rate_C:.2e}, R={config.learning_rate_R:.2e}")
    print(f"Model: {config.base_model}")
    print(f"Seed: {config.seed}")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(config)
    
    # Create separate optimizers for challenger and reasoner
    optimizer_C = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_C
    )
    optimizer_R = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_R
    )
    
    # Create policy
    policy = SimpleGRPO(model, optimizer_C, optimizer_R, device=device, use_amp=False)
    
    # Create trainer
    trainer = Trainer(config, model, tokenizer, optimizer_C, optimizer_R, policy, device)
    
    # Create run log
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = create_run_log(config.runs_dir, run_id)
    print(f"Logging to: {log_file}")
    
    # Training loop
    print("\nStarting training...")
    for step in range(1, config.T + 1):
        metrics = trainer.train_step(step, run_id, log_file, config.B)
        
        # Print metrics
        if step % config.log_interval == 0:
            print(
                f"Step {step}/{config.T}: "
                f"validity={metrics['validity_rate']:.3f}, "
                f"p_pass={metrics['p_pass']:.3f}, "
                f"rC={metrics['rC']:.3f}, "
                f"rR={metrics['rR']:.3f}, "
                f"loss={metrics['loss']:.4f}"
            )
        
        # Save checkpoint
        if step % config.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(config.checkpoints_dir, f"step_{step}")
            
            # Convert config to dict
            config_dict = {
                "T": config.T,
                "B": config.B,
                "G": config.G,
                "temp_C": config.temp_C,
                "temp_R": config.temp_R,
                "base_model": config.base_model,
                "dtype": config.dtype,
                "seed": config.seed,
            }
            
            trainer.save_checkpoint(step, checkpoint_dir, config_dict, metrics)
            print(f"Checkpoint saved: {checkpoint_dir}")
    
    print("\nTraining complete!")
    print(f"Final metrics:")
    print(f"  Validity rate: {np.mean(trainer.metrics['validity_rate']):.3f}")
    print(f"  Average p_pass: {np.mean(trainer.metrics['p_pass']):.3f}")
    print(f"  Average rC: {np.mean(trainer.metrics['rC']):.3f}")
    print(f"  Average rR: {np.mean(trainer.metrics['rR']):.3f}")
    print(f"  Average loss: {np.mean(trainer.metrics['loss']):.4f}")


if __name__ == "__main__":
    main()

