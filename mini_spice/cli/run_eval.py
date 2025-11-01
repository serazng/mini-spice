"""Evaluation script."""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from mini_spice.eval.evaluate import evaluate, load_eval_data, save_eval_results


def load_model_from_checkpoint(checkpoint_dir: str, base_model: str = None):
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if this is a LoRA checkpoint
    is_lora = (checkpoint_path / "adapter_config.json").exists()
    
    if is_lora and base_model:
        print(f"Loading LoRA checkpoint from {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model = PeftModel.from_pretrained(base_model_obj, checkpoint_dir)
        model = model.merge_and_unload()  # Merge adapters for evaluation
    else:
        print(f"Loading full model from {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    return model, tokenizer, device


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate mini-SPICE model")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--eval-file",
        required=True,
        help="Evaluation data JSON file"
    )
    parser.add_argument(
        "--suite",
        default="custom",
        help="Eval suite name (default: custom)"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (default: print to stdout)"
    )
    parser.add_argument(
        "--base-model",
        help="Base model name (required if checkpoint uses LoRA)"
    )
    parser.add_argument(
        "--run-id",
        default="eval_run",
        help="Run identifier (default: eval_run)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mini-SPICE Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval file: {args.eval_file}")
    print("=" * 60)
    
    # Load model
    model, tokenizer, device = load_model_from_checkpoint(
        args.checkpoint,
        base_model=args.base_model
    )
    
    # Load eval data
    eval_data = load_eval_data(args.eval_file)
    print(f"Loaded {len(eval_data)} eval items")
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate(model, tokenizer, eval_data, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['n']})")
    print("\nBy type:")
    for ans_type, acc in results['by_type'].items():
        print(f"  {ans_type}: {acc:.4f}")
    print("=" * 60)
    
    # Save results
    if args.output:
        save_eval_results(results, args.run_id, args.suite, args.output)
        print(f"\nResults saved to: {args.output}")
    else:
        # Also save to default location
        import os
        from datetime import datetime
        output_dir = "eval_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{args.suite}_{timestamp}.json")
        save_eval_results(results, args.run_id, args.suite, output_file)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

