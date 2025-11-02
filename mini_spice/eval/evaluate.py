"""Evaluation logic."""

import json
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_spice.roles import prompt_reasoner, parse_reasoner_output
from mini_spice.verifier import verify, AnswerType


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model on eval dataset."""
    correct = 0
    total = len(eval_data)
    correct_by_type = {}
    total_by_type = {}
    
    for item in eval_data:
        question = item["question"]
        answer = item["answer"]
        ans_type = item["type"]
        mcq_options = item.get("mcq_options")
        
        # Generate answer (greedy decode, temperature=0)
        prompt = prompt_reasoner(question, ans_type_hint=ans_type)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # Greedy decode
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Parse Reasoner output using the same contract as training
        reasoner_data, reasoner_valid, reasoner_error = parse_reasoner_output(generated_text)
        
        if not reasoner_valid or reasoner_data is None:
            # Invalid JSON output: treat as incorrect (consistent with training)
            prediction = ""
            is_correct = False
        else:
            # Extract final_answer from the JSON payload
            prediction = reasoner_data["final_answer"]
            
            # Verify correctness
            is_correct = verify(prediction, answer, ans_type, mcq_options)
        
        if is_correct:
            correct += 1
        
        # Track by type
        if ans_type not in total_by_type:
            total_by_type[ans_type] = 0
            correct_by_type[ans_type] = 0
        
        total_by_type[ans_type] += 1
        if is_correct:
            correct_by_type[ans_type] += 1
    
    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Compute accuracy by type
    accuracy_by_type = {
        ans_type: correct_by_type[ans_type] / total_by_type[ans_type]
        if total_by_type[ans_type] > 0 else 0.0
        for ans_type in total_by_type.keys()
    }
    
    return {
        "accuracy": accuracy,
        "n": total,
        "correct": correct,
        "by_type": accuracy_by_type
    }


def load_eval_data(eval_file: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSON file."""
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "items" in data:
        return data["items"]
    else:
        raise ValueError(f"Invalid eval data format in {eval_file}")


def save_eval_results(
    results: Dict[str, Any],
    run_id: str,
    suite: str,
    output_file: str
) -> None:
    """Save evaluation results to JSON file."""
    output = {
        "run_id": run_id,
        "suite": suite,
        "accuracy": results["accuracy"],
        "n": results["n"],
        "by_type": results["by_type"]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

