"""Self-play training loop."""

import os
import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .config import Config
from .roles import prompt_challenger, prompt_reasoner, parse_challenger_output, parse_reasoner_output
from .verifier import verify, AnswerType
from .rewards import reasoner_reward, challenger_reward
from .policy import SimpleGRPO
from .storage import log_episode, save_checkpoint_manifest


class Trainer:
    """Self-play trainer for Challenger and Reasoner.
    
    Uses separate optimizers for challenger and reasoner roles to prevent
    policy interference while maintaining a shared model architecture.
    """
    
    def __init__(
        self,
        config: Config,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer_C: torch.optim.Optimizer,
        optimizer_R: torch.optim.Optimizer,
        policy: SimpleGRPO,
        device: torch.device
    ):
        """Initialize trainer with separate optimizers for each role.
        
        Args:
            config: Training configuration.
            model: Shared model used by both challenger and reasoner.
            tokenizer: Tokenizer for text processing.
            optimizer_C: Optimizer for challenger role updates.
            optimizer_R: Optimizer for reasoner role updates.
            policy: Policy update handler (SimpleGRPO).
            device: Device to use for computations.
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer_C = optimizer_C
        self.optimizer_R = optimizer_R
        self.policy = policy
        self.device = device
        
        self.corpus_files = self._load_corpus()
        
        self.metrics = {
            "validity_rate": [],
            "p_pass": [],
            "rC": [],
            "rR": [],
            "loss": [],
            "token_usage": []
        }
    
    def _load_corpus(self) -> List[str]:
        """Load corpus files."""
        corpus_dir = Path(self.config.corpus_dir)
        if not corpus_dir.exists():
            raise ValueError(f"Corpus directory not found: {corpus_dir}")
        
        files = sorted(corpus_dir.glob("*.txt")) + sorted(corpus_dir.glob("*.md"))
        if len(files) == 0:
            raise ValueError(f"No corpus files found in {corpus_dir}")
        
        return [str(f) for f in files]
    
    def _sample_doc(self) -> Tuple[str, str]:
        """Sample a random document from corpus."""
        doc_path = random.choice(self.corpus_files)
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_text = f.read().strip()
        
        doc_id = os.path.basename(doc_path)
        return doc_text, doc_id
    
    def _generate_text(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate text from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_token_ids = inputs.input_ids[0].clone()
        
        # Set model to eval mode for stable generation
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=False
            )
        
        # Restore training mode
        if was_training:
            self.model.train()
        
        # Extract generated token IDs (everything after prompt)
        full_token_ids = outputs[0]  # Shape: [seq_len]
        prompt_len = prompt_token_ids.shape[0]
        generated_token_ids = full_token_ids[prompt_len:].clone()
        
        # Decode for text output
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        return generated_text, prompt_token_ids, generated_token_ids
    
    def _compute_logprob(
        self,
        prompt: str,
        text: str,
        answer_start: Optional[int] = None
    ) -> float:
        """Compute log-probability of generating text."""
        full_text = prompt + text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
        
        # Get log-probs for tokens after prompt
        answer_logprobs = []
        for i in range(prompt_len, inputs.input_ids.shape[1] - 1):
            token_id = inputs.input_ids[0, i + 1].item()
            logits_i = logits[i]
            logprobs_i = torch.nn.functional.log_softmax(logits_i, dim=-1)
            logprob_i = logprobs_i[token_id].item()
            answer_logprobs.append(logprob_i)
        
        if len(answer_logprobs) == 0:
            return 0.0
        
        return float(np.mean(answer_logprobs))
    
    def _compute_logprob_with_grad(
        self,
        prompt_token_ids: torch.Tensor,
        generated_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-probability with gradients enabled."""
        # Concatenate prompt and generated tokens
        full_token_ids = torch.cat([prompt_token_ids, generated_token_ids], dim=0)
        
        # Ensure tensors are on correct device
        full_token_ids = full_token_ids.to(self.device)
        
        # Get prompt length
        prompt_len = prompt_token_ids.shape[0]
        
        # Ensure model is in training mode for gradients
        was_training = self.model.training
        self.model.train()
        
        try:
            # Forward pass WITH gradients enabled
            inputs = {"input_ids": full_token_ids.unsqueeze(0)}
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Compute log-probs for generated tokens only
            answer_logprobs = []
            for i in range(prompt_len, full_token_ids.shape[0] - 1):
                token_id = full_token_ids[i + 1].item()
                logits_i = logits[i]
                logprobs_i = torch.nn.functional.log_softmax(logits_i, dim=-1)
                logprob_i = logprobs_i[token_id]
                answer_logprobs.append(logprob_i)
            
            if len(answer_logprobs) == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Stack and compute mean (preserves gradients)
            answer_logprobs_tensor = torch.stack(answer_logprobs)
            mean_logprob = torch.mean(answer_logprobs_tensor)
            
            return mean_logprob
        finally:
            # Restore original training mode
            if not was_training:
                self.model.eval()
    
    def _challenger_phase(
        self,
        step: int,
        run_id: str,
        log_file: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], float, int]:
        """Execute Challenger phase: generate question from document."""
        # Sample document
        doc_text, doc_id = self._sample_doc()
        
        # Generate prompt
        prompt = prompt_challenger(doc_text)
        
        # Generate Challenger output
        output_text, challenger_prompt_token_ids, challenger_generated_token_ids = self._generate_text(
            prompt,
            temperature=self.config.temp_C,
            max_new_tokens=512
        )
        
        # Count challenger generated tokens
        challenger_token_count = challenger_generated_token_ids.shape[0]
        
        # Parse and validate
        challenger_data, is_valid, error = parse_challenger_output(output_text)
        
        # Debug: Show what was generated
        if step <= 3 or not is_valid:
            print(f"  [DEBUG Step {step}] Generated text (first 200 chars): {output_text[:200]}")
            print(f"  [DEBUG Step {step}] Valid: {is_valid}, Error: {error}")
        
        if not is_valid or challenger_data is None:
            # Invalid output: apply penalty
            log_episode(
                log_file=log_file,
                step=step,
                role="C",
                doc_id=doc_id,
                valid=False,
                rC=self.config.invalid_penalty,
                notes=error,
                run_id=run_id
            )
            return None, [], self.config.invalid_penalty, challenger_token_count
        
        # Compute Challenger logprob with gradients enabled
        challenger_logprob = self._compute_logprob_with_grad(
            challenger_prompt_token_ids,
            challenger_generated_token_ids
        )
        
        # Debug output for valid Challenger outputs
        print(f"  [DEBUG] Valid Challenger output! Logprob: {challenger_logprob.item():.4f}, has_grad: {challenger_logprob.requires_grad and challenger_logprob.grad_fn is not None}")
        
        # Extract question and answer
        question = challenger_data["question"]
        answer_type = challenger_data["type"]
        gold_answer = challenger_data["answer"]
        mcq_options = challenger_data.get("mcq_options") if answer_type == "mcq" else None
        
        # Sample G Reasoner attempts
        reasoner_attempts = []
        correctness_list = []
        logprobs_reasoner = []
        total_token_count = challenger_token_count  # Start with challenger tokens
        
        for _ in range(self.config.G):
            # Reasoner prompt (NO DOCUMENT)
            reasoner_prompt = prompt_reasoner(question, ans_type_hint=answer_type)
            
            # Generate Reasoner output
            reasoner_output, reasoner_prompt_token_ids, reasoner_generated_token_ids = self._generate_text(
                reasoner_prompt,
                temperature=self.config.temp_R,
                max_new_tokens=256
            )
            
            # Count reasoner generated tokens
            reasoner_token_count = reasoner_generated_token_ids.shape[0]
            total_token_count += reasoner_token_count
            
            # Parse Reasoner output
            reasoner_data, reasoner_valid, reasoner_error = parse_reasoner_output(reasoner_output)
            
            if not reasoner_valid or reasoner_data is None:
                # Invalid Reasoner output: penalized and excluded from training
                prediction = ""
                is_correct = False
                # Don't compute logprob for invalid outputs - they're excluded from training
                logprob = None
            else:
                prediction = reasoner_data["final_answer"]
                
                # Verify correctness
                is_correct = verify(
                    prediction,
                    gold_answer,
                    answer_type,
                    mcq_options
                )
                
                # Compute log-probability of answer with gradients enabled (only for valid outputs)
                logprob = self._compute_logprob_with_grad(
                    reasoner_prompt_token_ids,
                    reasoner_generated_token_ids
                )
            
            correctness_list.append(1.0 if is_correct else 0.0)
            
            reasoner_attempts.append({
                "prediction": prediction,
                "correct": is_correct,
                "logprob": logprob,  # None for invalid outputs
                "valid": reasoner_valid,
                "error": reasoner_error if not reasoner_valid else None
            })
            
            # Only add logprob to list if valid (for training)
            if logprob is not None:
                logprobs_reasoner.append(logprob)
        
        # Compute Challenger reward
        rC, p_pass = challenger_reward(correctness_list, sigma=self.config.sigma)
        
        # Log Challenger episode
        log_episode(
            log_file=log_file,
            step=step,
            role="C",
            doc_id=doc_id,
            valid=True,
            type=answer_type,
            p_pass=p_pass,
            rC=rC,
            q=question,
            a_star=gold_answer,
            preds=[r["prediction"] for r in reasoner_attempts],
            run_id=run_id
        )
        
        challenger_data["correctness_list"] = correctness_list
        challenger_data["logprobs_reasoner"] = logprobs_reasoner
        challenger_data["p_pass"] = p_pass
        challenger_data["rC"] = rC
        challenger_data["doc_id"] = doc_id
        challenger_data["logprob"] = challenger_logprob  # Tensor with gradients
        
        return challenger_data, reasoner_attempts, rC, total_token_count
    
    def train_step(
        self,
        step: int,
        run_id: str,
        log_file: str,
        batch_size: int
    ) -> Dict[str, float]:
        """Execute one training step (batch of episodes)."""
        # Collect data from batch
        all_challenger_data = []
        all_reasoner_logprobs = []
        all_reasoner_rewards = []
        all_challenger_logprobs = []
        all_challenger_rewards = []
        
        validity_count = 0
        total_tokens = 0
        
        for _ in range(batch_size):
            # Challenger phase
            challenger_data, reasoner_attempts, rC, token_count = self._challenger_phase(
                step, run_id, log_file
            )
            
            # Accumulate token usage (count tokens even for invalid outputs)
            total_tokens += token_count
            
            if challenger_data is None:
                # Invalid Challenger output: penalized but excluded from training
                # Don't add to training batches - just log it
                continue
            
            validity_count += 1
            all_challenger_data.append(challenger_data)
            all_challenger_rewards.append(rC)
            
            # Get Challenger logprob (already a tensor with gradients)
            challenger_logprob = challenger_data.get("logprob")
            if challenger_logprob is None:
                # Fallback if somehow missing
                challenger_logprob = torch.tensor(0.0, device=self.device, requires_grad=True)
            all_challenger_logprobs.append(challenger_logprob)
            
            # Collect Reasoner data - only include valid outputs in training
            for reasoner_attempt in reasoner_attempts:
                if reasoner_attempt["valid"]:
                    # Valid outputs: include in training
                    all_reasoner_logprobs.append(reasoner_attempt["logprob"])
                    all_reasoner_rewards.append(1.0 if reasoner_attempt["correct"] else 0.0)
                else:
                    # Invalid outputs: penalized but excluded from training
                    # Log the invalid attempt but don't add to training batches
                    pass
        
        # Policy update with debug output
        if len(all_challenger_rewards) > 0 and len(all_reasoner_rewards) > 0:
            # Debug: Check if logprobs have gradients
            if len(all_challenger_logprobs) > 0:
                sample_C_lp = all_challenger_logprobs[0]
                if isinstance(sample_C_lp, torch.Tensor):
                    has_grad_C = sample_C_lp.requires_grad and sample_C_lp.grad_fn is not None
                    lp_val_C = sample_C_lp.item()
                else:
                    has_grad_C = False
                    lp_val_C = float(sample_C_lp)
            else:
                has_grad_C = False
                lp_val_C = 0.0
            
            if len(all_reasoner_logprobs) > 0:
                sample_R_lp = all_reasoner_logprobs[0]
                if isinstance(sample_R_lp, torch.Tensor):
                    has_grad_R = sample_R_lp.requires_grad and sample_R_lp.grad_fn is not None
                    lp_val_R = sample_R_lp.item()
                else:
                    has_grad_R = False
                    lp_val_R = float(sample_R_lp)
            else:
                has_grad_R = False
                lp_val_R = 0.0
            
            loss_C, loss_R, total_loss = self.policy.update(
                logprobs_roleC=all_challenger_logprobs,
                rewards_roleC=all_challenger_rewards,
                logprobs_roleR=all_reasoner_logprobs,
                rewards_roleR=all_reasoner_rewards,
                clip_grad_norm=self.config.grad_clip_norm
            )
            
            # Debug output for valid updates
            if step % self.config.log_interval == 0:
                print(f"  [DEBUG] C_logprobs: {len(all_challenger_logprobs)}, R_logprobs: {len(all_reasoner_logprobs)}")
                print(f"  [DEBUG] Sample C_logprob: {lp_val_C:.4f}, has_grad: {has_grad_C}")
                print(f"  [DEBUG] Sample R_logprob: {lp_val_R:.4f}, has_grad: {has_grad_R}")
                print(f"  [DEBUG] Losses: C={loss_C:.4f}, R={loss_R:.4f}, total={total_loss:.4f}")
        else:
            loss_C = 0.0
            loss_R = 0.0
            total_loss = 0.0
            if step % self.config.log_interval == 0:
                print(f"  [DEBUG] Skipped policy update: C_rewards={len(all_challenger_rewards)}, R_rewards={len(all_reasoner_rewards)}")
        
        # Compute metrics
        validity_rate = validity_count / batch_size
        avg_p_pass = np.mean([d["p_pass"] for d in all_challenger_data]) if all_challenger_data else 0.0
        avg_rC = np.mean(all_challenger_rewards) if all_challenger_rewards else 0.0
        avg_rR = np.mean(all_reasoner_rewards) if all_reasoner_rewards else 0.0
        
        metrics = {
            "validity_rate": validity_rate,
            "p_pass": avg_p_pass,
            "rC": avg_rC,
            "rR": avg_rR,
            "loss": total_loss,
            "loss_C": loss_C,
            "loss_R": loss_R,
            "token_usage": total_tokens
        }
        
        # Update tracked metrics
        self.metrics["validity_rate"].append(validity_rate)
        self.metrics["p_pass"].append(avg_p_pass)
        self.metrics["rC"].append(avg_rC)
        self.metrics["rR"].append(avg_rR)
        self.metrics["loss"].append(total_loss)
        self.metrics["token_usage"].append(total_tokens)
        
        return metrics
    
    def save_checkpoint(
        self,
        step: int,
        checkpoint_dir: str,
        config_dict: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> None:
        """Save training checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if self.config.use_lora:
            # Save only LoRA adapters
            self.model.save_pretrained(checkpoint_dir)
        else:
            # Save full model
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer states
        torch.save(self.optimizer_C.state_dict(), os.path.join(checkpoint_dir, "optimizer_C.pt"))
        torch.save(self.optimizer_R.state_dict(), os.path.join(checkpoint_dir, "optimizer_R.pt"))
        
        # Save manifest
        save_checkpoint_manifest(
            checkpoint_dir=checkpoint_dir,
            step=step,
            config=config_dict,
            metrics=metrics
        )

