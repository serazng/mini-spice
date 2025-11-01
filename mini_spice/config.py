"""Configuration management."""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Config:
    """Training configuration."""
    
    T: int = 100
    B: int = 8
    G: int = 3
    
    temp_C: float = 1.0
    temp_R: float = 1.0
    
    invalid_penalty: float = -0.1
    sigma: float = 0.15
    
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model: str = "microsoft/Phi-3-mini-4k-instruct"
    dtype: Optional[str] = None
    
    max_doc_tokens: int = 3000
    prompt_budget: int = 256
    max_seq_length: int = 4096
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    enable_gradient_checkpointing: bool = True
    low_cpu_mem_usage: bool = True
    use_flash_attention_2: bool = False
    
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: Optional[str] = None
    bnb_4bit_use_double_quant: bool = True
    
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    learning_rate: float = 1e-5
    seed: int = 42
    log_interval: int = 10
    checkpoint_interval: int = 20
    
    corpus_dir: str = "mini_spice/data/corpus"
    runs_dir: str = "runs"
    checkpoints_dir: str = "checkpoints"
    
    eval_datasets_dir: str = "mini_spice/eval/datasets"
    
    def __post_init__(self):
        """Set defaults."""
        # Read paths from environment variables or use defaults
        self.corpus_dir = os.getenv("MINI_SPICE_CORPUS_DIR", self.corpus_dir)
        self.runs_dir = os.getenv("MINI_SPICE_RUNS_DIR", self.runs_dir)
        self.checkpoints_dir = os.getenv("MINI_SPICE_CHECKPOINTS_DIR", self.checkpoints_dir)
        
        if self.dtype is None:
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    self.dtype = "bfloat16"
                else:
                    self.dtype = "float16"
            except ImportError:
                self.dtype = "float32"
        
        if self.bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = "bfloat16"
    
    def auto_detect_quantization(self) -> None:
        """Auto-detect quantization based on VRAM."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = device_props.total_memory / (1024**3)
            
            # For 1-2B models: <12-16GB -> use quantization
            # For 3-4B models: <24GB -> use quantization
            model_size_approx = 1.5  # Assuming 1.5B model
            
            if total_memory_gb < 12 and model_size_approx <= 2.0:
                # Use 4-bit + LoRA for smallest setups
                self.load_in_4bit = True
                self.use_lora = True
                self.max_seq_length = 1024
            elif total_memory_gb < 16 and model_size_approx <= 2.0:
                # Use 8-bit for medium setups
                self.load_in_8bit = True
                self.max_seq_length = 2048
            elif total_memory_gb < 24 and model_size_approx > 2.0:
                # Use 4-bit + LoRA for larger models
                self.load_in_4bit = True
                self.use_lora = True
                self.max_seq_length = 2048
        except Exception:
            # Fallback: no quantization
            pass

