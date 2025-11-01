"""Prepare corpus by chunking documents."""

import argparse
import os
from pathlib import Path
from typing import List
from transformers import AutoTokenizer


def fits(
    doc_text: str,
    tokenizer: AutoTokenizer,
    max_ctx: int,
    prompt_budget: int = 256
) -> bool:
    """Check if document fits within token limit."""
    tokens = tokenizer(doc_text, add_special_tokens=False).input_ids
    return len(tokens) <= (max_ctx - prompt_budget)


def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    max_ctx: int,
    prompt_budget: int = 256,
    strategy: str = "sentence"
) -> List[str]:
    """Chunk text into pieces that fit within token limit."""
    chunks = []
    max_tokens = max_ctx - prompt_budget
    
    if strategy == "sentence":
        # Split by sentences (simple approach: split on periods)
        sentences = text.split(". ")
        
        current_chunk = ""
        for sentence in sentences:
            candidate = current_chunk + sentence + ". "
            
            if fits(candidate, tokenizer, max_ctx, prompt_budget):
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # If single sentence doesn't fit, truncate it
                tokens = tokenizer(candidate, add_special_tokens=False).input_ids
                if len(tokens) > max_tokens:
                    # Truncate to fit
                    truncated = tokenizer.decode(
                        tokens[:max_tokens],
                        skip_special_tokens=True
                    )
                    chunks.append(truncated)
                else:
                    current_chunk = sentence + ". "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    elif strategy == "paragraph":
        # Split by paragraphs
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        for para in paragraphs:
            candidate = current_chunk + para + "\n\n"
            
            if fits(candidate, tokenizer, max_ctx, prompt_budget):
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # If single paragraph doesn't fit, truncate it
                tokens = tokenizer(candidate, add_special_tokens=False).input_ids
                if len(tokens) > max_tokens:
                    truncated = tokenizer.decode(
                        tokens[:max_tokens],
                        skip_special_tokens=True
                    )
                    chunks.append(truncated)
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    return chunks


def prepare_corpus(
    input_dir: str,
    output_dir: str,
    max_tokens: int = 3000,
    prompt_budget: int = 256,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    strategy: str = "sentence"
):
    """Prepare corpus by chunking documents."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process files
    txt_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.md"))
    
    if len(txt_files) == 0:
        raise ValueError(f"No .txt or .md files found in {input_dir}")
    
    chunk_count = 0
    metadata = []
    
    for file_path in txt_files:
        print(f"Processing {file_path.name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Chunk text
        chunks = chunk_text(text, tokenizer, max_tokens, prompt_budget, strategy)
        
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{chunk_count:06d}.txt"
            chunk_path = output_path / chunk_filename
            
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk)
            
            # Store metadata
            token_count = len(tokenizer(chunk, add_special_tokens=False).input_ids)
            metadata.append({
                "file": file_path.name,
                "chunk_id": chunk_count,
                "chunk_index": i,
                "token_count": token_count,
                "filename": chunk_filename
            })
            
            chunk_count += 1
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    import json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nProcessed {chunk_count} chunks from {len(txt_files)} files.")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare corpus by chunking documents by token count."
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="Input directory containing .txt/.md files"
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default="mini_spice/data/corpus",
        help="Output directory for chunked files (default: mini_spice/data/corpus)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum tokens per chunk (default: 3000)"
    )
    parser.add_argument(
        "--prompt-budget",
        type=int,
        default=256,
        help="Tokens reserved for prompt/system messages (default: 256)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for tokenizer (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--strategy",
        choices=["sentence", "paragraph"],
        default="sentence",
        help="Chunking strategy (default: sentence)"
    )
    
    args = parser.parse_args()
    
    prepare_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        prompt_budget=args.prompt_budget,
        model_name=args.model,
        strategy=args.strategy
    )


if __name__ == "__main__":
    main()

