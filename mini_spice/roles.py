"""Challenger and Reasoner prompts and JSON parsing."""

import json
from typing import Dict, Optional, Literal, Any
from enum import Enum


class AnswerType(str, Enum):
    """Answer type enum."""
    MCQ = "mcq"
    INTEGER = "integer"
    STRING = "string"
    EXPRESSION = "expression"


CHALLENGER_SYSTEM_MESSAGE = """You read a short document and craft a single rigorous question with a verifiable, typed answer.

CRITICAL: Output STRICT JSON with EXACTLY these rules:

1. REQUIRED fields: question, type, answer
   - type MUST be exactly one of: "mcq", "integer", "string", "expression"
   - type MUST NOT be empty

2. CONDITIONAL field: mcq_options
   - IF type == "mcq": YOU MUST include mcq_options (exactly 4 options labeled A), B), C), D))
   - IF type != "mcq" (i.e., "integer", "string", or "expression"): YOU MUST NOT include mcq_options
   - VIOLATION: Including mcq_options when type is not "mcq" will cause validation failure

3. VALID EXAMPLES:
   For integer/string/expression: {"question": "...", "type": "integer", "answer": "..."}
   For mcq: {"question": "...", "type": "mcq", "answer": "A", "mcq_options": ["A) ...", "B) ...", "C) ...", "D) ..."]}

4. The question must be answerable WITHOUT seeing the document if a capable reasoner has sufficient general knowledge. Keep the question concise. Do not include explanations."""

REASONER_SYSTEM_MESSAGE = """You receive a question WITHOUT any source document. Provide only the final answer in STRICT JSON: {"final_answer": "..."}. Match the requested type when obvious (e.g., choose A/B/C/D for mcq; return a simplified integer or algebraic expression for expression). No extra text."""


def prompt_challenger(doc: str) -> str:
    """Generate Challenger prompt from document."""
    return f"""{CHALLENGER_SYSTEM_MESSAGE}

Document:
{doc}

Output JSON:"""


def prompt_reasoner(q: str, ans_type_hint: Optional[str] = None) -> str:
    """Generate Reasoner prompt from question."""
    hint_text = f"\nAnswer type: {ans_type_hint}" if ans_type_hint else ""
    return f"""{REASONER_SYSTEM_MESSAGE}

Question:{hint_text}
{q}

Output JSON:"""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text using a tolerant streaming decoder.
    
    Scans text for the first valid JSON object by tracking brace depth,
    handling nested objects, arrays, and string escapes properly.
    """
    i = 0
    text_len = len(text)
    
    # Find first opening brace
    while i < text_len:
        if text[i] == '{':
            start = i
            depth = 0
            in_string = False
            escape_next = False
            
            # Scan forward tracking brace depth
            while i < text_len:
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"':
                    in_string = not in_string
                    i += 1
                    continue
                
                if in_string:
                    i += 1
                    continue
                
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        # Found complete object
                        candidate = text[start:i+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            # Invalid JSON, continue searching
                            pass
                        i += 1
                        break
                
                i += 1
        
        i += 1
    
    # Fallback: try parsing entire text
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    return None


def validate_challenger_output(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate Challenger output format with strict schema: {question, type, answer, mcq_options?}.
    
    Schema details:
    - Required fields: question, type, answer
    - type must be one of: "mcq", "integer", "string", "expression"
    - mcq_options is conditionally required:
      * MUST be included when type == "mcq" (list of exactly 4 options labeled A-D)
      * MUST NOT be included when type != "mcq" (integer, string, or expression)
    - Extra fields are not allowed; strict schema enforcement for alignment with SPICE paper.
    
    Returns:
        tuple: (is_valid, error_message) where is_valid is True if valid, False otherwise
    """
    required_fields = {"question", "type", "answer"}
    missing = required_fields - set(data.keys())
    if missing:
        return False, f"Missing required fields: {missing}"
    
    valid_types = {"mcq", "integer", "string", "expression"}
    if data["type"] not in valid_types:
        return False, f"Invalid type: {data['type']}. Must be one of {valid_types}"
    
    # Check mcq_options handling first, then check for extra fields
    is_mcq = data["type"] == "mcq"
    
    # Strict schema: only allow required fields + optional mcq_options (only for mcq)
    allowed_fields = {"question", "type", "answer"}
    if is_mcq:
        allowed_fields.add("mcq_options")
    
    extra_fields = set(data.keys()) - allowed_fields
    if extra_fields:
        return False, f"Extra fields not allowed: {extra_fields}. Schema must be {{question, type, answer, mcq_options?}}"
    
    if not isinstance(data["question"], str) or not data["question"].strip():
        return False, "question must be a non-empty string"
    
    if not isinstance(data["answer"], (str, int, float)):
        return False, "Answer must be a string or number"
    
    if isinstance(data["answer"], (int, float)):
        data["answer"] = str(data["answer"])
    elif not isinstance(data["answer"], str) or not data["answer"].strip():
        return False, "Answer must be a non-empty string"
    
    if data["type"] == "mcq":
        if "mcq_options" not in data:
            return False, "mcq type requires mcq_options field"
        
        options = data["mcq_options"]
        if not isinstance(options, list) or len(options) != 4:
            return False, "mcq_options must be a list with exactly 4 options"
        
        prefixes = ["A)", "B)", "C)", "D)"]
        for i, opt in enumerate(options):
            if not isinstance(opt, str) or not opt.strip():
                return False, f"Option {i+1} must be a non-empty string"
            if not opt.strip().startswith(prefixes[i]):
                return False, f"Option {i+1} must start with {prefixes[i]}"
        
        answer_norm = data["answer"].strip().upper()
        if answer_norm not in ["A", "B", "C", "D"]:
            answer_text = data["answer"].strip()
            matches = [i for i, opt in enumerate(options) if opt.strip().endswith(answer_text) or answer_text in opt]
            if not matches:
                return False, "Answer must match one of the MCQ options (A, B, C, or D)"
    
    return True, None


def validate_reasoner_output(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate Reasoner output format with strict schema: {final_answer}."""
    if "final_answer" not in data:
        return False, "Missing required field: final_answer"
    
    # Strict schema: only allow final_answer field
    allowed_fields = {"final_answer"}
    extra_fields = set(data.keys()) - allowed_fields
    if extra_fields:
        return False, f"Extra fields not allowed: {extra_fields}. Schema must be {{final_answer}}"
    
    if not isinstance(data["final_answer"], str):
        return False, "final_answer must be a string"
    
    if "\n" in data["final_answer"]:
        return False, "final_answer must be a single-line answer"
    
    return True, None


def parse_challenger_output(text: str) -> tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """Parse and validate Challenger output."""
    data = extract_json(text)
    if data is None:
        return None, False, "Failed to extract JSON from output"
    
    is_valid, error = validate_challenger_output(data)
    return data, is_valid, error


def parse_reasoner_output(text: str) -> tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """Parse and validate Reasoner output."""
    data = extract_json(text)
    if data is None:
        return None, False, "Failed to extract JSON from output"
    
    is_valid, error = validate_reasoner_output(data)
    return data, is_valid, error

