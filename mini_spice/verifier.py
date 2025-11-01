"""Type-aware answer verification."""

import re
from typing import Literal, Optional
import sympy
from sympy.parsing.sympy_parser import parse_expr


AnswerType = Literal["mcq", "integer", "string", "expression"]


def normalize_mcq(answer: str) -> str:
    """Normalize MCQ answer to A, B, C, or D."""
    answer = answer.strip().upper()
    if len(answer) > 0 and answer[0] in ["A", "B", "C", "D"]:
        return answer[0]
    return answer


def normalize_integer(answer: str) -> Optional[int]:
    """Normalize integer answer."""
    cleaned = re.sub(r'[\s,]+', '', answer.strip())
    cleaned = re.sub(r'^(answer|the\s+answer\s+is|result|solution)[\s:]*', '', cleaned, flags=re.IGNORECASE)
    match = re.search(r'-?\d+', cleaned)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    return None


def normalize_string(answer: str) -> str:
    """Normalize string answer for comparison."""
    return answer.strip().lower()


def normalize_expression(text: str) -> Optional[sympy.Basic]:
    """Normalize and parse expression using sympy."""
    try:
        cleaned = re.sub(r'^(answer|the\s+answer\s+is|result|solution)[\s:]*', '', text.strip(), flags=re.IGNORECASE)
        expr = parse_expr(cleaned, transformations='all')
        expr = sympy.simplify(expr)
        return expr
    except Exception:
        return None


def verify(
    pred: str,
    gold: str,
    ans_type: AnswerType,
    mcq_options: Optional[list[str]] = None
) -> bool:
    """Verify if prediction matches gold answer."""
    if ans_type == "mcq":
        pred_norm = normalize_mcq(pred)
        gold_norm = normalize_mcq(gold)
        
        if pred_norm == gold_norm:
            return True
        
        if mcq_options:
            for i, opt in enumerate(mcq_options):
                opt_letter = chr(65 + i)
                opt_text = opt.strip()
                if gold_norm == opt_letter or gold.lower() in opt_text.lower():
                    if pred_norm == opt_letter or pred.strip().lower() in opt_text.lower():
                        return True
        
        return False
    
    elif ans_type == "integer":
        pred_int = normalize_integer(pred)
        gold_int = normalize_integer(gold)
        
        if pred_int is None or gold_int is None:
            return False
        
        return pred_int == gold_int
    
    elif ans_type == "string":
        pred_norm = normalize_string(pred)
        gold_norm = normalize_string(gold)
        return pred_norm == gold_norm
    
    elif ans_type == "expression":
        pred_expr = normalize_expression(pred)
        gold_expr = normalize_expression(gold)
        
        if pred_expr is None or gold_expr is None:
            return normalize_string(pred) == normalize_string(gold)
        
        try:
            diff = sympy.simplify(pred_expr - gold_expr)
            if diff == 0:
                return True
            return False
        except Exception:
            return normalize_string(pred) == normalize_string(gold)
    
    else:
        return normalize_string(pred) == normalize_string(gold)

