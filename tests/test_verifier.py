"""Tests for verifier module."""

import pytest
from mini_spice.verifier import (
    verify,
    normalize_mcq,
    normalize_integer,
    normalize_string,
    normalize_expression
)


class TestNormalizeMCQ:
    """Tests for MCQ normalization."""
    
    def test_normalize_letter(self):
        assert normalize_mcq("A") == "A"
        assert normalize_mcq("B") == "B"
        assert normalize_mcq("a") == "A"
        assert normalize_mcq("b") == "B"
    
    def test_normalize_with_text(self):
        assert normalize_mcq("A) Option text") == "A"
        assert normalize_mcq("Answer: B") == "B"


class TestNormalizeInteger:
    """Tests for integer normalization."""
    
    def test_simple_integer(self):
        assert normalize_integer("42") == 42
        assert normalize_integer(" 123 ") == 123
        assert normalize_integer("-5") == -5
    
    def test_with_text(self):
        assert normalize_integer("Answer: 42") == 42
        assert normalize_integer("The answer is 123") == 123
    
    def test_invalid(self):
        assert normalize_integer("not a number") is None
        assert normalize_integer("abc") is None


class TestNormalizeString:
    """Tests for string normalization."""
    
    def test_normalize(self):
        assert normalize_string("Hello") == "hello"
        assert normalize_string("  WORLD  ") == "world"
        assert normalize_string("Test String") == "test string"


class TestNormalizeExpression:
    """Tests for expression normalization."""
    
    def test_simple_expression(self):
        result = normalize_expression("2 + 2")
        assert result is not None
        assert float(result) == 4.0
    
    def test_complex_expression(self):
        result = normalize_expression("x^2 + 2*x + 1")
        assert result is not None
    
    def test_invalid(self):
        assert normalize_expression("not an expression") is None


class TestVerify:
    """Tests for verification."""
    
    def test_verify_mcq(self):
        assert verify("A", "A", "mcq") is True
        assert verify("B", "A", "mcq") is False
        assert verify("a", "A", "mcq") is True  # Case insensitive
        assert verify("A", "B", "mcq") is False
    
    def test_verify_mcq_with_options(self):
        options = ["A) First", "B) Second", "C) Third", "D) Fourth"]
        assert verify("A", "A", "mcq", options) is True
        assert verify("First", "A", "mcq", options) is True
        assert verify("B", "A", "mcq", options) is False
    
    def test_verify_integer(self):
        assert verify("42", "42", "integer") is True
        assert verify("42", "43", "integer") is False
        assert verify(" 42 ", "42", "integer") is True
        assert verify("Answer: 42", "42", "integer") is True
    
    def test_verify_string(self):
        assert verify("hello", "hello", "string") is True
        assert verify("HELLO", "hello", "string") is True  # Case insensitive
        assert verify("world", "hello", "string") is False
        assert verify("  hello  ", "hello", "string") is True
    
    def test_verify_expression(self):
        assert verify("2 + 2", "4", "expression") is True
        assert verify("x + 1", "1 + x", "expression") is True  # Should be equivalent
        assert verify("2 * 3", "5", "expression") is False
        # Note: sympy equivalence checking may vary, so these tests are basic

