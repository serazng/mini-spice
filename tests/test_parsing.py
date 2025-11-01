"""Tests for role parsing and JSON extraction."""

import pytest
from mini_spice.roles import (
    extract_json,
    validate_challenger_output,
    validate_reasoner_output,
    parse_challenger_output,
    parse_reasoner_output,
    prompt_challenger,
    prompt_reasoner
)


class TestExtractJSON:
    """Tests for JSON extraction from text."""
    
    def test_clean_json(self):
        text = '{"question": "What is 2+2?", "type": "integer", "answer": "4"}'
        result = extract_json(text)
        assert result is not None
        assert result["question"] == "What is 2+2?"
        assert result["type"] == "integer"
        assert result["answer"] == "4"
    
    def test_json_with_whitespace(self):
        text = '  { "question" : "Test?" , "type" : "string" , "answer" : "test" }  '
        result = extract_json(text)
        assert result is not None
        assert result["question"] == "Test?"
    
    def test_json_with_prose(self):
        text = 'Here is the answer: {"question": "Test?", "type": "integer", "answer": "42"}. That\'s it!'
        result = extract_json(text)
        assert result is not None
        assert result["answer"] == "42"
    
    def test_nested_json(self):
        text = '{"question": "Test?", "type": "mcq", "answer": "A", "mcq_options": ["A) First", "B) Second"]}'
        result = extract_json(text)
        assert result is not None
        assert result["type"] == "mcq"
        assert len(result["mcq_options"]) == 2
    
    def test_invalid_json(self):
        text = "This is not JSON at all"
        result = extract_json(text)
        assert result is None


class TestValidateChallengerOutput:
    """Tests for Challenger output validation."""
    
    def test_valid_output(self):
        data = {
            "question": "What is 2+2?",
            "type": "integer",
            "answer": "4"
        }
        is_valid, error = validate_challenger_output(data)
        assert is_valid is True
        assert error is None
    
    def test_valid_mcq(self):
        data = {
            "question": "What is the capital?",
            "type": "mcq",
            "answer": "A",
            "mcq_options": ["A) Paris", "B) London", "C) Berlin", "D) Madrid"]
        }
        is_valid, error = validate_challenger_output(data)
        assert is_valid is True
        assert error is None
    
    def test_missing_field(self):
        data = {
            "question": "Test?",
            "type": "integer"
            # Missing "answer"
        }
        is_valid, error = validate_challenger_output(data)
        assert is_valid is False
        assert "answer" in error
    
    def test_invalid_type(self):
        data = {
            "question": "Test?",
            "type": "invalid_type",
            "answer": "test"
        }
        is_valid, error = validate_challenger_output(data)
        assert is_valid is False
        assert "type" in error
    
    def test_mcq_missing_options(self):
        data = {
            "question": "Test?",
            "type": "mcq",
            "answer": "A"
            # Missing "mcq_options"
        }
        is_valid, error = validate_challenger_output(data)
        assert is_valid is False
        assert "mcq_options" in error
    
    def test_mcq_wrong_option_count(self):
        data = {
            "question": "Test?",
            "type": "mcq",
            "answer": "A",
            "mcq_options": ["A) First", "B) Second"]  # Only 2, need 4
        }
        is_valid, error = validate_challenger_output(data)
        assert is_valid is False
        assert "4" in error
    
    def test_empty_question(self):
        data = {
            "question": "",
            "type": "integer",
            "answer": "4"
        }
        is_valid, error = validate_challenger_output(data)
        assert is_valid is False
        assert "question" in error


class TestValidateReasonerOutput:
    """Tests for Reasoner output validation."""
    
    def test_valid_output(self):
        data = {"final_answer": "42"}
        is_valid, error = validate_reasoner_output(data)
        assert is_valid is True
        assert error is None
    
    def test_missing_field(self):
        data = {"answer": "42"}  # Should be "final_answer"
        is_valid, error = validate_reasoner_output(data)
        assert is_valid is False
        assert "final_answer" in error
    
    def test_multiline_answer(self):
        data = {"final_answer": "Line 1\nLine 2"}
        is_valid, error = validate_reasoner_output(data)
        assert is_valid is False
        assert "single-line" in error.lower() or "line" in error.lower()
    
    def test_non_string_answer(self):
        data = {"final_answer": 42}  # Should be string
        is_valid, error = validate_reasoner_output(data)
        assert is_valid is False


class TestParseChallengerOutput:
    """Tests for Challenger output parsing."""
    
    def test_parse_valid(self):
        text = '{"question": "What is 2+2?", "type": "integer", "answer": "4"}'
        data, is_valid, error = parse_challenger_output(text)
        assert is_valid is True
        assert data is not None
        assert data["question"] == "What is 2+2?"
    
    def test_parse_invalid_json(self):
        text = "Not JSON"
        data, is_valid, error = parse_challenger_output(text)
        assert is_valid is False
        assert data is None
        assert error is not None
    
    def test_parse_invalid_schema(self):
        text = '{"question": "Test?", "type": "invalid", "answer": "test"}'
        data, is_valid, error = parse_challenger_output(text)
        assert is_valid is False
        assert error is not None


class TestParseReasonerOutput:
    """Tests for Reasoner output parsing."""
    
    def test_parse_valid(self):
        text = '{"final_answer": "42"}'
        data, is_valid, error = parse_reasoner_output(text)
        assert is_valid is True
        assert data is not None
        assert data["final_answer"] == "42"
    
    def test_parse_with_prose(self):
        text = 'Here is the answer: {"final_answer": "42"}. Done!'
        data, is_valid, error = parse_reasoner_output(text)
        assert is_valid is True
        assert data is not None
        assert data["final_answer"] == "42"
    
    def test_parse_invalid(self):
        text = "Not JSON"
        data, is_valid, error = parse_reasoner_output(text)
        assert is_valid is False
        assert data is None


class TestPrompts:
    """Tests for prompt generation."""
    
    def test_challenger_prompt(self):
        doc = "This is a test document."
        prompt = prompt_challenger(doc)
        
        assert doc in prompt
        assert "question" in prompt.lower() or "document" in prompt.lower()
    
    def test_reasoner_prompt(self):
        question = "What is 2+2?"
        prompt = prompt_reasoner(question)
        
        assert question in prompt
        assert "question" in prompt.lower()
    
    def test_reasoner_prompt_with_type_hint(self):
        question = "What is 2+2?"
        prompt = prompt_reasoner(question, ans_type_hint="integer")
        
        assert question in prompt
        assert "integer" in prompt.lower()

