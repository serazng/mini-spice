"""Command-line interface for mini-SPICE."""

from .prepare_corpus import main as prepare_corpus_main
from .run_train import main as run_train_main
from .run_eval import main as run_eval_main

__all__ = ["prepare_corpus_main", "run_train_main", "run_eval_main"]

