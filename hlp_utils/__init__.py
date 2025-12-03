"""
HLP Utils - Utility Functions Package

This package provides utility functions for the Historical
Linguistics Platform, including Greek Unicode handling,
tokenization, morphology utilities, and tree operations.

Modules:
    greek_unicode: Greek Unicode normalization
    tokenization: Text tokenization utilities
    morphology: Morphology utilities
    tree_utils: Tree transformation operations

University of Athens - Nikolaos Lavidas
"""

from hlp_utils.greek_unicode import (
    normalize_greek,
    strip_accents,
    strip_breathing,
    normalize_sigma,
    to_beta_code,
    from_beta_code,
    is_greek,
    get_accent_type,
    GreekNormalizer,
)

from hlp_utils.tokenization import (
    tokenize_text,
    tokenize_greek,
    tokenize_latin,
    sentence_split,
    word_tokenize,
    Tokenizer,
    TokenizerConfig,
)

from hlp_utils.morphology_utils import (
    parse_proiel_morphology,
    parse_ud_features,
    format_proiel_morphology,
    format_ud_features,
    MorphologyParser,
    MorphologyFormatter,
)

from hlp_utils.tree_utils import (
    build_dependency_tree,
    linearize_tree,
    find_head,
    get_dependents,
    get_subtree,
    is_projective,
    TreeNode,
    DependencyTree,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "normalize_greek",
    "strip_accents",
    "strip_breathing",
    "normalize_sigma",
    "to_beta_code",
    "from_beta_code",
    "is_greek",
    "get_accent_type",
    "GreekNormalizer",
    "tokenize_text",
    "tokenize_greek",
    "tokenize_latin",
    "sentence_split",
    "word_tokenize",
    "Tokenizer",
    "TokenizerConfig",
    "parse_proiel_morphology",
    "parse_ud_features",
    "format_proiel_morphology",
    "format_ud_features",
    "MorphologyParser",
    "MorphologyFormatter",
    "build_dependency_tree",
    "linearize_tree",
    "find_head",
    "get_dependents",
    "get_subtree",
    "is_projective",
    "TreeNode",
    "DependencyTree",
]
