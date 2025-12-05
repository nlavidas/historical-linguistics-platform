"""
HLP CLI - Command Line Interface Package

This package provides command-line tools for the Historical
Linguistics Platform.

Modules:
    cli: Main CLI application
    commands: CLI command implementations

University of Athens - Nikolaos Lavidas
"""

from hlp_cli.cli import main, cli

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "main",
    "cli",
]
