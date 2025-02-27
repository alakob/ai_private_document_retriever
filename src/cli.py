"""
CLI entry point for document processing system.
"""

import click
import logging
from pathlib import Path
from .components.cli.commands import cli

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    cli() 