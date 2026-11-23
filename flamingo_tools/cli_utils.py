"""@private
"""
import argparse
from typing import Collection, Dict


def check_mode_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    mode: str,
    applies_to: Dict[str, Collection[str]],
) -> None:
    """Reject an option that the selected mode does not use.

    Args:
        parser: The parser, used for the error message and the default values.
        args: The parsed arguments.
        mode: The selected mode, as named in the error message.
        applies_to: The destination of each option, mapped to the modes that use it.
    """
    for dest, modes in applies_to.items():
        if mode not in modes and getattr(args, dest) != parser.get_default(dest):
            parser.error(f"--{dest} does not apply to {mode}.")
