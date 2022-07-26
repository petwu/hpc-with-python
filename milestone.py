"""
Top-level script to execute a specifc milestone.
Usage: python3 milestone.py --help
"""
import argparse
from src import milestones

args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = args.add_subparsers(metavar="subcmd", required=True, title="subcommands",
                                 description="Run `<subcmd> --help` for details on each subcommand.")
milestones.add_subcommands(subparsers)
args = args.parse_args()
args.handler(args)
