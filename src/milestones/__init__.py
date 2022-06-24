import argparse

def add_subcommands(subparsers: argparse._SubParsersAction):
    from .m1 import args as m1_args
    from .m2 import args as m2_args
    from .m3 import args as m3_args
    from .m4 import args as m4_args
    m1_args(subparsers)
    m2_args(subparsers)
    m3_args(subparsers)
    m4_args(subparsers)
