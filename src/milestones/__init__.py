import argparse

def add_subcommands(subparsers: argparse._SubParsersAction):
    from .m1 import args as m1_args
    from .m2 import args as m2_args
    from .m3 import args as m3_args
    from .m4 import args as m4_args
    from .m5 import args as m5_args
    from .m6 import args as m6_args
    from .m7 import args as m7_args
    m1_args(subparsers)
    m2_args(subparsers)
    m3_args(subparsers)
    m4_args(subparsers)
    m5_args(subparsers)
    m6_args(subparsers)
    m7_args(subparsers)
