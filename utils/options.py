import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--port", type=int, default=8000)

cmd_opts = parser.parse_args()
