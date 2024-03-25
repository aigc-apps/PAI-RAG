import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--log-level", type=str.upper, default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help="Minimum log level to print.")

cmd_opts = parser.parse_args()
