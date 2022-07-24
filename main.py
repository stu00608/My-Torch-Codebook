import yaml
import sys
import argparse
from utils.solver import CircleSolver
from utils.files import file_choices

PATHS = yaml.safe_load(open("paths.yaml"))
for k in PATHS: sys.path.append(PATHS[k])

# TODO: Write description to functions.
# TODO: Make a classifier solver.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=lambda s:file_choices(("yaml"),s), required=True)        
    parser.add_argument('--gpu_id', type=str, default="")
    args = parser.parse_args()

    solver = CircleSolver(args.config)
    solver.run()