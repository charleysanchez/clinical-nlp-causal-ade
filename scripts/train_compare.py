import argparse

import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    ap.add_argument(
        "--paths", required=False, help="Path to environment pahts YAML"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths) if args.paths else {}

    # merge
    merged = {**cfg, **paths}

    print(merged)


if __name__ == "__main__":
    main()
