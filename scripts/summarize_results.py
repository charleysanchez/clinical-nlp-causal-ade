import argparse

def arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="reports", help="Root directory to scan for metrics")
    ap.add_argument("--pattern", default="**/metrics/*.json", help="Glob pattern under runs_dir")
    ap.add_argument("--out_csv", default="reports/summary.csv")
    ap.add_argument("--out_md", default="reports/summary.md")