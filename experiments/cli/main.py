import argparse
import json

from experiments.analyzers.run_analysis import analyze_run
from experiments.analyzers.paired import compare_runs
from experiments.analyzers.validator import validate_run
from experiments.orchestrators.suite_runner import run_suite


def build_parser():
    parser = argparse.ArgumentParser(description="NavDP–MINCO experiment toolkit (mock-safe by default)")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-suite"); run.add_argument("--config", required=True); run.add_argument("--backend", default="mock", choices=["mock"]); run.add_argument("--resume", action="store_true")
    validate = commands.add_parser("validate"); validate.add_argument("run_dir")
    analyze = commands.add_parser("analyze-run"); analyze.add_argument("run_dir")
    compare = commands.add_parser("compare"); compare.add_argument("--baseline", required=True); compare.add_argument("--method", required=True); compare.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run-suite": result = run_suite(args.config, args.backend, args.resume); print(json.dumps(result.__dict__))
    elif args.command == "validate": result = validate_run(args.run_dir); print(json.dumps(result, indent=2)); return 0 if result["valid"] else 1
    elif args.command == "analyze-run": print(json.dumps(analyze_run(args.run_dir), indent=2))
    else: print(json.dumps(compare_runs(args.baseline, args.method, args.output), indent=2))
    return 0
