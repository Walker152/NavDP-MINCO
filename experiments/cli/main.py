import argparse
import json

from experiments.analyzers.run_analysis import analyze_run
from experiments.analyzers.paired import compare_runs
from experiments.analyzers.readonly import analyze_suite_readonly
from experiments.analyzers.validator import validate_run
from experiments.closure import run_codex_closure
from experiments.dynamic_pilot import prepare_dynamic_pilot
from experiments.orchestrators.suite_runner import run_suite
from experiments.static.benchmark import (
    generate_static_cases,
    replay_static_case,
    run_static_benchmark,
    validate_static_benchmark,
)
from experiments.static.selection import run_boundary_selection


def build_parser():
    parser = argparse.ArgumentParser(description="NavDP–MINCO experiment toolkit (mock-safe by default)")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-suite"); run.add_argument("--config", required=True); run.add_argument("--backend", default=None, choices=["mock", "isaac"]); run.add_argument("--resume", action="store_true"); run.add_argument("--retry-failed", action="store_true"); run.add_argument("--dry-run", action="store_true"); run.add_argument("--allow-real-simulation", action="store_true"); run.add_argument("--skip-video", action="store_true"); run.add_argument("--analysis-only", action="store_true")
    validate = commands.add_parser("validate"); validate.add_argument("run_dir")
    analyze = commands.add_parser("analyze-run"); analyze.add_argument("run_dir")
    compare = commands.add_parser("compare"); compare.add_argument("--baseline", required=True); compare.add_argument("--method", required=True); compare.add_argument("--output", required=True)
    readonly = commands.add_parser("analyze-suite-readonly")
    readonly.add_argument("--suite", required=True)
    readonly.add_argument("--output", required=True)
    readonly.add_argument("--max-trace-cases", type=int, default=12)
    readonly.add_argument("--resume", action="store_true")
    generate = commands.add_parser("static-generate-cases")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    replay = commands.add_parser("static-replay")
    replay.add_argument("--case", required=True)
    replay.add_argument("--config", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--mode", required=True, choices=["inspect-only", "recompute"])
    benchmark = commands.add_parser("static-benchmark")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--trace-limit", type=int, default=None)
    analyze_static = commands.add_parser("static-analyze")
    analyze_static.add_argument("--result-dir", required=True)
    select_static = commands.add_parser("static-select-cases")
    select_static.add_argument("--config", required=True)
    select_static.add_argument("--output", required=True)
    dynamic = commands.add_parser("dynamic-prepare")
    dynamic.add_argument("--selected-cases", required=True)
    dynamic.add_argument("--calibration", required=True)
    dynamic.add_argument("--legacy-profile", required=True)
    dynamic.add_argument("--safe-profile", required=True)
    dynamic.add_argument("--output", required=True)
    closure = commands.add_parser("codex-closure")
    closure.add_argument("--output", required=True)
    closure.add_argument("--static-only", action="store_true")
    closure.add_argument("--select-cases", action="store_true")
    closure.add_argument("--dynamic-dry-run", action="store_true")
    closure.add_argument("--allow-real-simulation", action="store_true")
    closure.add_argument("--analysis-only-readonly", action="store_true")
    closure.add_argument("--analysis-suite")
    closure.add_argument("--resume", action="store_true")
    closure.add_argument("--retry-failed", action="store_true")
    closure.add_argument("--skip-video", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run-suite": result = run_suite(args.config, args.backend, args.resume, retry_failed=args.retry_failed, dry_run=args.dry_run, analysis_only=args.analysis_only, allow_real_simulation=args.allow_real_simulation, skip_video=args.skip_video); print(json.dumps(result.__dict__))
    elif args.command == "validate": result = validate_run(args.run_dir); print(json.dumps(result, indent=2)); return 0 if result["valid"] else 1
    elif args.command == "analyze-run": print(json.dumps(analyze_run(args.run_dir), indent=2))
    elif args.command == "analyze-suite-readonly":
        result = analyze_suite_readonly(
            args.suite,
            args.output,
            max_trace_cases=args.max_trace_cases,
            resume=args.resume,
        )
        print(
            json.dumps(
                {
                    "input_suite": str(result.input_suite),
                    "output_dir": str(result.output_dir),
                    "protected_receipt_count": len(result.protected_hashes),
                },
                indent=2,
            )
        )
    elif args.command == "static-generate-cases":
        paths = generate_static_cases(args.config, args.output)
        print(json.dumps({"case_count": len(paths), "cases": [str(path) for path in paths]}, indent=2))
    elif args.command == "static-replay":
        result, metrics = replay_static_case(
            args.case, args.config, args.output, args.mode
        )
        print(json.dumps({"status": result.status, "metrics": metrics}, indent=2))
    elif args.command == "static-benchmark":
        result = run_static_benchmark(
            args.config,
            args.output,
            trace_limit=args.trace_limit,
        )
        print(json.dumps({"output_dir": str(result.output_dir), "case_count": result.case_count, "deterministic": result.deterministic}, indent=2))
    elif args.command == "static-analyze":
        errors = validate_static_benchmark(args.result_dir)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    elif args.command == "static-select-cases":
        result = run_boundary_selection(args.config, args.output)
        print(
            json.dumps(
                {
                    "selection_version": result["selection_version"],
                    "best2": result["best2"],
                    "worst2": result["worst2"],
                    "output_dir": str(args.output),
                },
                indent=2,
            )
        )
    elif args.command == "dynamic-prepare":
        result = prepare_dynamic_pilot(
            args.selected_cases,
            args.calibration,
            args.legacy_profile,
            args.safe_profile,
            args.output,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "run_count": result["run_count"],
                    "started_processes": result["started_processes"],
                    "output_dir": str(args.output),
                },
                indent=2,
            )
        )
    elif args.command == "codex-closure":
        result = run_codex_closure(
            repo_root=".",
            output_dir=args.output,
            static_only=args.static_only,
            select_cases=args.select_cases,
            dynamic_dry_run=args.dynamic_dry_run,
            allow_real_simulation=args.allow_real_simulation,
            analysis_only_readonly=args.analysis_only_readonly,
            analysis_suite=args.analysis_suite,
            resume=args.resume,
            retry_failed=args.retry_failed,
            skip_video=args.skip_video,
        )
        print(json.dumps(result, indent=2))
        return 0 if not result["validation_errors"] else 1
    else: print(json.dumps(compare_runs(args.baseline, args.method, args.output), indent=2))
    return 0
