#!/usr/bin/env python3
"""RAGAS evaluation CLI for the FRAG framework.

Evaluate your RAG pipeline with RAGAS metrics, compare configurations,
and generate synthetic test datasets — all from the command line.

See the accompanying README.md for a detailed usage guide.

Usage examples::

    # Evaluate with the current config
    python cl-tools/evaluate.py run --testset eval_data/my_testset.json

    # Evaluate with config overrides (compare HyDE on vs off)
    python cl-tools/evaluate.py run --testset eval_data/my_testset.json \\
        --config baseline \\
        --config hyde_on --set use_hyde=true

    # Generate a synthetic test set from your indexed documents
    python cl-tools/evaluate.py generate-testset \\
        --num-questions 20 --output eval_data/auto_testset.json

    # List past evaluation results
    python cl-tools/evaluate.py results

    # Show detailed per-question scores for a result file
    python cl-tools/evaluate.py results --file eval_results/result.json --detail
"""

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Resolve repo root so the script can be run from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_REPO_ROOT, ".env"))


# -----------------------------------------------------------------------
# Subcommand: run
# -----------------------------------------------------------------------

def cmd_run(args) -> None:
    """Execute a RAGAS evaluation run."""
    from evaluation.testset import load_testset
    from evaluation.runner import run_evaluation
    from config.eval_config import eval_results_dir, default_metrics

    # Load test dataset.
    samples = load_testset(args.testset)
    print(f"Loaded {len(samples)} test samples from: {args.testset}")

    # Parse configs from the CLI.
    configs = _parse_configs(args)

    # Guard against phantom chunking evaluation
    chunking_keys = {"chunk_size", "overlap_size", "chunking_method", "token_chunk_size", "semantic_breakpoint_percentile"}
    for conf_name, overrides in configs.items():
        if any(k in overrides for k in chunking_keys):
            print(f"\nERROR in config '{conf_name}': You cannot override chunking settings via CLI.")
            print("Evaluation uses the existing ChromaDB index. To evaluate different chunk sizes, you must update config/embedding_config.py and run vector_db_setup.py first.")
            sys.exit(1)

    # Metrics.
    metric_names = args.metrics.split(",") if args.metrics else default_metrics

    # Run.
    report = run_evaluation(
        testset_path=args.testset,
        samples=samples,
        configs=configs,
        metric_names=metric_names,
        env_path=os.path.join(_REPO_ROOT, ".env"),
        config_filter=None,
        judge_model_override=args.judge_model,
        judge_base_url_override=args.judge_base_url,
    )

    # Display results.
    report.print_table()

    # Save.
    output_dir = args.output_dir or eval_results_dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"{ts}_results.json")
    report.to_json(output_path)

    if args.detail:
        report.print_detail()


def _parse_configs(args) -> dict:
    """Parse --config and --set flags into a configs dict.

    The pattern is:
        --config baseline
        --config hyde_on --set use_hyde=true
        --config hybrid  --set use_hybrid_search=true --set n_results=10

    If no --config is given, a single "current" config (no overrides) is used.

    The parser groups --set flags with the most recently declared --config.
    """
    if not args.config_and_set:
        return {"current": {}}

    configs = {}
    current_name = "current"
    current_overrides = {}

    for token in args.config_and_set:
        if token.startswith("config:"):
            # Save previous config if any.
            if current_name is not None:
                configs[current_name] = current_overrides
            current_name = token[len("config:"):]
            current_overrides = {}
        elif "=" in token:
            key, _, value = token.partition("=")
            current_overrides[key] = _coerce_value(value)
        else:
            # Treat as a config name without the prefix.
            if current_name is not None:
                configs[current_name] = current_overrides
            current_name = token
            current_overrides = {}

    if current_name is not None:
        configs[current_name] = current_overrides

    return configs if configs else {"current": {}}


def _coerce_value(value: str):
    """Convert a string CLI value to the appropriate Python type."""
    lower = value.lower()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# -----------------------------------------------------------------------
# Subcommand: generate-testset
# -----------------------------------------------------------------------

def cmd_generate_testset(args) -> None:
    """Generate a synthetic test set from indexed documents."""
    from evaluation.testset import generate_synthetic_testset, save_testset
    from config.embedding_config import collection_name, db_directory
    from config.llm_config import (
        use_openai, openai_model, openai_base_url, llm_model,
    )

    api_key = os.environ.get("OPENAI_API_KEY")

    samples = generate_synthetic_testset(
        num_questions=args.num_questions,
        collection_name=args.collection or collection_name,
        db_directory=db_directory,
        use_openai=use_openai,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        ollama_model=llm_model,
        api_key=api_key,
    )

    if not samples:
        print("No QA pairs were generated.  Check LLM connectivity.")
        sys.exit(1)

    save_testset(samples, args.output)


# -----------------------------------------------------------------------
# Subcommand: results
# -----------------------------------------------------------------------

def cmd_results(args) -> None:
    """List or display past evaluation results."""
    from evaluation.report import EvalReport, print_result_list
    from config.eval_config import eval_results_dir

    results_dir = args.output_dir or eval_results_dir

    if args.file:
        report = EvalReport.from_json(args.file)
        report.print_table()
        if args.detail:
            report.print_detail(args.config_name)
    else:
        print_result_list(results_dir)


# -----------------------------------------------------------------------
# CLI parser
# -----------------------------------------------------------------------

class _ConfigAction(argparse.Action):
    """Custom action that collects --config and --set into one flat list."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest) or []
        if option_string == "--config":
            current.append(f"config:{values}")
        elif option_string == "--set":
            if isinstance(values, list):
                current.extend(values)
            else:
                current.append(values)
        setattr(namespace, self.dest, current)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "RAGAS evaluation for the FRAG framework.  "
            "Measure retrieval and generation quality, compare configs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Evaluate with the current config
  python cl-tools/evaluate.py run --testset eval_data/test.json

  # Compare two configs
  python cl-tools/evaluate.py run --testset eval_data/test.json \\
      --config baseline \\
      --config with_hyde --set use_hyde=true

  # Compare multiple settings
  python cl-tools/evaluate.py run --testset eval_data/test.json \\
      --config small_chunks --set chunk_size=20 --set overlap_size=3 \\
      --config large_chunks --set chunk_size=60 --set overlap_size=10 \\
      --config hybrid --set use_hybrid_search=true --set n_results=10

  # Generate a synthetic test set
  python cl-tools/evaluate.py generate-testset \\
      --num-questions 20 --output eval_data/auto_testset.json

  # List past results
  python cl-tools/evaluate.py results

  # View a specific result in detail
  python cl-tools/evaluate.py results --file eval_results/result.json --detail
        """,
    )
    sub = parser.add_subparsers(dest="command", metavar="subcommand")
    sub.required = True

    # -- run ---------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run a RAGAS evaluation against a test dataset.",
    )
    p_run.add_argument(
        "--testset", "-t",
        required=True,
        help="Path to the test-set file (.json or .csv).",
    )
    p_run.add_argument(
        "--config",
        dest="config_and_set",
        action=_ConfigAction,
        help=(
            "Name a configuration profile.  Follow with --set key=value "
            "flags to override settings.  Repeat for multiple configs."
        ),
    )
    p_run.add_argument(
        "--set",
        dest="config_and_set",
        action=_ConfigAction,
        nargs="+",
        metavar="KEY=VALUE",
        help=(
            "Override a config setting for the most recently declared "
            "--config.  e.g. --set use_hyde=true --set n_results=10"
        ),
    )
    p_run.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated RAGAS metric names.  "
            "Default: faithfulness,answer_relevancy,context_precision,"
            "context_recall"
        ),
    )
    p_run.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results (default: eval_results/).",
    )
    p_run.add_argument(
        "--detail",
        action="store_true",
        help="Print per-question score breakdown after the summary table.",
    )
    p_run.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Override the LLM model used as the RAGAS judge.  Use this to "
            "pick a non-thinking model if evaluation fails with JSON "
            "parsing errors.  e.g. --judge-model gpt-4o-mini"
        ),
    )
    p_run.add_argument(
        "--judge-base-url",
        default=None,
        help=(
            "Override the base URL for the judge LLM API.  "
            "e.g. --judge-base-url https://api.openai.com/v1"
        ),
    )

    # -- generate-testset ---------------------------------------------------
    p_gen = sub.add_parser(
        "generate-testset",
        help="Generate a synthetic test set from indexed documents.",
    )
    p_gen.add_argument(
        "--num-questions", "-n",
        type=int,
        default=20,
        help="Number of QA pairs to generate (default: 20).",
    )
    p_gen.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path (.json or .csv).",
    )
    p_gen.add_argument(
        "--collection",
        default=None,
        help=(
            "ChromaDB collection to sample from "
            "(default: active collection from embedding_config.py)."
        ),
    )

    # -- results ------------------------------------------------------------
    p_res = sub.add_parser(
        "results",
        help="List or view past evaluation results.",
    )
    p_res.add_argument(
        "--file", "-f",
        default=None,
        help="Path to a specific result JSON file to display.",
    )
    p_res.add_argument(
        "--detail",
        action="store_true",
        help="Show per-question scores (use with --file).",
    )
    p_res.add_argument(
        "--config-name",
        default=None,
        help="Show detail for only this config (use with --detail).",
    )
    p_res.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing result files (default: eval_results/).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "generate-testset": cmd_generate_testset,
        "results": cmd_results,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
