"""Evaluation result storage and CLI display.

An :class:`EvalReport` aggregates per-configuration RAGAS scores and
supports serialisation to JSON, loading from JSON, and rendering as a
formatted terminal table.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class QuestionResult:
    """Scores for a single question within one config run."""

    question: str
    ground_truth: str
    answer: str
    retrieved_contexts: List[str]
    scores: Dict[str, Optional[float]]  # metric_name -> score


@dataclass
class ConfigResult:
    """Evaluation results for one configuration profile."""

    config_name: str
    overrides: Dict[str, object]
    aggregate_scores: Dict[str, Optional[float]]
    per_question: List[QuestionResult] = field(default_factory=list)


@dataclass
class EvalReport:
    """Complete evaluation report for one or more configs."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    testset_path: str = ""
    num_questions: int = 0
    metrics: List[str] = field(default_factory=list)
    configs: List[ConfigResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, path: str) -> None:
        """Save the report to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False,
                      default=str)
        print(f"Results saved to: {path}")

    @classmethod
    def from_json(cls, path: str) -> "EvalReport":
        """Load a report from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        configs = []
        for c in data.get("configs", []):
            per_q = [
                QuestionResult(**q) for q in c.get("per_question", [])
            ]
            configs.append(ConfigResult(
                config_name=c["config_name"],
                overrides=c.get("overrides", {}),
                aggregate_scores=c.get("aggregate_scores", {}),
                per_question=per_q,
            ))

        return cls(
            timestamp=data.get("timestamp", ""),
            testset_path=data.get("testset_path", ""),
            num_questions=data.get("num_questions", 0),
            metrics=data.get("metrics", []),
            configs=configs,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_table(self) -> None:
        """Print a formatted comparison table to stdout."""
        if not self.configs:
            print("No results to display.")
            return

        metrics = self.metrics or []
        if not metrics and self.configs:
            metrics = list(self.configs[0].aggregate_scores.keys())

        # Column widths.
        name_w = max(len("Config"),
                     max(len(c.config_name) for c in self.configs))
        col_w = max(12, max((len(m) for m in metrics), default=12))

        # Header.
        header = f" {'Config':<{name_w}} "
        for m in metrics:
            header += f"| {m:^{col_w}} "
        header += f"| {'Avg':^{col_w}} "

        sep = "-" * len(header)
        print()
        print(sep)
        print(header)
        print(sep)

        best_avg = -1.0
        best_name = ""

        for c in self.configs:
            row = f" {c.config_name:<{name_w}} "
            scores = []
            for m in metrics:
                val = c.aggregate_scores.get(m)
                if val is not None:
                    row += f"| {val:^{col_w}.4f} "
                    scores.append(val)
                else:
                    row += f"| {'N/A':^{col_w}} "

            avg = sum(scores) / len(scores) if scores else 0.0
            row += f"| {avg:^{col_w}.4f} "
            print(row)

            if avg > best_avg:
                best_avg = avg
                best_name = c.config_name

        print(sep)
        if best_name:
            print(f"\n  Best overall: {best_name} (avg: {best_avg:.4f})")
        print()

    def print_detail(self, config_name: Optional[str] = None) -> None:
        """Print per-question scores for one or all configs."""
        targets = self.configs
        if config_name:
            targets = [c for c in self.configs if c.config_name == config_name]
            if not targets:
                print(f"Config '{config_name}' not found in this report.")
                return

        for c in targets:
            print(f"\n{'=' * 60}")
            print(f"  Config: {c.config_name}")
            if c.overrides:
                print(f"  Overrides: {c.overrides}")
            print(f"{'=' * 60}")

            for i, q in enumerate(c.per_question):
                print(f"\n  Q{i + 1}: {q.question}")
                print(f"  Expected: {q.ground_truth[:120]}")
                print(f"  Answer:   {q.answer[:120]}")
                score_parts = []
                for m, v in q.scores.items():
                    if v is not None:
                        score_parts.append(f"{m}={v:.3f}")
                    else:
                        score_parts.append(f"{m}=N/A")
                print(f"  Scores:   {', '.join(score_parts)}")

            print()


# -----------------------------------------------------------------------
# Utilities for listing saved results
# -----------------------------------------------------------------------

def list_result_files(results_dir: str) -> List[str]:
    """Return paths to all result JSON files in *results_dir*, newest first."""
    if not os.path.isdir(results_dir):
        return []
    files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def print_result_list(results_dir: str) -> None:
    """Print a table of all saved evaluation results."""
    files = list_result_files(results_dir)
    if not files:
        print(f"No evaluation results found in: {results_dir}")
        return

    print(f"\n  Evaluation results in: {results_dir}\n")
    print(f"  {'#':<4} {'Date':20} {'Questions':>10} {'Configs':>8}  File")
    print(f"  {'-' * 70}")

    for i, fpath in enumerate(files):
        try:
            report = EvalReport.from_json(fpath)
            ts = report.timestamp[:19].replace("T", " ")
            nq = report.num_questions
            nc = len(report.configs)
        except Exception:
            ts = "???"
            nq = "?"
            nc = "?"
        print(f"  {i + 1:<4} {ts:20} {nq:>10} {nc:>8}  {os.path.basename(fpath)}")

    print()
