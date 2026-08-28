"""Command-line entry point for live end-to-end evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.analytics import InMemoryAnalyticsEventSink
from src.api import ChatApplicationService
from src.app import (
    ApplicationConfigurationError,
    ApplicationInitializationError,
    create_application,
)

from .dataset import EvaluationDatasetError, load_dataset
from .reporting import render_console_report, report_json
from .runner import EvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the configured customer-support application."
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to an evaluation JSON dataset"
    )
    parser.add_argument(
        "--output", help="Optional path for the machine-readable JSON report"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of the readable console report"
    )
    parser.add_argument(
        "--user-id",
        default="evaluation-cli-user",
        help="Isolated user identifier used only for this evaluation run",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    services = None
    try:
        dataset = load_dataset(args.dataset)
        analytics_sink = InMemoryAnalyticsEventSink()
        services = create_application(analytics_sink=analytics_sink)
        runner = EvaluationRunner(
            ChatApplicationService(
                services.conversation_engine,
                analytics_sink=analytics_sink,
            ),
            state_reader=services.conversation_engine,
            analytics_source=analytics_sink,
            user_id=args.user_id,
        )
        report = runner.run(dataset)
        serialized = report_json(report)
        if args.output:
            Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        print(serialized if args.json else render_console_report(report))
        return 0
    except (
        EvaluationDatasetError,
        ApplicationConfigurationError,
        ApplicationInitializationError,
    ) as exc:
        print(f"Evaluation could not start: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("Evaluation output could not be written.", file=sys.stderr)
        return 2
    except Exception:
        # The CLI is a user-facing boundary; provider/driver details stay private.
        print("Evaluation failed while running the configured services.", file=sys.stderr)
        return 1
    finally:
        if services is not None:
            services.close()


if __name__ == "__main__":  # pragma: no cover - exercised through main() tests.
    raise SystemExit(main())
