"""
Experiment Runner
=================
Runs the controlled A/B comparison experiment for RQ3.

Experiment design:
  - Same set of attack scenarios run against both defense modes
  - Results logged separately per mode
  - EvaluationAnalyzer computes M1/M2/M3 for each mode

Usage:
    python run_experiment.py --scenarios data/attack_scenarios.jsonl
    python run_experiment.py --interactive
"""

import argparse
import json
import logging
import os

from middleware import DeceptionMiddleware, DefenseMode
from evaluation import EvaluationAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_scenarios(path: str) -> list[dict]:
    """
    Load attack scenarios from JSONL.
    Each line: {"session_id": "s1", "turns": ["prompt1", "prompt2", ...], "attack_type": "jailbreak"}
    """
    scenarios = []
    with open(path) as f:
        for line in f:
            scenarios.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(scenarios)} scenarios from {path}")
    return scenarios


def run_experiment(scenarios: list[dict], mode: DefenseMode):
    """Run all scenarios through the middleware in the given defense mode."""
    mw = DeceptionMiddleware(mode=mode)
    logger.info(f"\n{'='*50}\nRunning experiment: mode={mode.value}\n{'='*50}")

    for scenario in scenarios:
        session_id = f"{mode.value}_{scenario['session_id']}"
        mw.reset_session(session_id)

        logger.info(f"  Scenario: {scenario['session_id']} | attack: {scenario.get('attack_type', 'unknown')}")
        for turn_prompt in scenario["turns"]:
            response = mw.process(turn_prompt, session_id=session_id)
            logger.info(
                f"    turn={response.turn_index} "
                f"triggered={response.defense_triggered} "
                f"label={response.classification.label}"
            )

    return mw.logger.log_path


def interactive_demo():
    """
    Interactive CLI for manually testing the middleware.
    Useful for qualitative evaluation and demos.
    """
    print("\nLLM Defense Middleware — Interactive Demo")
    print("Type 'mode block' or 'mode deceive' to switch defense mode.")
    print("Type 'quit' to exit.\n")

    mode = DefenseMode.DECEIVE
    mw = DeceptionMiddleware(mode=mode)
    session = "interactive"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input == "quit":
            break
        if user_input.startswith("mode "):
            chosen = user_input.split(" ", 1)[1]
            if chosen in ("block", "deceive", "none"):
                mode = DefenseMode(chosen)
                mw = DeceptionMiddleware(mode=mode)
                mw.reset_session(session)
                print(f"[Switched to mode: {mode.value}]\n")
            else:
                print("[Unknown mode. Use: block | deceive | none]")
            continue

        response = mw.process(user_input, session_id=session)

        if response.defense_triggered:
            print(f"[⚠ Defense triggered — {response.defense_mode} | attack: {response.classification.attack_type} | conf: {response.classification.confidence:.2f}]")
        print(f"Assistant: {response.final_response}\n")

    # Show summary
    log_path = mw.logger.log_path
    print(f"\nSession log saved to: {log_path}")
    analyzer = EvaluationAnalyzer(log_path)
    analyzer.print_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Defense Middleware Experiment Runner")
    parser.add_argument("--scenarios",   type=str, help="Path to attack scenarios JSONL file")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI demo")
    parser.add_argument("--api-key",     type=str, default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--report",      type=str, help="Compute metrics from existing log file")
    args = parser.parse_args()

    if args.interactive:
        interactive_demo()

    elif args.scenarios:
        scenarios = load_scenarios(args.scenarios)
        log_block   = run_experiment(scenarios, DefenseMode.BLOCK)
        log_deceive = run_experiment(scenarios, DefenseMode.DECEIVE)

        print("\n--- BLOCK mode ---")
        EvaluationAnalyzer(log_block).print_report()

        print("\n--- DECEIVE mode ---")
        EvaluationAnalyzer(log_deceive).print_report()

    elif args.report:
        EvaluationAnalyzer(args.report).print_report()

    else:
        parser.print_help()
