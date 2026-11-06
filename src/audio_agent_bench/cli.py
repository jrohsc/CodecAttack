"""`python -m audio_agent_bench run <scenario_dir> [--model] [--channel] [--out]`."""
from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_scenario
from .scenario_loader import list_scenarios, load_scenario


def _cmd_run(args: argparse.Namespace) -> None:
    scenario = load_scenario(args.scenario)
    if args.model:
        scenario.target_models = [args.model]
    if args.channel:
        from .models import ChannelSpec
        scenario.channels = [ChannelSpec(kind=args.channel)]
    out = Path(args.out or Path(args.scenario) / "results")
    run_scenario(scenario, out_dir=out)
    print(f"done -> {out}")


def _cmd_list(args: argparse.Namespace) -> None:
    for d in list_scenarios(args.root):
        print(d)


def main() -> None:
    p = argparse.ArgumentParser("audio_agent_bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("scenario")
    pr.add_argument("--model", default=None)
    pr.add_argument("--channel", default=None)
    pr.add_argument("--out", default=None)
    pr.set_defaults(func=_cmd_run)

    pl = sub.add_parser("list")
    pl.add_argument("--root", default="scenarios")
    pl.set_defaults(func=_cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
