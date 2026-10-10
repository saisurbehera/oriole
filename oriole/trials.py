"""Run bounded Codex gameplay trials using ChatGPT subscription authentication."""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .game import ROOT, session_path
from .viewer import read_events

DEFAULT_MODELS = ["gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-astra"]
RUNS = ROOT / ".local/trials"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def subscription_environment():
    if not shutil.which("codex"):
        raise ValueError("Install Codex CLI and sign in using codex login")
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    status = subprocess.run(
        ["codex", "login", "status"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if status.returncode or "chatgpt" not in (status.stdout + status.stderr).lower():
        raise ValueError(
            "Sign in with your ChatGPT account using codex login; these trials require subscription authentication"
        )
    return env


def game_command(session, *arguments):
    env = {**os.environ, "ORIOLE_SESSION": session}
    command = [str(ROOT / "play"), "--json", *arguments]
    result = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=40
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Local game command failed")
    return json.loads(result.stdout)


def milestones(rows):
    """Report observable progress rather than treating an episode ending as a win."""
    events = []
    deepest = 0
    highest_level = 0
    first_damage = first_kill = False
    previous_hp = None
    for row in rows:
        stats = row["stats"]
        names = []
        if row.get("event") == "reset":
            names.append("started")
        if stats["depth"] > deepest:
            deepest = stats["depth"]
            names.append(f"reached_floor_{deepest}")
        if stats["level"] > highest_level:
            highest_level = stats["level"]
            if highest_level > 1:
                names.append(f"experience_level_{highest_level}")
        if previous_hp is not None and stats["hp"] < previous_hp and not first_damage:
            names.append("first_damage")
            first_damage = True
        if "you kill " in row.get("message", "").lower() and not first_kill:
            names.append("first_monster_killed")
            first_kill = True
        if row["done"]:
            names.append(row.get("end_reason") or "episode_ended")
        for name in names:
            events.append(
                {
                    "milestone": name,
                    "keys": row["steps"],
                    "turn": stats["turn"],
                    "depth": stats["depth"],
                    "recorded_at": row.get("recorded_at"),
                }
            )
        previous_hp = stats["hp"]
    return events


def summarize(rows):
    if not rows:
        return {
            "actions": 0,
            "deepest_depth": 0,
            "highest_score": 0,
            "done": False,
            "final_stats": {},
            "unique_positions": 0,
            "search_keys": 0,
        }
    last = rows[-1]
    counts = Counter(row.get("key") for row in rows if row.get("event") == "action")
    positions = {
        (
            row["stats"].get("dungeon_number", 0),
            row["stats"].get("dungeon_level", row["stats"]["depth"]),
            row["stats"]["x"],
            row["stats"]["y"],
        )
        for row in rows
    }
    return {
        "actions": last["steps"],
        "deepest_depth": max(row["stats"]["depth"] for row in rows),
        "highest_score": max(row["stats"]["score"] for row in rows),
        "final_stats": last["stats"],
        "done": last["done"],
        "unique_positions": len(positions),
        "search_keys": counts["search"],
        "min_hp": min(row["stats"]["hp"] for row in rows),
        "key_counts": dict(counts),
    }


def stop_process(process):
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def codex_command(model, effort, folder):
    return [
        "codex",
        "exec",
        "--ignore-user-config",
        "--model",
        model,
        "--sandbox",
        "workspace-write",
        "-c",
        'approval_policy="never"',
        "-c",
        'forced_login_method="chatgpt"',
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        "sandbox_workspace_write.network_access=true",
        "--json",
        "--output-last-message",
        str(folder / "model-summary.txt"),
        "-",
    ]


def emit(event, text, as_json):
    print(json.dumps(event) if as_json else text, flush=True)


def run_trial(
    model,
    seed,
    seconds,
    max_keys,
    effort,
    env,
    runs_dir=RUNS,
    cancelled=None,
    as_json=False,
):
    started = datetime.now(timezone.utc)
    suffix = uuid.uuid4().hex[:8]
    session = "t" + suffix
    folder = (
        runs_dir / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{model}-seed{seed}-{suffix}"
    )
    folder.mkdir(parents=True)
    local = session_path(session)
    metadata = {
        "model_requested": model,
        "seed": seed,
        "character": "val-hum-fem-law",
        "seconds": seconds,
        "max_keys": max_keys,
        "reasoning_effort": effort,
        "session": session,
        "started_at": started.isoformat(),
        "auth_method": "chatgpt",
        "harness_version": 2,
    }
    write_json(folder / "metadata.json", metadata)
    process = None
    begin = time.monotonic()
    reason = "setup_error"
    error_message = None
    try:
        initial = game_command(
            session, "reset", "--seed", str(seed), "--max-steps", str(max_keys)
        )
        write_json(folder / "initial-state.json", initial)
        write_json(folder / "state.json", initial)
        metadata["initial_screen_sha256"] = hashlib.sha256(
            initial["screen"].encode()
        ).hexdigest()
        prompt = f"""Play NetHack through ./play following AGENTS.md. This is a model comparison,
not a coding task. Your isolated session is {session}, selected automatically by
ORIOLE_SESSION. Never change that variable or access another session. The game is
already started with seed {seed}, a lawful human Valkyrie, and a {max_keys}-key limit.
Read AGENTS.md and ./play observe. Do not reset or stop the game, edit any files,
use another model, delegate, call an API, or write scripts to choose actions.
Choose actions yourself through ./play act, consulting ./play knowledge if useful.
Your goal is survival and exploration toward ascension: improve equipment, find
stairs, descend, and reach as deep as possible. Use one key or short sequences and
inspect the screen after each. Read menus; watch HP and hunger. Keep track of
visited areas in your reasoning and avoid cycling around the same room. Search
near plausible secret doors or dead-end corridors, then move on. Do not spend the
whole run searching the same squares. Use actual coordinates and map symbols.
Play actively for up to {seconds} seconds or {max_keys} keys, or until the episode
ends. Do not finish after a few turns just to report status. A supervisor enforces
the budget. Stop immediately when done, and do not restart after death. At the end
report depth, score, turn, HP, and the real reason for stopping. An ended episode
is not proof of ascension. Only use ./play, reading AGENTS.md, and date commands.
"""
        (folder / "prompt.txt").write_text(prompt)
        command = codex_command(model, effort, folder)
        with (
            (folder / "codex-events.jsonl").open("w") as output,
            (folder / "codex-stderr.log").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**env, "ORIOLE_SESSION": session},
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            begin = time.monotonic()
            metadata["pid"] = process.pid
            write_json(folder / "metadata.json", metadata)
            process.stdin.write(prompt)
            process.stdin.close()
            emit(
                {"event": "trial.started", **metadata, "folder": str(folder)},
                f"Started {model}, seed {seed}: ./play --session {session} watch",
                as_json,
            )
            last_report = -30
            while True:
                elapsed = time.monotonic() - begin
                if cancelled is not None and cancelled.is_set():
                    reason = "interrupted"
                    stop_process(process)
                    break
                rows = read_events(local / "episode.jsonl")
                if rows:
                    write_json(folder / "state.json", rows[-1])
                    (folder / "episode.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
                    write_json(folder / "milestones.json", milestones(rows))
                if process.poll() is not None:
                    if rows and rows[-1]["done"]:
                        reason = (
                            "key_limit"
                            if rows[-1].get("end_reason") == "step_limit"
                            else "game_ended"
                        )
                    else:
                        reason = (
                            "model_finished"
                            if process.returncode == 0
                            else "codex_error"
                        )
                    break
                if rows and rows[-1]["done"]:
                    reason = (
                        "key_limit"
                        if rows[-1].get("end_reason") == "step_limit"
                        else "game_ended"
                    )
                    # Give Codex a short chance to collect its own final report.
                    try:
                        process.wait(timeout=min(5, max(0, seconds - elapsed)))
                    except subprocess.TimeoutExpired:
                        stop_process(process)
                    break
                if elapsed >= seconds:
                    reason = "time_limit"
                    stop_process(process)
                    break
                if elapsed - last_report >= 30 and rows:
                    last_report = elapsed
                    stats = rows[-1]["stats"]
                    emit(
                        {
                            "event": "trial.progress",
                            "model_requested": model,
                            "seed": seed,
                            "elapsed_seconds": round(elapsed, 1),
                            "stats": stats,
                            "keys": rows[-1]["steps"],
                            "stage": rows[-1].get("stage", "exploring"),
                        },
                        f"{model}: {elapsed:.0f}s | floor {stats['depth']} | turn {stats['turn']} | "
                        f"score {stats['score']} | HP {stats['hp']}/{stats['max_hp']} | "
                        f"keys {rows[-1]['steps']} | {rows[-1].get('stage', 'exploring')}",
                        as_json,
                    )
                time.sleep(1)
    except KeyboardInterrupt:
        reason = "interrupted"
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        error_message = str(exc)
        reason = "setup_error" if process is None else "codex_error"
    finally:
        if process is not None:
            stop_process(process)
        if reason == "codex_error" and error_message is None:
            errors = [
                event
                for event in read_events(folder / "codex-events.jsonl")
                if event.get("type") in ("error", "turn.failed")
            ]
            if errors:
                detail = errors[-1].get("error", errors[-1])
                error_message = (
                    detail.get("message") if isinstance(detail, dict) else str(detail)
                )
        rows = read_events(local / "episode.jsonl")
        if rows:
            write_json(folder / "state.json", rows[-1])
            write_json(folder / "final-state.json", rows[-1])
            (folder / "episode.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        result = {
            **metadata,
            **summarize(rows),
            "elapsed_seconds": round(time.monotonic() - begin, 1),
            "stop_reason": reason,
            "exit_code": process.returncode if process else None,
            "error": error_message,
            "folder": str(folder),
        }
        write_json(folder / "milestones.json", milestones(rows))
        write_json(folder / "result.json", result)
        try:
            game_command(session, "stop")
        except (ValueError, OSError, subprocess.SubprocessError):
            pass
    replay_path = folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder
    emit(
        {"event": "trial.finished", **result},
        f"Finished {model}: floor {result['deepest_depth']}, score {result['highest_score']}, "
        f"{result['actions']} keys, {reason}. Replay: ./play replay {replay_path}",
        as_json,
    )
    return result


def table(results):
    print(
        f"{'Model':<16} {'Seed':>5} {'Sec':>6} {'Floor':>5} {'Turn':>6} {'Score':>7} {'Keys':>6} {'Stop reason'}"
    )
    for result in results:
        stats = result.get("final_stats", {})
        reason = result.get(
            "stop_reason",
            "time_limit" if result.get("time_budget_reached") else "legacy",
        )
        print(
            f"{result['model_requested']:<16} {str(result.get('seed', '?')):>5} "
            f"{result.get('elapsed_seconds', 0):>6.0f} {result.get('deepest_depth', 0):>5} "
            f"{stats.get('turn', 0):>6} {result.get('highest_score', 0):>7} "
            f"{result.get('actions', 0):>6} {reason}"
        )


def add_trial_parsers(sub):
    for name in ("trial", "bench"):
        parser = sub.add_parser(
            name, help="Run bounded Codex trials using your ChatGPT subscription"
        )
        if name == "trial":
            parser.add_argument("--model", required=True)
            parser.add_argument("--seed", type=int, default=43)
        else:
            parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
            parser.add_argument("--seeds", nargs="+", type=int, default=[43])
            parser.add_argument(
                "--jobs", type=int, default=1, help="Concurrent independent games (1–4)"
            )
        parser.add_argument("--seconds", type=int, default=300)
        parser.add_argument("--max-keys", type=int, default=1500)
        parser.add_argument(
            "--effort",
            choices=("low", "medium", "high", "xhigh", "max"),
            default="medium",
        )
    results = sub.add_parser("results", help="Show the saved model comparison table")
    results.add_argument("--runs-dir", type=Path, default=RUNS)
    results.add_argument(
        "--live",
        action="store_true",
        help="Include running trials and their watch commands",
    )


def dispatch(args):
    if args.command == "results":
        results = [
            json.loads(path.read_text())
            for path in sorted(args.runs_dir.glob("*/result.json"))
        ]
        if args.live:
            for path in sorted(args.runs_dir.glob("*/metadata.json")):
                if (path.parent / "result.json").exists():
                    continue
                rows = read_events(path.parent / "episode.jsonl")
                if not rows:
                    continue
                metadata = json.loads(path.read_text())
                elapsed = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(metadata["started_at"])
                ).total_seconds()
                results.append(
                    {
                        **metadata,
                        **summarize(rows),
                        "elapsed_seconds": elapsed,
                        "stop_reason": "running",
                        "folder": str(path.parent),
                    }
                )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            table(results)
            if args.live:
                for result in results:
                    if result.get("stop_reason") == "running":
                        print(
                            f"Watch {result['model_requested']}: ./play --session {result['session']} watch"
                        )
        return
    if args.seconds < 1 or args.max_keys < 1:
        raise ValueError("seconds and max-keys must be positive")
    models = [args.model] if args.command == "trial" else args.models
    seeds = [args.seed] if args.command == "trial" else args.seeds
    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", model) for model in models
    ):
        raise ValueError("Supply valid Codex model IDs")
    if any(not 0 <= seed <= 4294967295 for seed in seeds):
        raise ValueError("seeds must be between 0 and 4294967295")
    jobs = 1 if args.command == "trial" else args.jobs
    if not 1 <= jobs <= 4:
        raise ValueError("jobs must be between 1 and 4")
    env = subscription_environment()
    tasks = [(model, seed) for seed in seeds for model in models]
    cancelled = threading.Event()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                run_trial,
                model,
                seed,
                args.seconds,
                args.max_keys,
                args.effort,
                env,
                cancelled=cancelled,
                as_json=args.json,
            )
            for model, seed in tasks
        ]
        try:
            results = [future.result() for future in futures]
        except KeyboardInterrupt:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
    if args.json:
        print(json.dumps({"event": "benchmark.completed", "results": results}))
    else:
        table(results)
