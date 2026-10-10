"""Local Unix socket keeps the game alive across separate Codex shell calls."""

import argparse
import json
import os
import socket
import socketserver
import subprocess
import sys
import time
from pathlib import Path

from . import game
from .game import Game, KEYS, LOCAL, ROOT, knowledge, session_path

SOCKET = LOCAL / "game.sock"


def request(payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(30)
        client.connect(str(SOCKET))
        client.sendall(json.dumps(payload).encode() + b"\n")
        with client.makefile("rb") as stream:
            result = json.loads(stream.readline())
    if "error" in result:
        raise ValueError(result["error"])
    return result


def serve():
    LOCAL.mkdir(parents=True, exist_ok=True)
    # Never remove a live server's socket.
    if SOCKET.exists():
        try:
            request({"command": "ping"})
        except (ConnectionRefusedError, FileNotFoundError):
            SOCKET.unlink(missing_ok=True)
        else:
            raise ValueError("A game server is already running")
    game = Game()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                payload = json.loads(self.rfile.readline(16384))
                command = payload.pop("command")
                if command == "ping":
                    result = {"pid": os.getpid(), "running": True}
                elif command == "reset":
                    result = game.reset(**payload)
                elif command == "observe":
                    result = game.observe()
                elif command == "act":
                    result = game.act(**payload)
                elif command == "stop":
                    result = {"stopped": True}
                    self.server.running = False
                else:
                    raise ValueError(f"Unknown command: {command}")
            except Exception as exc:
                result = {"error": str(exc)}
            self.wfile.write(json.dumps(result).encode() + b"\n")

    try:
        with socketserver.UnixStreamServer(str(SOCKET), Handler) as server:
            SOCKET.chmod(0o600)
            server.running = True
            while server.running:
                server.handle_request()
    finally:
        game.close()
        SOCKET.unlink(missing_ok=True)


def ensure_server():
    try:
        return request({"command": "ping"})
    except (FileNotFoundError, ConnectionRefusedError):
        pass
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / "game-server.log").open("a") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "oriole", "serve"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    for _ in range(100):
        try:
            return request({"command": "ping"})
        except (FileNotFoundError, ConnectionRefusedError):
            if child.poll() is not None:
                break
            time.sleep(0.05)
    raise ValueError(f"Game server could not start; see {LOCAL / 'game-server.log'}")


def display(result, as_json=False):
    if as_json or "screen" not in result:
        print(json.dumps(result, indent=2))
    else:
        print(result["screen"])
        print(
            json.dumps({key: value for key, value in result.items() if key != "screen"})
        )


def main():
    parser = argparse.ArgumentParser(
        description="Play NetHack locally: ./play reset, observe, act north"
    )
    parser.add_argument(
        "--json", action="store_true", help="Return machine readable JSON"
    )
    parser.add_argument(
        "--session",
        default=os.environ.get("ORIOLE_SESSION", "default"),
        help="Separate game session (default: default)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    reset = sub.add_parser(
        "reset", help="Start a new game (replaces the current episode)"
    )
    reset.add_argument("--seed", type=int, default=42)
    reset.add_argument("--character", default="val-hum-fem-law")
    reset.add_argument("--max-steps", type=int, default=100000)
    sub.add_parser("observe", help="Read the current screen without advancing the game")
    act = sub.add_parser("act", help="Send named actions or literal NetHack keys")
    act.add_argument("keys", nargs="+")
    act.add_argument("--repeat", type=int, default=1)
    lookup = sub.add_parser("knowledge", help="Search the local wiki knowledge graph")
    lookup.add_argument("query")
    lookup.add_argument("--limit", type=int, default=12, choices=range(1, 101))
    for name in ("actions", "status", "stop", "serve", "human"):
        sub.add_parser(name)
    from .trials import add_trial_parsers

    add_trial_parsers(sub)
    watch = sub.add_parser("watch", help="Watch a live game without taking actions")
    watch.add_argument("--run", type=Path, help="Watch a trial's saved live state")
    watch.add_argument("--once", action="store_true")
    watch.add_argument("--interval", type=float, default=1)
    replay = sub.add_parser(
        "replay", help="Replay the saved dungeon screens from a trial"
    )
    replay.add_argument("run", type=Path)
    replay.add_argument("--delay", type=float, default=0.1)
    replay.add_argument("--start", type=int, default=0, help="Start at this key number")
    replay.add_argument("--frames", type=int, help="Limit the number of frames shown")
    args = parser.parse_args()
    try:
        global LOCAL, SOCKET
        LOCAL = session_path(args.session)
        SOCKET = LOCAL / "game.sock"
        game.LOCAL = LOCAL
        os.environ["ORIOLE_SESSION"] = args.session
        if args.command in ("trial", "bench", "results"):
            from .trials import dispatch

            dispatch(args)
            return
        if args.command in ("watch", "replay"):
            from .viewer import watch_game, replay_game

            if args.command == "watch":
                watch_game(args, lambda: request({"command": "observe"}))
            else:
                replay_game(args)
            return
        if args.command == "serve":
            serve()
            return
        if args.command == "knowledge":
            result = knowledge(args.query, args.limit)
        elif args.command == "actions":
            result = {
                "named_keys": KEYS,
                "usage": "./play act open east; ./play act '>'",
                "note": "Each key is processed in sequence. Inspect prompts before answering.",
            }
        elif args.command in ("status", "stop"):
            result = request(
                {"command": "ping" if args.command == "status" else "stop"}
            )
        else:
            ensure_server()
            if args.command == "human":
                try:
                    result = request({"command": "observe"})
                except ValueError:
                    result = request({"command": "reset"})
                while True:
                    display(result)
                    if result["done"]:
                        break
                    keys = input(
                        "Action (north, s, open east; :quit to leave): "
                    ).split()
                    if keys == [":quit"]:
                        break
                    if keys:
                        try:
                            result = request({"command": "act", "keys": keys})
                        except ValueError as exc:
                            print(exc, file=sys.stderr)
                return
            payload = {"command": args.command}
            if args.command == "reset":
                payload.update(
                    seed=args.seed, character=args.character, max_steps=args.max_steps
                )
            elif args.command == "act":
                payload.update(keys=args.keys, repeat=args.repeat)
            result = request(payload)
        display(result, args.json)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"oriole: {exc}\n")
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
