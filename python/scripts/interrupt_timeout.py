from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a command and send keyboard interrupt if it exceeds timeout.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Maximum allowed runtime in seconds before interrupt (default: 300).",
    )
    parser.add_argument(
        "--grace-seconds",
        type=int,
        default=20,
        help="Grace period after interrupt before force terminate.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Example: -- python -m pytest tests -q",
    )
    return parser


def _send_keyboard_interrupt(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP allows CTRL_BREAK_EVENT delivery.
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGINT)


def main() -> int:
    args = _build_parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        print("No command provided.", file=sys.stderr)
        return 2

    timeout = max(int(args.timeout_seconds), 1)
    grace = max(int(args.grace_seconds), 1)

    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    print(f"[timeout-runner] starting: {' '.join(command)}")
    print(f"[timeout-runner] timeout: {timeout}s, grace: {grace}s")

    start = time.monotonic()
    proc = subprocess.Popen(command, creationflags=creation_flags)

    while True:
        code = proc.poll()
        if code is not None:
            elapsed = time.monotonic() - start
            print(f"[timeout-runner] completed in {elapsed:.1f}s with exit code {code}")
            return int(code)

        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            print("[timeout-runner] timeout exceeded, sending keyboard interrupt...")
            try:
                _send_keyboard_interrupt(proc)
            except Exception as exc:
                print(f"[timeout-runner] interrupt failed: {exc}", file=sys.stderr)

            interrupt_start = time.monotonic()
            while True:
                code_after = proc.poll()
                if code_after is not None:
                    total_elapsed = time.monotonic() - start
                    print(
                        f"[timeout-runner] interrupted after {total_elapsed:.1f}s with exit code {code_after}"
                    )
                    return int(code_after)

                if time.monotonic() - interrupt_start >= grace:
                    print("[timeout-runner] grace exceeded, terminating process...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        print(
                            "[timeout-runner] terminate timed out, killing process..."
                        )
                        proc.kill()
                        proc.wait(timeout=10)
                    total_elapsed = time.monotonic() - start
                    final_code = (
                        int(proc.returncode) if proc.returncode is not None else 1
                    )
                    print(
                        f"[timeout-runner] stopped after {total_elapsed:.1f}s with exit code {final_code}"
                    )
                    return final_code

                time.sleep(0.1)

        time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
