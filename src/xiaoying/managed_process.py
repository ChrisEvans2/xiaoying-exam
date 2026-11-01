"""Register detached document parsers before exec so launcher shutdown can reap them."""

import json
import os
from pathlib import Path
import subprocess
import sys

REGISTRY_ENV = "XIAOYING_PROCESS_REGISTRY"


def process_identity(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,pgid=,lstart="], capture_output=True, text=True
    )
    fields = result.stdout.split(maxsplit=2)
    if len(fields) != 3:
        return None
    return {"pid": int(fields[0]), "group": int(fields[1]), "started": fields[2].strip()}


def registered_groups(directory):
    groups = set()
    for path in directory.glob("*.json"):
        try:
            recorded = json.loads(path.read_text())
            current = process_identity(recorded["pid"])
            if current == recorded and current["group"] == current["pid"]:
                groups.add(current["group"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return groups


def main():
    parent, command = int(sys.argv[1]), sys.argv[2:]
    directory = Path(os.environ[REGISTRY_ENV])
    # A Worker can die while this interpreter is starting. Never start work for a dead parent.
    if os.getppid() != parent or not command or not directory.is_dir() or (directory / "closing").exists():
        return 1
    identity = process_identity(os.getpid())
    if identity is None or identity["group"] != os.getpid():
        return 1
    path = directory / f"{os.getpid()}.json"
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity))
        temporary.replace(path)
        if os.getppid() != parent or (directory / "closing").exists():
            path.unlink(missing_ok=True)
            return 1
        os.execvpe(command[0], command, os.environ)
    finally:
        # Only reached on exec failure; on success the Worker's wait removes this file.
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
