"""Launcher process and failure tests: temporary files, loopback sockets, no real DB/models."""

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.request import urlopen

import psycopg
import pytest

from xiaoying import launcher
from xiaoying.managed_process import process_identity


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.03)
    raise AssertionError("Timed out waiting for local process fixture")


def running(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


def kill_fixture(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.fixture
def processes(tmp_path):
    manager = launcher.Processes(tmp_path)
    yield manager
    manager.stop()


def test_reserved_port_excludes_competitor_and_releases_after_close():
    reserved = launcher.available_port(0)
    port = reserved.getsockname()[1]
    try:
        with pytest.raises(launcher.StartupError, match="端口"):
            launcher.available_port(port)
    finally:
        reserved.close()
    launcher.available_port(port).close()


@pytest.mark.parametrize("profile", ["live", "mock"])
def test_launcher_children_receive_explicit_profile(monkeypatch, tmp_path, profile):
    monkeypatch.setenv("XIAOYING_PROFILE", "mock" if profile == "live" else "live")
    output = tmp_path / "profile.txt"
    processes = launcher.Processes(tmp_path, profile=profile)
    try:
        processes.run(
            "profile", [sys.executable, "-c",
                        "import os,sys; from pathlib import Path; "
                        "Path(sys.argv[1]).write_text(os.environ['XIAOYING_PROFILE'])", str(output)],
            timeout=5,
        )
        assert output.read_text() == profile
    finally:
        processes.stop()


HTTP_FIXTURE = """
import json, os, sys, uvicorn
from pathlib import Path
from fastapi import FastAPI
app = FastAPI()
@app.get('/health')
def health():
    return {'status': 'ok'}
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(str(os.getpid()))
uvicorn.run(app, fd=int(sys.argv[1]), log_level='error')
"""


def test_reserved_socket_passes_to_actual_uvicorn_without_rebinding(processes):
    listener = launcher.available_port(0)
    port = listener.getsockname()[1]
    proc = processes.start(
        "api-fixture",
        [sys.executable, "-c", HTTP_FIXTURE, str(listener.fileno())],
        pass_fds=(listener.fileno(),),
    )
    listener.close()
    launcher.wait_ready(processes, [], f"http://127.0.0.1:{port}", timeout=5)
    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
        assert json.load(response) == {"status": "ok"}
    assert proc.poll() is None
    processes.stop()
    assert proc.poll() is not None


@pytest.mark.parametrize("parent_exits_first", [False, True])
def test_cleanup_includes_descendant_with_independent_session(processes, tmp_path, parent_exits_first):
    pidfile = tmp_path / "detached.pid"
    child_ready = tmp_path / "child-executed"
    release = tmp_path / "parent-may-exit"
    child_code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(child_ready)!r}).touch(); time.sleep(120)"
    )
    parent_code = textwrap.dedent(f"""
        import os, subprocess, sys, time
        from pathlib import Path
        child = subprocess.Popen([sys.executable, '-m', 'xiaoying.managed_process',
            str(os.getpid()), sys.executable, '-c', {child_code!r}], start_new_session=True)
        Path({str(pidfile)!r}).write_text(str(child.pid))
        while not Path({str(release)!r}).exists():
            time.sleep(0.02)
    """)
    parent = processes.start("build-fixture", [sys.executable, "-c", parent_code])
    pid = int(eventually(lambda: pidfile.read_text() if pidfile.exists() else None))
    try:
        eventually(child_ready.exists)
        assert running(pid)
        if parent_exits_first:
            release.touch()
            eventually(lambda: parent.poll() is not None)
            with pytest.raises(launcher.StartupError, match="已退出"):
                processes.check()
        processes.stop()
        eventually(lambda: not running(pid), timeout=2)
        assert parent.poll() is not None
    finally:
        if running(pid):
            kill_fixture(pid)


def test_detached_parser_does_not_execute_if_worker_died_before_registration(processes, tmp_path):
    pidfile = tmp_path / "child.pid"
    start_wrapper = tmp_path / "start-wrapper"
    exit_parent = tmp_path / "exit-parent"
    executed = tmp_path / "must-not-execute"
    payload = f"from pathlib import Path; Path({str(executed)!r}).touch()"
    delayed_wrapper = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        while not Path({str(start_wrapper)!r}).exists():
            time.sleep(0.02)
        from xiaoying.managed_process import main
        sys.argv = ['managed-process', sys.argv[1], sys.executable, '-c', {payload!r}]
        raise SystemExit(main())
    """)
    parent_code = textwrap.dedent(f"""
        import os, subprocess, sys, time
        from pathlib import Path
        proc = subprocess.Popen([sys.executable, '-c', {delayed_wrapper!r}, str(os.getpid())],
                                start_new_session=True)
        Path({str(pidfile)!r}).write_text(str(proc.pid))
        while not Path({str(exit_parent)!r}).exists():
            time.sleep(0.02)
    """)
    parent = processes.start("worker-fixture", [sys.executable, "-c", parent_code])
    pid = int(eventually(lambda: pidfile.read_text() if pidfile.exists() else None))
    try:
        exit_parent.touch()
        eventually(lambda: parent.poll() is not None)
        start_wrapper.touch()
        eventually(lambda: not running(pid))
        assert not executed.exists()
        assert not list(processes.registry.glob("*.json"))
    finally:
        if running(pid):
            kill_fixture(pid)


def test_stale_registration_does_not_kill_unrelated_process(processes):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        identity = process_identity(unrelated.pid)
        assert identity
        identity["started"] = "a stale process with a reused PID"
        (processes.registry / f"{unrelated.pid}.json").write_text(json.dumps(identity))
        processes.start("owned-fixture", [sys.executable, "-c", "import time; time.sleep(120)"])
        processes.stop()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_shutdown_refuses_parser_started_after_initial_snapshot(processes, tmp_path, monkeypatch):
    release = tmp_path / "start-parser"
    pidfile = tmp_path / "late-parser.pid"
    executed = tmp_path / "late-parser-executed"
    payload = f"import time; from pathlib import Path; Path({str(executed)!r}).touch(); time.sleep(120)"
    parent_code = textwrap.dedent(f"""
        import os, subprocess, sys, time
        from pathlib import Path
        while not Path({str(release)!r}).exists():
            time.sleep(0.02)
        child = subprocess.Popen([sys.executable, '-m', 'xiaoying.managed_process',
            str(os.getpid()), sys.executable, '-c', {payload!r}], start_new_session=True)
        Path({str(pidfile)!r}).write_text(str(child.pid))
        child.wait()
        time.sleep(120)
    """)
    processes.start("worker-fixture", [sys.executable, "-c", parent_code])
    original_groups = launcher.registered_groups
    first_snapshot = True

    def after_snapshot(directory):
        nonlocal first_snapshot
        groups = original_groups(directory)
        if first_snapshot:
            first_snapshot = False
            release.touch()
            pid = int(eventually(lambda: pidfile.read_text() if pidfile.exists() else None))
            eventually(lambda: executed.exists() or not running(pid))
        return groups

    monkeypatch.setattr(launcher, "registered_groups", after_snapshot)
    try:
        processes.stop()
        assert not executed.exists()
        pid = int(pidfile.read_text())
        eventually(lambda: not running(pid))
    finally:
        if pidfile.exists() and running(int(pidfile.read_text())):
            kill_fixture(int(pidfile.read_text()))


@pytest.mark.parametrize("exit_code", [0, 7])
def test_completed_service_is_a_supervision_failure(processes, exit_code):
    proc = processes.start("service-fixture", [sys.executable, "-c", f"raise SystemExit({exit_code})"])
    eventually(lambda: proc.poll() is not None)
    with pytest.raises(launcher.StartupError, match="service-fixture 已退出"):
        processes.check()


def test_timed_out_preparation_remains_owned_until_cleanup(processes):
    with pytest.raises(launcher.StartupError, match="超时"):
        processes.run("slow-preparation", [sys.executable, "-c", "import time;time.sleep(120)"], 0.1)
    proc = processes.children["slow-preparation"]
    processes.stop()
    assert proc.poll() is not None
    assert not running(proc.pid)


DEFAULT_DATABASE = "postgresql://xiaoying:xiaoying@127.0.0.1:55432/xiaoying"


class PreparationRecorder:
    def __init__(self):
        self.calls = []

    def run(self, *args):
        self.calls.append(args)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://xiaoying:xiaoying@db.example.invalid:55432/xiaoying",
        "postgresql://xiaoying:xiaoying@127.0.0.1:5432/xiaoying",
        "postgresql://xiaoying:custom-secret@127.0.0.1:55432/xiaoying",
        "postgresql://xiaoying:xiaoying@127.0.0.1:55432/custom",
        DEFAULT_DATABASE + "?hostaddr=127.0.0.1",
    ],
)
def test_unavailable_custom_database_never_starts_docker(monkeypatch, url):
    processes = PreparationRecorder()
    monkeypatch.setattr(launcher, "database_ready", lambda _: False)
    with pytest.raises(launcher.StartupError, match="配置的 PostgreSQL") as error:
        launcher.prepare_database(SimpleNamespace(database_url=url), processes)
    assert not processes.calls
    assert "custom-secret" not in str(error.value)


def test_reachable_custom_database_is_reused_and_migrated(monkeypatch):
    processes = PreparationRecorder()
    migrated = []
    url = "postgresql://custom:private@db.example.invalid:5432/company"
    monkeypatch.setattr(launcher, "database_ready", lambda value: value == url)
    monkeypatch.setattr(launcher, "migrate", migrated.append)
    launcher.prepare_database(SimpleNamespace(database_url=url), processes)
    assert not processes.calls
    params = launcher.conninfo_to_dict(migrated[0])
    assert params["host"] == "db.example.invalid" and params["dbname"] == "company"
    assert params["connect_timeout"] == "5"


def test_default_database_auto_start_rechecks_readiness(monkeypatch):
    processes = PreparationRecorder()
    migrated = []
    readiness = iter([False, True])
    monkeypatch.setattr(launcher, "database_ready", lambda _: next(readiness))
    monkeypatch.setattr(
        launcher.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError())
    )
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "/fixture/docker")
    monkeypatch.setattr(launcher, "migrate", migrated.append)
    launcher.prepare_database(SimpleNamespace(database_url=DEFAULT_DATABASE), processes)
    assert len(processes.calls) == 1
    command = processes.calls[0][1]
    assert command[:5] == ["docker", "compose", "-p", "xiaoying-exam", "up"]
    assert "--wait" in command and command[-1] == "db"
    assert len(migrated) == 1


def test_default_database_auth_failure_does_not_start_competing_container(monkeypatch):
    processes = PreparationRecorder()
    monkeypatch.setattr(launcher, "database_ready", lambda _: False)
    monkeypatch.setattr(launcher.socket, "create_connection", lambda *a, **k: nullcontext())
    with pytest.raises(launcher.StartupError, match="数据库端口已有服务"):
        launcher.prepare_database(SimpleNamespace(database_url=DEFAULT_DATABASE), processes)
    assert not processes.calls


def test_invalid_database_url_is_rejected_without_exposing_connection_details(monkeypatch):
    processes = PreparationRecorder()
    secret = "review-private-password"
    malformed_url = f"postgresql://xiaoying:{secret}@127.0.0.1:55432/xiaoying?invalid"

    def must_not_probe(_):
        pytest.fail("Malformed configuration must be rejected before connecting")

    monkeypatch.setattr(launcher, "database_ready", must_not_probe)
    with pytest.raises(launcher.StartupError, match="DATABASE_URL 格式无效") as error:
        launcher.prepare_database(SimpleNamespace(database_url=malformed_url), processes)
    assert secret not in str(error.value)
    assert malformed_url not in str(error.value)
    assert error.value.__suppress_context__
    assert not processes.calls


def test_migration_error_is_sanitized(monkeypatch):
    monkeypatch.setattr(launcher, "database_ready", lambda _: True)

    def fail(_):
        raise psycopg.OperationalError("private-password should not escape")

    monkeypatch.setattr(launcher, "migrate", fail)
    with pytest.raises(launcher.StartupError, match="数据库初始化失败") as error:
        launcher.prepare_database(SimpleNamespace(database_url=DEFAULT_DATABASE), PreparationRecorder())
    assert "private-password" not in str(error.value)


def test_stale_worker_readiness_cannot_mark_launcher_ready(processes, tmp_path):
    proc = processes.start("worker-fixture", [sys.executable, "-c", "import time; time.sleep(120)"])
    marker = tmp_path / "ready"
    marker.write_text(str(proc.pid + 1))
    with pytest.raises(launcher.StartupError, match="服务启动超时"):
        launcher.wait_ready(processes, [(marker, proc)], "http://127.0.0.1:1", timeout=0.1)


def make_harness(tmp_path, *, slow_prepare=False):
    """Keep launcher main/locking/sockets/cleanup real; replace only DB and business children."""
    root = tmp_path / "repository with spaces"
    root.mkdir()
    child_code = textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        Path(sys.argv[1]).write_text(str(os.getpid()))
        Path(sys.argv[2]).write_text(str(os.getpid()))
        time.sleep(120)
    """)
    harness = root / "fixture_launcher.py"
    harness.write_text(
        textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        from types import SimpleNamespace
        from xiaoying import launcher
        launcher.ROOT = Path({str(root)!r})
        launcher.load_settings = lambda **kwargs: SimpleNamespace(provider_mode='mock', deepseek_api_key='')
        launcher.prepare_database = lambda *args: None
        if {slow_prepare!r}:
            def prepare(processes):
                processes.run('preparation', [sys.executable, '-c', {child_code!r},
                    str(launcher.ROOT / 'preparation.ready'), str(launcher.ROOT / 'preparation.pid')], 60)
            launcher.prepare_frontend = prepare
        else:
            launcher.prepare_frontend = lambda *args: None
        original_start = launcher.Processes.start
        def start_fixture(self, name, command, pass_fds=()):
            if name == 'api':
                command = [sys.executable, '-c', {HTTP_FIXTURE!r}, str(pass_fds[0]),
                           str(launcher.ROOT / (name + '.pid'))]
            elif name in ('run', 'build', 'scheduler'):
                ready = command[command.index('--ready-file') + 1]
                command = [sys.executable, '-c', {child_code!r}, ready,
                           str(launcher.ROOT / (name + '.pid'))]
            return original_start(self, name, command, pass_fds)
        launcher.Processes.start = start_fixture
        raise SystemExit(launcher.main())
    """)
    )
    return root, harness


def start_harness(root, harness, port, suffix="first"):
    log = root / f"{suffix}.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [sys.executable, str(harness), *(["--port", str(port)] if port else []), "--no-browser"],
            cwd=root.parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process, log


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_harness_ready(process, log):
    eventually(lambda: "已启动：" in log.read_text(), timeout=8)
    assert process.poll() is None, log.read_text()


def test_main_prevents_duplicate_start_and_ctrl_c_preserves_files(tmp_path):
    root, harness = make_harness(tmp_path)
    sentinel = root / "knowledge-data.fixture"
    sentinel.write_text("preserve me")
    port = free_port()
    first, log = start_harness(root, harness, port)
    try:
        wait_harness_ready(first, log)
        pids = [int((root / f"{name}.pid").read_text()) for name in ("api", "run", "build", "scheduler")]
        second, second_log = start_harness(root, harness, free_port(), "second")
        try:
            assert second.wait(timeout=5) == 1
            assert "本项目已在运行" in second_log.read_text()
            assert all(running(pid) for pid in pids)
        finally:
            if second.poll() is None:
                kill_fixture(second.pid)
                second.wait(timeout=5)
        first.send_signal(signal.SIGINT)
        assert first.wait(timeout=8) == 0, log.read_text()
        assert all(not running(pid) for pid in pids)
        assert sentinel.read_text() == "preserve me"
        restarted, restart_log = start_harness(root, harness, port, "restart")
        try:
            wait_harness_ready(restarted, restart_log)
            restarted.send_signal(signal.SIGTERM)
            assert restarted.wait(timeout=8) == 0
        finally:
            if restarted.poll() is None:
                restarted.send_signal(signal.SIGTERM)
                restarted.wait(timeout=8)
    finally:
        if first.poll() is None:
            first.send_signal(signal.SIGTERM)
            first.wait(timeout=8)


def test_main_failure_of_one_service_stops_owned_siblings(tmp_path):
    root, harness = make_harness(tmp_path)
    process, log = start_harness(root, harness, free_port())
    try:
        wait_harness_ready(process, log)
        pids = {
            name: int((root / f"{name}.pid").read_text()) for name in ("api", "run", "build", "scheduler")
        }
        os.kill(pids["scheduler"], signal.SIGTERM)
        assert process.wait(timeout=8) == 1, log.read_text()
        assert "scheduler 已退出" in log.read_text()
        assert all(not running(pid) for pid in pids.values())
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=8)


def test_main_sigterm_during_preparation_cleans_started_process(tmp_path):
    root, harness = make_harness(tmp_path, slow_prepare=True)
    process, log = start_harness(root, harness, free_port())
    try:
        pid = int(
            eventually(
                lambda: (
                    (root / "preparation.pid").read_text() if (root / "preparation.pid").exists() else None
                )
            )
        )
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 0, log.read_text()
        assert not running(pid)
        assert not (root / "api.pid").exists()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=8)


def test_main_occupied_port_leaves_existing_listener_untouched(tmp_path):
    root, harness = make_harness(tmp_path)
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen()
        process, log = start_harness(root, harness, existing.getsockname()[1])
        try:
            assert process.wait(timeout=5) == 1, log.read_text()
            assert "端口" in log.read_text()
            with socket.create_connection(existing.getsockname(), timeout=1):
                connection, _ = existing.accept()
                connection.close()
            assert not (root / "api.pid").exists()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=8)


def test_default_port_skips_occupied_listener(tmp_path):
    root, harness = make_harness(tmp_path)
    blocker = None
    try:
        # Keep an existing 8000 listener untouched, or occupy it with our own fixture.
        try:
            blocker = launcher.available_port(8000)
        except launcher.StartupError:
            pass
        expected = None
        for port in range(8001, 8011):
            try:
                probe = launcher.available_port(port)
            except launcher.StartupError:
                continue
            probe.close()
            expected = port
            break
        if expected is None:
            pytest.skip("All default ports are already owned by external processes")
        process, log = start_harness(root, harness, None)
        try:
            wait_harness_ready(process, log)
            assert f"已启动：http://127.0.0.1:{expected}" in log.read_text()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=8)
    finally:
        if blocker is not None:
            blocker.close()


def test_default_port_exhaustion_does_not_start_services(tmp_path):
    root, harness = make_harness(tmp_path)
    harness.write_text(
        harness.read_text().replace(
            "raise SystemExit(launcher.main())",
            "def occupied(port):\n    raise launcher.StartupError('occupied fixture')\n"
            "launcher.available_port = occupied\nraise SystemExit(launcher.main())",
        )
    )
    process, log = start_harness(root, harness, None)
    try:
        assert process.wait(timeout=5) == 1
        assert "8000–8010 均已占用" in log.read_text()
        assert not (root / "api.pid").exists()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=8)


@pytest.mark.parametrize("installed", [False, True])
def test_shell_entrypoint_handles_other_cwd_and_spaces(tmp_path, installed):
    root = tmp_path / "project with spaces"
    root.mkdir()
    script = root / "start.sh"
    script.write_text((launcher.ROOT / "start.sh").read_text())
    if installed:
        interpreter = root / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
        interpreter.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", str(script), "--port", "8123"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if installed:
        assert result.returncode == 0
        assert result.stdout.splitlines() == [
            str(root.resolve()),
            "-m",
            "xiaoying.launcher",
            "--port",
            "8123",
        ]
    else:
        assert result.returncode == 1
        assert "尚未安装 Python 环境" in result.stderr
