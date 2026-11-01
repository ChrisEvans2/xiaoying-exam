"""One local entry point for the built web UI, API and all queue workers (POSIX)."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from xiaoying.config import load_settings
from xiaoying.managed_process import REGISTRY_ENV, registered_groups
from xiaoying.storage.db import create_isolated_database, migrate

ROOT = Path(__file__).resolve().parents[2]


class StartupError(Exception):
    pass


def say(message):
    print(message, flush=True)


def available_port(port):
    sock = socket.socket()
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise StartupError(f"端口 {port} 已被占用；请停止原服务，或使用 ./start.sh --port 8001。") from None
    return sock


def compose_database(params):
    """Only the unmodified database provisioned by this repository is auto-started."""
    return (
        params.get("host") in {"127.0.0.1", "localhost"}
        and params.get("port") == "55432"
        and params.get("dbname") == "xiaoying"
        and params.get("user") == "xiaoying"
        and params.get("password") == "xiaoying"
        and not any(params.get(key) for key in ("hostaddr", "service"))
    )


def database_ready(url):
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.OperationalError:
        return False


def process_groups(roots):
    """Include descendants that start their own session, notably MinerU."""
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid="], text=True)
    rows = [tuple(map(int, line.split())) for line in output.splitlines() if line.strip()]
    owned = set(roots)
    while True:
        children = {pid for pid, parent, group in rows if parent in owned or group in roots}
        if children <= owned:
            break
        owned |= children
    return {group for pid, _, group in rows if pid in owned and group != os.getpgrp()}


class Processes:
    def __init__(self, directory, profile="live"):
        self.directory = directory
        self.profile = profile
        self.children = {}
        self._registry = tempfile.TemporaryDirectory(prefix="children-", dir=directory)
        self.registry = Path(self._registry.name)

    def start(self, name, command, pass_fds=()):
        log = self.directory / f"{name}.log"
        with log.open("ab") as stream:
            stream.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())
            stream.flush()
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=pass_fds,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "XIAOYING_PROFILE": self.profile,
                    REGISTRY_ENV: str(self.registry),
                },
            )
        self.children[name] = proc
        return proc

    def run(self, name, command, timeout):
        proc = self.start(name, command)
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                raise StartupError(f"{name} 超时，见 {self.directory / (name + '.log')}")
            time.sleep(0.1)
        if proc.returncode:
            raise StartupError(f"{name} 失败，见 {self.directory / (name + '.log')}")
        del self.children[name]

    def check(self):
        for name, proc in self.children.items():
            if proc.poll() is not None:
                raise StartupError(f"{name} 已退出，见 {self.directory / (name + '.log')}")

    def stop(self):
        if not self.children:
            self._registry.cleanup()
            return
        (self.registry / "closing").touch()
        roots = {p.pid for p in self.children.values()}
        # Snapshot before terminating parents; otherwise detached descendants get reparented.
        try:
            groups = process_groups(roots) | roots
        except (OSError, subprocess.SubprocessError, ValueError):
            groups = roots
        groups |= registered_groups(self.registry)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            groups |= registered_groups(self.registry)
            for group in groups:
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
            if sig == signal.SIGTERM:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if all(p.poll() is not None for p in self.children.values()):
                        break
                    time.sleep(0.1)
        for proc in self.children.values():
            proc.wait()
        self.children.clear()
        self._registry.cleanup()


def prepare_database(settings, processes):
    try:
        params = conninfo_to_dict(settings.database_url)
    except psycopg.Error:
        raise StartupError("DATABASE_URL 格式无效，请按 .env.example 检查数据库连接地址。") from None
    if getattr(settings, "provider_mode", None) == "mock" and not database_ready(settings.database_url):
        business = load_settings(profile="live")
        if not database_ready(business.database_url):
            prepare_database(business, processes)
        try:
            create_isolated_database(settings.database_url, business.database_url)
        except psycopg.Error:
            raise StartupError("mock 数据库创建失败，请检查 .env.mock 连接与创建数据库权限。") from None
    if not database_ready(settings.database_url):
        if not compose_database(params):
            raise StartupError("配置的 PostgreSQL 无法连接，请按 README 启动本机/远程数据库并检查 .env。")
        try:
            with socket.create_connection((params["host"], 55432), timeout=1):
                raise StartupError("数据库端口已有服务，但连接失败；请检查 .env 的账号、密码和库名。")
        except OSError:
            pass
        if not shutil.which("docker"):
            raise StartupError(
                "未找到 Docker。请按 README 本机方案安装并启动 PostgreSQL/pgvector，配置 .env。"
            )
        say("正在启动项目数据库…")
        processes.run(
            "database",
            ["docker", "compose", "-p", "xiaoying-exam", "up", "-d", "--wait", "--wait-timeout", "60", "db"],
            90,
        )
        if not database_ready(settings.database_url):
            raise StartupError("数据库仍无法连接，请检查 .env 和 .runtime/logs/database.log。")
    say("正在检查数据库结构…")
    try:
        migrate(
            make_conninfo(
                settings.database_url,
                connect_timeout=5,
                options="-c lock_timeout=5000 -c statement_timeout=30000",
            )
        )
    except psycopg.Error:
        raise StartupError(
            "数据库初始化失败，请检查 pgvector、建表权限和数据库是否正在执行其他迁移。"
        ) from None


def frontend_fingerprint():
    frontend = ROOT / "frontend"
    sources = sorted(
        p
        for p in frontend.rglob("*")
        if p.is_file()
        and not {"node_modules", "dist"}.intersection(p.relative_to(frontend).parts)
        and p.suffix != ".tsbuildinfo"
    )
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(frontend)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_frontend(processes):
    frontend = ROOT / "frontend"
    stamp = ROOT / ".runtime/frontend-build.sha256"
    fingerprint = frontend_fingerprint()
    if (frontend / "dist/index.html").is_file() and stamp.exists() and stamp.read_text() == fingerprint:
        return
    if not shutil.which("npm") or not shutil.which("node"):
        raise StartupError("网页需要编译，请先安装 Node.js 24（最低 22.13）及 npm，再运行 ./start.sh。")
    version = subprocess.check_output(["node", "--version"], text=True).strip().lstrip("v")
    numbers = tuple(int(part) for part in version.split(".")[:2])
    if numbers < (22, 13):
        raise StartupError("Node.js 版本过低，请安装 Node.js 24（最低 22.13）。")
    # npm ci is only needed when the lockfile changed or the installation is absent.
    dependency_stamp = ROOT / ".runtime/frontend-deps.sha256"
    lock_hash = hashlib.sha256((frontend / "package-lock.json").read_bytes()).hexdigest()
    if (
        not (frontend / "node_modules/.bin/vite").is_file()
        or not dependency_stamp.exists()
        or dependency_stamp.read_text() != lock_hash
    ):
        say("正在按锁文件安装网页依赖…")
        processes.run("frontend-install", ["npm", "--prefix", "frontend", "ci"], 600)
        dependency_stamp.write_text(lock_hash)
    say("正在编译网页…")
    processes.run("frontend-build", ["npm", "--prefix", "frontend", "run", "build"], 180)
    if not (frontend / "dist/index.html").is_file():
        raise StartupError("网页构建未生成入口文件，请检查 .runtime/logs/frontend-build.log。")
    stamp.write_text(fingerprint)


def wait_ready(processes, ready_files, url, timeout=45):
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        processes.check()
        workers_ready = all(path.exists() and path.read_text() == str(proc.pid) for path, proc in ready_files)
        if workers_ready:
            try:
                with opener.open(url + "/health", timeout=1) as response:
                    healthy = json.load(response).get("status") == "ok"
                if healthy:
                    processes.check()
                    return
            except (URLError, OSError, ValueError):
                pass
        time.sleep(0.2)
    raise StartupError("服务启动超时，请查看 .runtime/logs/ 下的 API 和 Worker 日志。")


def main():
    parser = argparse.ArgumentParser(description="一条命令启动网页、HTTP API 和全部 Worker")
    parser.add_argument("--port", type=int, help="指定端口；默认从 8000–8010 选择空闲端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--mock", action="store_true", help="显式启动隔离的离线演示环境（.env.mock）")
    args = parser.parse_args()
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    os.chdir(ROOT)
    profile = "mock" if args.mock else "live"
    runtime = ROOT / (".runtime/mock" if args.mock else ".runtime")
    logs = runtime / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = Processes(logs, profile)
    listener = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (runtime / "launcher.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StartupError("本项目已在运行，请使用原终端停止后再启动。") from None
            # Keep the lock until every owned process has been reaped.
            try:
                if args.port is None:
                    for port in range(8000, 8011):
                        try:
                            listener = available_port(port)
                        except StartupError:
                            continue
                        args.port = port
                        break
                    if listener is None:
                        raise StartupError("8000–8010 均已占用，请使用 ./start.sh --port 指定其他端口。")
                else:
                    listener = available_port(args.port)
                try:
                    settings = load_settings(profile=profile)
                except ValueError:
                    raise StartupError(
                        "配置无效：业务入口要求 .env 使用 live；mock 请用 --mock 和独立的 .env.mock 数据库/文件目录。"
                    ) from None
                if settings.provider_mode == "live" and not importlib.util.find_spec("sentence_transformers"):
                    raise StartupError("缺少 BGE 运行库，请先执行 uv sync --frozen --extra models。")
                prepare_frontend(processes)
                prepare_database(settings, processes)
                with tempfile.TemporaryDirectory(prefix="startup-", dir=runtime) as state:
                    processes.start(
                        "api",
                        [
                            sys.executable,
                            "-m",
                            "uvicorn",
                            "xiaoying.api:app",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(args.port),
                            "--fd",
                            str(listener.fileno()),
                        ],
                        pass_fds=(listener.fileno(),),
                    )
                    listener.close()
                    listener = None
                    ready_files = []
                    for kind in ("run", "build", "scheduler"):
                        ready = Path(state) / kind
                        proc = processes.start(
                            kind,
                            [
                                sys.executable,
                                "-m",
                                "xiaoying.worker",
                                "--kind",
                                kind,
                                "--concurrency",
                                "2" if kind == "run" else "1",
                                "--ready-file",
                                str(ready),
                            ],
                        )
                        ready_files.append((ready, proc))
                    url = f"http://127.0.0.1:{args.port}"
                    wait_ready(processes, ready_files, url)
                    say(f"\n已启动：{url}  ·  模式：{settings.provider_mode}")
                    say(
                        "在网页中新建知识库、上传/替换文档、重新分块和问答。Ctrl-C 停止应用；数据库与数据保留。"
                    )
                    if settings.provider_mode == "live" and not settings.deepseek_api_key:
                        say("提示：尚未填写 DEEPSEEK_API_KEY，可先管理文档；真实问答需填写 .env 后重启。")
                    say(f"日志：{logs.relative_to(ROOT)}/")
                    if not args.no_browser:
                        webbrowser.open(url)
                    while True:
                        processes.check()
                        time.sleep(0.5)
            finally:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                processes.stop()
    except KeyboardInterrupt:
        say("\n应用已停止，知识库及数据库数据已保留。")
        return 0
    except (StartupError, OSError) as exc:
        say(f"启动失败：{exc}")
        return 1
    finally:
        # Also cover errors before the lock is acquired.
        processes.stop()
        if listener is not None:
            listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
