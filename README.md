# 企业知识问答 Agent

上传企业文档，通过网页或 HTTP API 提问，获得带原文引用的回答，并查看检索、读取和追问的 Trace。支持 Markdown、Word（`.docx`）、HTML、PDF、TXT，以及文档替换和知识库版本管理。

主要入口是 **HTTP API + 网页**，CLI 问答为可选工具。后端使用 Python、FastAPI、LangGraph、PostgreSQL/pgvector；前端使用 React/TypeScript。DeepSeek 负责问题理解与决策，BGE 负责向量和重排，MinerU 负责 PDF 解析。

[环境](#环境) · [安装](#安装) · [运行](#运行) · [评审入口](#评审入口)

完成首次安装后，日常只需运行 **`./start.sh`**。打开网页即可新建知识库、上传或替换文档、修改分块参数并重新构建，然后开始问答。

## 环境

### 软件版本与用途

| 软件/服务 | 版本要求或推荐版本 | 何时需要 |
| --- | --- | --- |
| 操作系统 | macOS、Linux；Windows 使用 WSL2 + Ubuntu | 以下命令适用于 Bash/Zsh；原生 Windows 的 PDF 子进程管理尚未适配 |
| Python | **推荐 3.12**；项目声明支持 `>=3.12,<3.14` | 后端、Worker、文档处理；由 uv 安装即可 |
| uv | 推荐 **0.12.3 或更新版本** | 安装 Python、创建虚拟环境、按 `uv.lock` 安装依赖 |
| Node.js | **推荐 24.x**；当前前端依赖最低要求 **22.13.0** | 编译/开发网页；只使用 HTTP API 时可不安装 |
| npm | 使用 Node.js 随附版本；已验证 npm 11.x | 按 `frontend/package-lock.json` 安装前端依赖 |
| PostgreSQL | **16.x** | 必需的数据服务，可容器运行或本机安装 |
| pgvector | **0.8.x**，必须安装到实际运行的 PostgreSQL 版本 | 数据库扩展；Python 的 `pgvector` 包不能代替它 |
| Docker + Compose | **可选**；使用 `docker compose` 插件命令，支持 Compose v2/v5 | 仅用于运行数据库；没有 Docker 时使用下文的本机安装步骤 |
| DeepSeek API | API Key；默认模型 `deepseek-v4-flash`，可配置 | 真实问答模式；无须下载 DeepSeek 权重 |
| BGE-M3 + bge-reranker-v2-m3 | `.env.example` 中已固定模型 commit SHA | 真实模式需要下载两个模型的权重 |
| MinerU | **2.7.6，pipeline 后端** | 真实模式处理 PDF 时需要软件包及配套模型 |

已验证的开发环境：Python 3.12.14、Node.js 24.19、npm 11.x、uv 0.12.3、Docker 29.6.2 / Compose 5.3.1、PostgreSQL 16。Node.js 下限来自当前锁定的 Vite/PDF.js 依赖；不要用 Node 18 或较早的 Node 22 编译。

Docker 内只有数据库；API、Worker 和模型默认在本机运行。BGE 和 MinerU 可以使用 CPU，GPU 不是启动条件，但真实模型加载和 PDF 解析需要额外内存、磁盘与时间；本项目尚无硬件性能基准。NVIDIA 环境的 PyTorch/CUDA 安装应按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 配置。

### 两种运行模式

| 模式 | 需要下载/配置什么 | 能验证什么 |
| --- | --- | --- |
| `mock` 离线演示 | Python 依赖、数据库；网页另装 Node 依赖 | 上传、版本、检索/读取流程、Trace；**不证明真实模型回答效果** |
| `live` 真实问答 | 以上依赖 + DeepSeek Key + 两个 BGE 模型；PDF 再安装 MinerU 和模型 | 使用真实模型执行问答；首次运行前先下载并检查模型 |

只处理 MD/Word/HTML/TXT 时可暂不安装 MinerU；示例资料中含 PDF，导入全部 8 份资料需要准备它，或使用明确标记的 mock 模式。

## 安装

所有命令均在**仓库根目录**执行，除非另有说明；已有 `.env`、数据库或模型缓存时应复用，避免覆盖。

### 让 AI Agent 协助部署

在打开本仓库的 Claude Code、Codex 或其他具有终端权限的 AI Agent 中，复制这一句：

> 请按本仓库 README 的“环境、安装、运行”完成本机部署：先检查并复用已有软件与配置，安装 Python 3.12、uv 和网页所需的 Node.js 24，已有可用 Docker 时用 Compose 启动数据库，否则直接在本机安装 PostgreSQL 16 与对应 pgvector，保留现有 .env 和数据，按锁文件安装依赖，有 DeepSeek 密钥时配置 live 并下载固定修订的 BGE 与 MinerU 2.7.6 pipeline 模型、没有密钥时先启动明确标记的 mock 模式，使用 ./start.sh 启动整个应用，验证网页、/health 和全部 Worker 就绪，将新建知识库、上传/替换文档和重新分块留给我在网页操作，并报告实际版本、启动地址、模型路径和未通过项，密钥由我在本机填写且不要输出到日志。

这句话用于首次环境部署，需要有终端执行能力的 Agent。日常启动使用仓库自带的 `./start.sh`；它会复用已经安装的 Python 环境和模型。

### 1. 安装基础工具与项目依赖

- uv：按 [官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装；macOS 已有 Homebrew 时可运行 `brew install uv`。
- Node.js：需要网页时安装 [Node.js 24](https://nodejs.org/en/download)，包含 npm；只使用 API 可跳过。
- Docker：已有可用环境就复用；没有则直接使用第 3 节本机数据库方案，无须为了本项目专门安装 Docker。

```bash
uv --version
uv python install 3.12
uv sync --frozen --python 3.12

# 需要网页时执行
node --version
npm --version
npm --prefix frontend ci
```

`uv sync` 创建 `.venv`，安装 FastAPI、LangGraph、psycopg、pgvector Python 客户端、文档解析器等，具体版本以 [uv.lock](uv.lock) 为准。前端依赖以 [package-lock.json](frontend/package-lock.json) 为准。两个命令都**不会下载 BGE 或 MinerU 模型权重**。

后文使用 `.venv/bin/python` / `.venv/bin/xiaoying` 运行已安装环境，避免后续不带 `--extra models` 的 `uv run` 自动移除真实模式的可选依赖。

### 2. 配置 `.env`

首次使用时复制模板；已存在则直接编辑：

```bash
if [ ! -f .env ]; then
  cp .env.example .env
fi
```

`.env` 只用于真实业务环境，保持以下配置：

```dotenv
# 真实问答
PROVIDER_MODE=live
PDF_BACKEND=mineru
DEEPSEEK_API_KEY=填写你自己的密钥
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING=true
DEEPSEEK_REASONING_EFFORT=high
DEEPSEEK_MAX_TOKENS=16384
```

`./start.sh` 和 `./start.sh --port 8001` 默认使用 live，端口不影响模式。若 `.env` 或终端仍配置 `PROVIDER_MODE=mock`，业务入口会报配置错误，不会静默进入 mock；请将 `.env` 改回 live，并清除终端中遗留的 `PROVIDER_MODE` / `PDF_BACKEND`。

离线演示必须显式运行 `./start.sh --mock`。它只读取独立的 `.env.mock` 和 `XIAOYING_MOCK_` 前缀变量，不继承业务 `.env`、数据库连接或 API 密钥。默认值可直接使用，需要自定义时复制 `.env.mock.example` 为 `.env.mock`。API 和各 Worker 继承启动器选定的同一环境。

| 环境 | 数据库（默认，同一 PostgreSQL 服务） | 文件与日志 | 入口 |
| --- | --- | --- | --- |
| 真实业务 | `xiaoying` | `.runtime/files`、`.runtime/logs` | `./start.sh` |
| mock 演示 | `xiaoying_mock` | `.runtime/mock/files`、`.runtime/mock/logs` | `./start.sh --mock` |
| 自动测试 | `xiaoying_test` | pytest 的临时目录 | `.venv/bin/python -m pytest -q` |

mock/测试数据库首次使用时会创建，现有数据库和业务数据保留；账号需要创建数据库权限，或请管理员提前创建独立数据库。mock、测试的数据库名必须与业务库不同，mock 文件目录也不能与业务目录重叠。自动测试连接可通过 `XIAOYING_TEST_DATABASE_URL` 指定独立数据库；不允许指回业务库或 mock 库。`.env` 和 `.env.mock` 都被 Git 忽略。

问题扩展和 Agent 决策默认开启思考模式。`DEEPSEEK_REASONING_EFFORT` 可设 `low/high/max`，`DEEPSEEK_MAX_TOKENS` 是每次响应的思考与最终输出合计上限；原有单次 45 秒、任务累计 180 秒预算仍生效。修改配置后重启 `./start.sh` 并新建问题；已有任务使用配置快照，不在恢复途中切换模式。前端 Trace 显示工具动作和结果，推理字段仅在后端保存以满足多轮工具调用协议。协议与恢复说明见 [设计](docs/02-solution.md)。

### 3. 准备数据库：选择一种方式

**A. 已有 Docker：使用 Compose**

```bash
docker --version
docker compose version
docker info
docker compose -p xiaoying-exam up -d db
docker compose -p xiaoying-exam ps
```

等待 `db` 显示 healthy；如 Docker 已安装但未启动，先启动 Docker 服务。`.env` 使用模板中的本地开发连接：

```dotenv
DATABASE_URL=postgresql://xiaoying:xiaoying@127.0.0.1:55432/xiaoying
```

容器将本机 `55432` 映射至 PostgreSQL 的 `5432`，数据位于 Docker 命名卷。示例账号仅用于本机开发，服务只绑定回环地址。

**B. 没有 Docker：直接在本机安装 PostgreSQL 16 + pgvector**

不使用容器，应用代码不需要修改。下面步骤用于首次安装；已有 PostgreSQL 时，先确认端口与 `pg_config` 对应的主版本，再复用现有实例。角色/数据库已存在时跳过创建，扩展仍须检查。

<details>
<summary>macOS：Homebrew 安装 PostgreSQL 16，并为它编译 pgvector</summary>

需要 Homebrew、Git 和 Xcode Command Line Tools；缺编译工具时执行 `xcode-select --install`。安装并启动 PostgreSQL：

```bash
brew install postgresql@16
export PATH="$(brew --prefix postgresql@16)/bin:$PATH"
brew services start postgresql@16
psql --version
pg_isready -h 127.0.0.1 -p 5432
```

当前 Homebrew 的 `pgvector` 公式面向 PostgreSQL 17/18，因此 PostgreSQL 16 使用指定 `PG_CONFIG` 的源码安装，确保扩展放到正确目录：

```bash
mkdir -p .runtime
git clone --branch v0.8.7 --depth 1 https://github.com/pgvector/pgvector.git .runtime/pgvector-src
make -C .runtime/pgvector-src PG_CONFIG="$(brew --prefix postgresql@16)/bin/pg_config"
make -C .runtime/pgvector-src PG_CONFIG="$(brew --prefix postgresql@16)/bin/pg_config" install

# 首次创建时输入并记住自己的本地数据库密码
createuser --login --pwprompt xiaoying
createdb --owner=xiaoying xiaoying
# 用本机安装时的管理员身份启用扩展
psql -d xiaoying -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

重复安装已有源码目录时跳过 `git clone`，不要重新初始化已有数据库。参考：[PostgreSQL 16 Homebrew 公式](https://formulae.brew.sh/formula/postgresql@16)、[pgvector 安装与 PG_CONFIG](https://github.com/pgvector/pgvector#installation)。

</details>

<details>
<summary>Ubuntu 22.04/24.04 或 WSL2 Ubuntu：使用 PostgreSQL 官方 APT 仓库</summary>

```bash
sudo apt update
sudo apt install -y postgresql-common ca-certificates
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
sudo apt update
sudo apt install -y postgresql-16 postgresql-client-16 postgresql-16-pgvector
sudo systemctl enable --now postgresql

# 首次创建时输入并记住自己的本地数据库密码
sudo -u postgres createuser --login --pwprompt xiaoying
sudo -u postgres createdb --owner=xiaoying xiaoying
sudo -u postgres psql -d xiaoying -c 'CREATE EXTENSION IF NOT EXISTS vector;'
pg_isready -h 127.0.0.1 -p 5432
```

WSL 未启用 systemd 时，将启动命令换成 `sudo service postgresql start`。已有多个集群时用 `pg_lsclusters` 确认 PostgreSQL 16 实际端口，连接地址使用该端口。

参考：[PostgreSQL Ubuntu 安装](https://www.postgresql.org/download/linux/ubuntu/)、[pgvector APT 安装](https://github.com/pgvector/pgvector#apt)。

</details>

本机方案在 `.env` 中改成实际地址，**通常是 5432，不是 Docker 的 55432**：

```dotenv
DATABASE_URL=postgresql://xiaoying:替换为你的密码@127.0.0.1:5432/xiaoying
```

密码含 `@`、`:`、`/` 等 URI 特殊字符时须进行 URL 编码。也可连接已有的远程 PostgreSQL，但数据库管理员须先安装并启用 `vector` 扩展，并提供可以在该数据库创建应用表的账号。

### 4. 下载真实模式的模型与 PDF 依赖

**mock 模式跳过本节。** live 模式需要 BGE；只有处理 PDF 时才需要 MinerU。Python 软件包、模型权重和 DeepSeek API 密钥是三项不同的准备工作。

**BGE：安装运行库，下载并检查两个固定修订的模型。**

```bash
uv sync --frozen --python 3.12 --extra models
.venv/bin/python - <<'PY'
from xiaoying.config import Settings
from xiaoying.providers.embeddings import providers
settings = Settings()
assert settings.provider_mode == "live", "先在 .env 切换到 live"
embedding, reranker = providers(settings)
vector = embedding.encode(["模型下载与检查"])[0]
assert len(vector) == 1024
print("BGE-M3 就绪，向量维度：", len(vector))
print("reranker 检查：", reranker.rank("住宿报销", ["住宿报销需要符合城市标准"]))
PY
```

该命令通过 `sentence-transformers` 下载模型和 tokenizer，并进行一次本地推理，不调用 DeepSeek。缓存默认位于当前用户的 `~/.cache/huggingface`；后续同一用户的 Worker 复用缓存，但每个进程仍需加载模型。保留 `.env.example` 中的固定修订，不改成 `main`。模型来源：[BGE-M3](https://huggingface.co/BAAI/bge-m3)、[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。

**MinerU：在独立 Python 环境安装 pipeline 依赖，再下载 PDF 解析模型。**

```bash
uv venv --python 3.12 .runtime/mineru
uv pip install --python .runtime/mineru/bin/python 'mineru[pipeline]==2.7.6'
.runtime/mineru/bin/mineru-models-download --source huggingface --model_type pipeline
```

Hugging Face 连接不可用时，下载命令可改为 `--source modelscope`。下载工具会记录模型目录到当前用户的 `~/mineru.json`；启动 Worker 时使用同一系统用户。这里安装 `pipeline` extra，包含本项目 PDF 后端所需依赖；无需下载 VLM 模型，也无需安装 MinerU WebUI。不要只装裸 `mineru` 包就认为 OCR 依赖和模型齐全。版本与命令已对照 [MinerU 2.7.6 发布包](https://pypi.org/project/mineru/2.7.6/) 核对；本项目不直接兼容 MinerU 4 的新 CLI/输出格式。

下载后先单独检查示例 PDF：

```bash
.runtime/mineru/bin/mineru -p data/policies/D02-finance-intake.pdf -o .runtime/mineru-check -b pipeline --source local
```

应在输出目录中找到 `*_content_list.json`。没有 GPU 或自动选择设备不适用时，可追加 `-d cpu`；首次检查耗时较长，先完成检查再提交受超时预算限制的入库任务。

最后生成可复制的配置行，将输出替换到 `.env` 的 `MINERU_COMMAND`：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
command = [str(Path(".runtime/mineru/bin/mineru").resolve()), "-p", "{input}", "-o", "{output}", "-b", "pipeline", "--source", "local"]
print("MINERU_COMMAND=" + json.dumps(command, ensure_ascii=False))
PY
```

如刚才使用了 `-d cpu`，也把 `"-d","cpu"` 加入 JSON 数组。独立环境避免 MinerU 的 Torch/Transformers 依赖与问答后端互相影响；`--source local` 使用已经下载好的模型。

### 5. 完成首次准备

确认 `.env` 已保存、数据库已安装、需要的模型已下载，即可进入下一节。`./start.sh` 会自动执行数据库迁移并编译网页，不需要手动初始化或运行导入脚本。

- **Docker 数据库**：使用本仓库默认连接时，启动器发现数据库未运行会自动执行 Compose；Docker 服务需已启动。
- **本机或远程 PostgreSQL**：启动器直接复用 `.env` 中的连接；请先按上文安装并启动数据库服务、启用 pgvector。没有 Docker 也可以运行整个 Agent。
- **真实问答**：在 `.env` 填好 `DEEPSEEK_API_KEY`。未填写时仍可进入网页管理文档，终端会提示问答配置尚未完成。

<details>
<summary>手动迁移与编译（仅排查问题时使用）</summary>

```bash
.venv/bin/xiaoying init-db
npm --prefix frontend run build
```

本机数据库须先由管理员完成 `CREATE EXTENSION vector`；应用迁移创建业务表和 LangGraph checkpoint 表，可重复执行。

</details>

## 运行

### 一条命令启动整个 Agent

在仓库根目录执行：

```bash
./start.sh
```

就绪后自动打开网页，默认地址为 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**；若端口被占用，会自动尝试 8001–8010，**以终端显示和自动打开的地址为准**。终端出现“已启动”即表示网页/API 和三个 Worker 均已就绪。

启动器会读取 `.env`，检查或启动上述默认 Docker 数据库、执行迁移、按锁文件准备网页依赖并编译网页，然后统一启动：

| 组件 | 用途 |
| --- | --- |
| 网页 + HTTP API | 同一个地址访问页面、文档管理、对话和 Trace |
| 问答 Worker | 执行 Agent，问答全局并发最多 2 个 |
| 构建 Worker | 文档解析、分块和向量化，构建全局并发最多 1 个 |
| 调度 Worker | 处理超时和失效租约恢复 |

网页源码未变且构建产物完整时会复用已有编译结果。启动器不导入示例资料，知识库内容由你在网页管理。

**停止：在启动终端按 Ctrl-C**，统一停止这次启动的应用进程；数据库和知识库数据保留。再次使用仍是 `./start.sh`。日志在 `.runtime/logs/`；如果某个服务异常退出，启动器会停止其余应用进程并指出对应日志。

### 在网页管理知识库

| 你要做的事 | 网页操作 |
| --- | --- |
| 新建知识库 | 左侧展开“新建知识库”，输入名称并创建；进入“文档与版本”，拖入文件，点击“构建并发布新版本” |
| 重新分块 | 选择已有知识库，在“文档与版本 → 元数据与分块设置”修改参数，直接点击“构建并发布新版本”；无需重新上传原文件 |
| 替换旧文档 | 上传新版文件，使用与旧文档相同的 `document_id`；确认待构建变更显示“替换”，再点击“构建并发布新版本” |
| 开始问答 | 构建完成后进入“知识问答”，发送问题；右侧查看 Trace，点击引用查看原文 |

可将 `data/policies` 中的 8 份合成制度拖入网页体验。构建在后台持续执行，可以离开页面、切换知识库或刷新，返回后自动恢复任务与进度；当前任务完成后再进行下一批上传或配置修改。未提交的上传清单按知识库保存在当前浏览器，关闭或刷新页面后仍可继续。

“新增/替换”按 **`document_id`** 判断：当前知识库没有该 ID 时新增，已有该 ID 时替换，文件名相同并不代表替换。只有新增或变更的文档需要解析和计算向量；配置一致时，未变文档复用已有原文块、分块与向量。相同内容、格式和元数据的重复上传也可复用原修订；分块、模型或解析配置变化时重新构建。页面显示“复用 / 新构建”数量。

变更文件产生新的文档修订，并发布新的知识库版本；旧文件修订和历史版本继续保留。已经开始的问题仍使用它开始时绑定的版本。旧版本曾在业务库内构建的 mock 索引需要在此页面重新构建为 live；新创建的 mock 数据只进入隔离的 mock 库。索引与当前模型不兼容时，提问接口会直接返回 409 和重建提示，不创建问答任务或调用模型。

<details>
<summary>文档元数据与分块参数说明</summary>

文件缺少内置元数据时，在“元数据与分块设置”填写，例如：

```json
{"document_id":"POLICY-001","title":"差旅制度","version_date":"YYYY-MM-DD","scope":"中国大陆员工境内差旅"}
```

将 `YYYY-MM-DD` 替换为实际版本日期。替换时填写旧文档的 ID；如果文件本身包含元数据，应保持文件和填写值一致。不同文档使用不同 ID，批量上传时不要给不同文件填同一 ID。

语义分块依据句段向量相似度和结构边界，默认目标 400 / 最大 800 / 重叠 80 tokens，语义相似度阈值 0.62。重新分块会基于当前已发布版本中的文档创建新版本，具体策略见 [设计](docs/02-solution.md)。

</details>

<details>
<summary>调试参数与分进程启动（可选）</summary>

```bash
# 希望固定使用某个端口时（该端口被占用则报错）
./start.sh --port 8001
# 远程终端或不希望自动打开浏览器时
./start.sh --no-browser
# 隔离的离线演示；可与业务服务使用不同端口同时运行
./start.sh --mock --port 8002
```

需要单独调试某个进程时，先停止一键启动的应用，再分别运行下列命令：

| 终端 | 命令 |
| --- | --- |
| 1 | `.venv/bin/python -m uvicorn xiaoying.api:app --host 127.0.0.1 --port 8000` |
| 2 | `.venv/bin/python -m xiaoying.worker --kind run --concurrency 2` |
| 3 | `.venv/bin/python -m xiaoying.worker --kind build` |
| 4 | `.venv/bin/python -m xiaoying.worker --kind scheduler` |

- [Swagger / HTTP API 文档](http://127.0.0.1:8000/docs)
- [OpenAPI JSON](http://127.0.0.1:8000/openapi.json)
- 健康检查：`curl --fail http://127.0.0.1:8000/health`

健康检查返回 `status: ok` 和当前模式；问答业务效果仍需使用真实资料和问题验证。

</details>

### HTTP API 对接示例

以下基于一个已经发布的知识库；新建、上传、构建的完整字段可在 `/docs` 交互调用。每个**新的写请求**使用新的 `Idempotency-Key`，同一请求网络重试复用原键。

```bash
# 1. 创建会话，将 kb_id 替换为你的知识库 ID
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/conversations \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: example-conversation-001' \
  -d '{"kb_id":"替换为知识库ID"}'

# 2. 提交问题，将路径中的会话 ID 换成上一步返回的 id
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/conversations/替换为会话ID/runs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: example-question-001' \
  -d '{"question":"在上海出差，住宿费每晚最多能报多少？"}'

# 3. 用返回的 run_id 读取事件和结果；SSE 在终态后关闭
curl -N http://127.0.0.1:8000/api/v1/runs/替换为run_id/events
curl --fail http://127.0.0.1:8000/api/v1/runs/替换为run_id
```

提交问题返回202及 `run_id`、`job_id`、`events_url`。结果中的 `task_status` 为 `answered`、`awaiting_user`、`insufficient` 或 `incomplete`；尚在排队/运行时可为空。断线续读事件可发送 `Last-Event-ID`，补充通过 `POST /api/v1/runs/{run_id}/clarification` 提交。

### CLI 可选工具、开发与停止

以下是便捷工具，不是网页/API 使用前提：

```bash
# 按 .env 模式批量导入示例资料，返回 kb_id
.venv/bin/xiaoying seed
# 可选的单次 CLI 问答
.venv/bin/xiaoying ask --kb-id <kb_id> '上海住宿上限是多少？'
# 导出一次运行及 Trace
.venv/bin/xiaoying export <run_id> --output examples/my-run.json
```

仅想离线验证 CLI 时，显式加 `--mock`：先执行 `.venv/bin/xiaoying --mock init-db`，再执行 `.venv/bin/xiaoying --mock seed`。这些命令使用独立的 `xiaoying_mock` 数据库和 `.runtime/mock/files`，不修改业务知识库；查询也必须使用 `--mock`。离线回答会标注 `incomplete / MOCK_ONLY`。

前端热更新开发可在 `./start.sh` 运行期间另开终端执行 `npm --prefix frontend run dev`，将 API 转发到默认 8000 端口。停止整个应用按启动终端的 Ctrl-C。需要另外停止 Docker 数据库时用 `docker compose -p xiaoying-exam stop db`；本机数据库按自己的服务管理方式停止。原文件在 `.runtime/files`，Docker 数据在命名卷，本机数据库数据在其本机数据目录；这些位置与模型缓存共同构成需要保留的数据。

### 常见问题

| 现象 | 检查方向 |
| --- | --- |
| Node 版本错误 / 网页构建失败 | 使用 Node 24 或至少22.13，重新执行 `npm ci` |
| 数据库连接失败 | Docker 服务是否启动；本机端口通常5432，Compose端口55432；检查 `.env` |
| `extension "vector" is not available` | pgvector 是否安装到当前服务器主版本；不是仅安装了 Python 包 |
| 创建扩展权限不足 | 请管理员在该数据库启用扩展，再由数据库拥有者执行迁移 |
| 网页未就绪 / 编译失败 | 查看 `.runtime/logs/frontend-build.log`；`./start.sh` 会自动编译，需满足 Node 版本要求 |
| 重复启动 / 端口被占用 | 回到原启动终端按 Ctrl-C；其他程序占用端口时使用 `./start.sh --port 8001`，无需结束其他程序 |
| 问答/构建一直排队 | 使用 `./start.sh` 启动全部 Worker；查看 `.runtime/logs/run.log`、`build.log` 和 `scheduler.log` |
| `MODELS_NOT_INSTALLED` | 安装 `uv sync --frozen --extra models`，随后用 `.venv/bin/python` 启动 |
| `MODEL_NOT_CONFIGURED` | 在本机 `.env` 填写 DeepSeek Key 后重启 API/Worker |
| `MINERU_UNAVAILABLE` / PDF解析失败 | 检查命令路径、2.7.6 pipeline 依赖、本地模型及 `MINERU_TOOLS_CONFIG_JSON` 指定的配置（默认 `~/mineru.json`）；日志在 `.runtime/files/parsed/` |
| `INDEX_MODEL_MISMATCH` | 查询模型与知识库模型不同；不要混用 mock/live 索引，重新构建 |

## 评审入口

### 用时记录

业务分析约30分钟，文档数据与问题测试用例生成约35分钟，模块化设计约45分钟，V1开发约40分钟，V1 Bug修复约1小时，测试验收约20分钟，合计约 **3小时50分钟**。详见 [分阶段用时](docs/04-ai-usage.md#用时记录)。

### 当前验证状态

完整工程回归 **241 passed，0 failures、0 errors、0 skipped**，报告为 [test-results.xml](examples/test-results.xml)。测试使用独立数据库、真实 PostgreSQL/pgvector 与 LangGraph，模型使用测试替身；覆盖版本、竞态、checkpoint、引用、超时、SSE、启动与环境隔离，以及会话审计、需求分流、预算收尾、增量构建和验收环境保护。工程测试不计作真实模型回答正确率。前端此前已通过 TypeScript/Vite 构建；本次验收未修改前端。

本机已验证 live 与 mock 服务同时运行时使用不同数据库，完整回归及脚本演示前后业务库 22 张表的数据指纹一致。原有“青屿差旅制度”已基于保留的 8 份文档重建并发布真实 BGE 索引（66 个 chunk，1024 维）；旧版本继续保留。旧 mock 索引的提问会直接返回 409，不创建问题、消息或调用模型。

真实 DeepSeek + BGE + MinerU 链路已在独立环境完成 **9 条原始用例及 1 条500→600文档替换变体**，覆盖明确答案、冲突、资料外、相似问法、上下文追问和工具失败；选定集的事实、引用和动作核验 **10/10通过**。见 [结果汇总](examples/live/exam-20261102-final/summary.json)、[逐条审阅](examples/live/exam-20261102-final/review.json) 和 [评估说明](docs/03-evaluation.md)。7 条已回答、2 条资料不足、1 条执行未完成；异常处理通过不代表成功回答。原20题仍有11题、长篇40题均未执行，也没有独立盲测正确率。首轮发现的两处错误及原输出一并保留。

本次资料冻结自原20题对应的Git快照，八份原文件和哈希随记录保存；工作区D07的并发修订被保留，未将该修订算作已验收。此前的两轮 [思考模式冒烟](examples/thinking-smoke.json) 继续保留为协议证据。

本项目当前是本地单工作区应用，没有登录、租户授权或公网部署配置。上传限制20 MiB、PDF最多200页、每个知识库版本最多100份文档；旧 `.doc` 需先转换。纠错记忆、知识图谱构建保留 TODO。

### 可复现验证

自动测试、mock 演示和业务 Worker 使用不同数据库，可以保持业务服务运行。mock 脚本与 mock 网页共用 mock 库，执行脚本夹具前请停止 mock 网页的 Worker，避免竞争同一演示任务：

```bash
.venv/bin/python -m pytest -q --junitxml=examples/test-results.xml
.venv/bin/python -m ruff check src scripts tests --select F,E9,B023
npm --prefix frontend run build

# 明确标记的脚本夹具：在 xiaoying_mock 内执行 PG/LangGraph
.venv/bin/python scripts/demo_cases.py

# 配置 live、模型和原始8份资料索引后，运行真实模型评估
.venv/bin/python scripts/evaluate.py --kb-id <live_kb_id> --case all

# 一键创建独立环境，复现考试六类场景及文档替换；使用新输出目录
.venv/bin/python scripts/exam_acceptance.py \
  --output examples/live/exam-rerun \
  --source-revision a3e478af065c3542c80fc199750d8bef309fe3c0
```

真实评估检查文档SHA与发布版本，不把参考答案提供给模型；自动比较状态，事实和引用仍须另行审阅。本次审阅由AI对照原文完成，未冒称考生独立人工验收。`exam_acceptance.py` 自动创建独立数据库、冻结资料并导出结果，源码变化时拒绝混用旧记录恢复；不会自动宣称语义通过。脚本夹具覆盖正常、冲突、工具恢复、持续失败、500→600替换，**不能充当真实模型决策证据**。

### 代码与材料

| 路径 | 内容 |
| --- | --- |
| `start.sh`、`src/xiaoying/launcher.py` | 统一启动、就绪检查、日志和应用进程收尾 |
| `frontend/src` | 选库、对话、上传/版本、SSE Trace、引用和PDF预览 |
| `src/xiaoying/api.py`、`services` | HTTP契约、幂等、版本发布、补充生命周期与引用门禁 |
| `src/xiaoying/agent`、`providers` | DeepSeek/LangGraph、BGE及显式测试适配器 |
| `src/xiaoying/ingestion`、`rag` | 多格式解析与语义分块；关键词/向量混合检索、RRF、rerank |
| `src/xiaoying/storage`、`worker.py`、`migrations` | PostgreSQL、checkpoint、持久化队列、租约与恢复 |
| `scripts`、`tests`、`examples` | 入库/评估脚本、可执行测试、实际运行记录 |

LangGraph 1.2.19 负责状态图与 interrupt，checkpoint-postgres 3.2.0 负责持久化，外层添加租约事务门禁；后者使用版本锁定的 `_cursor` 扩展点，升级依赖时须重跑恢复与并发测试。默认180秒活跃预算、600秒等待补充、12次模型调用、8次检索/读取调用，恢复不清零预算；完整设计见下列文档。

- [业务需求](docs/01-requirements.md) · [方案设计](docs/02-solution.md)
- [评估与实际结果](docs/03-evaluation.md) · [独立 review](docs/review-notes.md)
- [AI 使用记录](docs/04-ai-usage.md) · [开发复盘](docs/05-retrospective.md)
- [脚本运行记录](examples/README.md) · [测试报告](examples/test-results.xml)
