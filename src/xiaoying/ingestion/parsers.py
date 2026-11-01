import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import zipfile
from datetime import date
from functools import wraps
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from markdown_it import MarkdownIt
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from lxml.etree import XMLSyntaxError
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.exceptions import InvalidXmlError

from xiaoying.domain import AppError, Block, DocumentIR
from xiaoying.managed_process import REGISTRY_ENV

FORMATS = {".md", ".docx", ".html", ".pdf", ".txt"}
CLAUSE = re.compile(r"^\s*(?:\d+[.、．)]|第[一二三四五六七八九十百\d]+条)")
META = {
    "文档 ID": "document_id",
    "文档ID": "document_id",
    "版本日期": "version_date",
    "公司": "company",
    "适用范围": "scope",
    "合成材料": "synthetic",
}


def decode(data):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise AppError("ENCODING", "无法识别文本编码", 422)


def document_errors(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AppError:
            raise
        except (
            PyPdfError,
            XMLSyntaxError,
            PackageNotFoundError,
            InvalidXmlError,
            zipfile.BadZipFile,
            KeyError,
            ValueError,
            TypeError,
            IndexError,
        ) as exc:
            raise AppError("INVALID_DOCUMENT", "文件损坏或文档结构不受支持，请检查后重新上传", 422) from exc

    return wrapped


@document_errors
def validate_file(path, settings):
    ext = path.suffix.lower()
    if ext not in FORMATS:
        raise AppError("UNSUPPORTED_FORMAT", "支持 md、docx、html、pdf、txt；旧 doc 请先转换", 415)
    size = path.stat().st_size
    if size == 0:
        raise AppError("EMPTY_FILE", "文件为空", 422)
    if size > settings.max_upload_bytes:
        raise AppError("FILE_TOO_LARGE", "文件超过上传大小限制", 413)
    if ext == ".pdf":
        if not path.read_bytes().startswith(b"%PDF-"):
            raise AppError("INVALID_FORMAT", "PDF 文件签名不正确", 415)
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise AppError("ENCRYPTED_PDF", "请先解密 PDF", 422)
        if len(reader.pages) > settings.max_pdf_pages:
            raise AppError("TOO_MANY_PAGES", "PDF 页数超过限制", 413)
    elif ext == ".docx":
        try:
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" not in z.namelist():
                    raise ValueError("not docx")
                if sum(i.file_size for i in z.infolist()) > settings.max_upload_bytes * 20:
                    raise AppError("EXPANDED_FILE_TOO_LARGE", "Word 解压大小超限", 413)
        except (zipfile.BadZipFile, ValueError):
            raise AppError("INVALID_FORMAT", "不是有效的 docx 文件", 415) from None
    else:
        text = decode(path.read_bytes())
        if "\x00" in text:
            raise AppError("INVALID_FORMAT", "文本中包含二进制数据", 415)
        if ext == ".html" and not re.search(r"<(?:html|article|main|body|div|p)\b", text, re.I):
            raise AppError("INVALID_FORMAT", "未识别 HTML 正文结构", 415)


def metadata_from_text(text):
    result = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(文档\s*ID|版本日期|公司|适用范围|合成材料)\s*[:：]\s*(.*?)\s*$", line)
        if match:
            result[META.get(match[1], "document_id")] = match[2]
    lines = [x.strip().lstrip("# ") for x in text.splitlines() if x.strip()]
    if lines:
        result["title"] = lines[0]
    return result


@document_errors
def inspect_metadata(path, overrides=None):
    ext = path.suffix.lower()
    if ext == ".docx":
        text = "\n".join(p.text for p in Document(path).paragraphs)
        meta = metadata_from_text(text)
    elif ext == ".pdf":
        meta = metadata_from_text("\n".join(p.extract_text() or "" for p in PdfReader(path).pages[:3]))
    else:
        text = decode(path.read_bytes())
        if ext == ".md" and text.startswith("---"):
            match = re.match(r"^---\s*\n(.*?)\n---(?:\s*\n|$)", text, re.S)
            try:
                meta = yaml.safe_load(match[1]) if match else {}
            except yaml.YAMLError:
                raise AppError("INVALID_METADATA", "Markdown 元数据格式错误", 422) from None
            if not isinstance(meta, dict):
                raise AppError("INVALID_METADATA", "元数据必须为键值对象", 422)
        elif ext == ".html":
            soup = BeautifulSoup(text, "html.parser")
            meta = {t.get("name"): t.get("content") for t in soup.select("meta[name][content]")}
            if "title" not in meta and soup.title:
                meta["title"] = soup.title.get_text()
        else:
            meta = metadata_from_text(text)
    for key, value in (overrides or {}).items():
        if key in meta and str(meta[key]) != str(value):
            raise AppError("METADATA_CONFLICT", f"上传元数据与原文不一致：{key}", 422)
        meta[key] = value
    for key in ("document_id", "title", "version_date", "scope"):
        if not meta.get(key):
            raise AppError("MISSING_METADATA", f"请补充元数据：{key}", 422)
    if not re.fullmatch(r"[\w.-]{1,120}", str(meta["document_id"])):
        raise AppError("INVALID_METADATA", "文档 ID 仅允许文字、数字、下划线、点和连字符", 422)
    try:
        meta["version_date"] = date.fromisoformat(str(meta["version_date"])).isoformat()
    except ValueError:
        raise AppError("INVALID_METADATA", "版本日期须为 YYYY-MM-DD", 422) from None
    return {k: str(v) if isinstance(v, date) else v for k, v in meta.items()}


def text_blocks(text, page=None):
    blocks, heading, buffer, start = [], [], [], 1

    def flush(end):
        nonlocal buffer
        value = "\n".join(buffer).strip()
        if value:
            loc = {"line_start": start, "line_end": end}
            if page is not None:
                loc["page"] = page
            match = CLAUSE.match(value)
            if match:
                loc["clause"] = match[0].strip()
            blocks.append(
                Block(
                    block_id=f"b{len(blocks)}",
                    text=value,
                    order=len(blocks),
                    heading_path=list(heading),
                    locator=loc,
                    type="clause" if match else "paragraph",
                    hard_boundary=bool(match),
                )
            )
        buffer = []

    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or CLAUSE.match(line) or line.startswith("#"):
            flush(n - 1)
            start = n
        if line.startswith("#"):
            heading = [line.lstrip("# ").strip()]
        elif line.strip():
            if not buffer:
                start = n
            buffer.append(line)
    flush(len(text.splitlines()))
    return blocks


def parse_markdown(path):
    text = decode(path.read_bytes())
    # Blank YAML lines in place so original line positions stay stable.
    if text.startswith("---"):
        match = re.match(r"^---\s*\n.*?\n---(?:\s*\n|$)", text, re.S)
        if match:
            text = "\n" * match[0].count("\n") + text[match.end() :]
    lines, heading, result = text.splitlines(), [], []
    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    in_table = False
    for token in tokens:
        if token.type == "table_close":
            in_table = False
            continue
        if in_table:
            continue
        if token.type == "heading_open":
            level = int(token.tag[1:])
            heading = heading[: level - 1]
        if token.type == "table_open" and token.map:
            in_table = True
            start, end = token.map
            result.append(
                Block(
                    block_id=f"b{len(result)}",
                    text="\n".join(lines[start:end]),
                    type="table",
                    heading_path=list(heading),
                    hard_boundary=True,
                    locator={"line_start": start + 1, "line_end": end},
                )
            )
            continue
        if token.type not in {"inline", "fence", "code_block"} or not token.map:
            continue
        start, end = token.map
        raw = "\n".join(lines[start:end])
        if raw.lstrip().startswith("#"):
            heading.append(token.content)
            continue
        if not raw.strip():
            continue
        # Table tokens' inline rows have no map: the table is preserved below as one atomic block.
        result.append(
            Block(
                block_id=f"b{len(result)}",
                text=raw.strip(),
                order=len(result),
                heading_path=list(heading),
                locator={"line_start": start + 1, "line_end": end},
                hard_boundary=bool(CLAUSE.match(raw)),
                type="clause" if CLAUSE.match(raw) else "paragraph",
            )
        )
    result.sort(key=lambda b: b.locator["line_start"])
    return result


def parse_docx(path):
    doc, result, heading = Document(path), [], []
    for index, child in enumerate(doc.element.body):
        if child.tag == qn("w:p"):
            p = Paragraph(child, doc)
            if not p.text.strip():
                continue
            if p.style and (p.style.name.startswith("Heading") or p.style.name == "Title"):
                heading = [p.text]
                continue
            numbered = bool(child.xpath("./w:pPr/w:numPr")) or bool(CLAUSE.match(p.text))
            result.append(
                Block(
                    block_id=f"p{index}",
                    text=p.text,
                    type="clause" if numbered else "paragraph",
                    heading_path=list(heading),
                    hard_boundary=numbered,
                    locator={"paragraph": index},
                )
            )
        elif child.tag == qn("w:tbl"):
            table = Table(child, doc)
            rows = [" | ".join(c.text for c in row.cells) for row in table.rows]
            for n, row in enumerate(rows[1:] or rows):
                text = rows[0] + "\n" + row if len(rows) > 1 else row
                result.append(
                    Block(
                        block_id=f"t{index}r{n}",
                        text=text,
                        type="table_row",
                        hard_boundary=True,
                        heading_path=list(heading),
                        locator={"table": index, "row": n + 1, "header_row": 0},
                    )
                )
    return result


def parse_html(path):
    soup = BeautifulSoup(decode(path.read_bytes()), "html.parser")
    for node in list(soup.select('script,style,nav,footer,header,[hidden],[aria-hidden="true"]')):
        node.decompose()
    for node in list(soup.select("[style]")):
        if re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", node.get("style", ""), re.I):
            node.decompose()
    root = soup.select_one("article,main") or soup.body or soup
    result, heading = [], []
    for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "div"]):
        if node.name == "div" and node.find(["p", "li", "tr", "div", "h1", "h2", "h3", "h4"]):
            continue
        if node.name == "div" and node.find_parent(["li", "td", "th"]):
            continue
        if node.name.startswith("h"):
            level = int(node.name[1:])
            heading = heading[: level - 1] + [node.get_text(" ", strip=True)]
            continue
        if node.name == "p" and node.find_parent(["li", "td", "th"]):
            continue
        if node.name == "li" and node.find_parent("li"):
            continue
        value = node.get_text(" ", strip=True)
        if not value:
            continue
        parts, current = [], node
        while current and current != root:
            siblings = current.find_previous_siblings(current.name)
            parts.append(f"{current.name}:nth-of-type({len(siblings) + 1})")
            current = current.parent
        loc = {"dom_path": " > ".join(reversed(parts)), "item_id": node.get("id")}
        if node.name == "tr":
            table = node.find_parent("table")
            header = table.find("tr")
            if node == header and header.find("th"):
                continue
            if header and header != node:
                value = header.get_text(" | ", strip=True) + "\n" + value
        if node.name == "li":
            loc["clause"] = len(node.find_previous_siblings("li")) + 1
        result.append(
            Block(
                block_id=f"b{len(result)}",
                text=value,
                heading_path=list(heading),
                type="table_row" if node.name == "tr" else "item" if node.name == "li" else "paragraph",
                hard_boundary=node.name in {"li", "tr"},
                locator=loc,
            )
        )
    return result


def mineru_blocks(payload):
    """Adapt MinerU 2.x content_list JSON; reject incompatible contracts instead of guessing."""
    if not isinstance(payload, list):
        raise AppError("MINERU_SCHEMA", "预期 MinerU 2.x content_list 数组，请核对解析器版本", 422)
    result, heading = [], []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise AppError("MINERU_SCHEMA", "PDF 元素须为带类型的对象", 422)
        kind = item.get("type")
        if kind in {"header", "footer", "page_number", "discarded"}:
            continue
        if (
            not isinstance(item.get("page_idx"), int)
            or isinstance(item.get("page_idx"), bool)
            or item["page_idx"] < 0
        ):
            raise AppError("MINERU_SCHEMA", "PDF 块缺少页码，不能定位", 422)
        page = int(item["page_idx"]) + 1
        if kind == "table":
            raw = item.get("table_body", "")
            if not isinstance(raw, str):
                raise AppError("MINERU_SCHEMA", "PDF 表格内容格式错误", 422)
            table = BeautifulSoup(raw, "html.parser")
            table_rows = [
                " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"]))
                for row in table.find_all("tr")
            ]
            if not table_rows:
                raise AppError("MINERU_SCHEMA", "PDF 表格缺少可靠行结构", 422)
            value = "\n".join(table_rows)
        else:
            value = item.get("text", "")
        if not isinstance(value, str):
            raise AppError("MINERU_SCHEMA", "PDF 文本块必须包含字符串", 422)
        if not value.strip():
            continue
        if item.get("text_level", 0) > 0:
            heading = [value.strip()]
        bbox = item.get("bbox")
        if bbox is not None and (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(x, (int, float))
                and not isinstance(x, bool)
                and math.isfinite(x)
                and 0 <= x <= 1000
                for x in bbox
            )
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
        ):
            raise AppError("MINERU_SCHEMA", "无效的 PDF 区域坐标", 422)
        loc = {
            "page": page,
            "bbox": bbox,
            "coordinate_system": "normalized_1000",
            "reading_order": len(result),
        }
        hard = kind == "table" or bool(CLAUSE.match(value)) or item.get("text_level", 0) > 0
        if result and bbox and result[-1].locator.get("bbox") and page == result[-1].locator["page"]:
            prev = result[-1].locator["bbox"]
            # Distinct columns have non-overlapping horizontal ranges; large vertical gaps separate subjects.
            overlap_x = min(prev[2], bbox[2]) - max(prev[0], bbox[0])
            if overlap_x <= 0 or bbox[1] - prev[3] > 80:
                hard = True
        if kind == "table":
            for n, row in enumerate(table_rows[1:] or table_rows):
                result.append(
                    Block(
                        block_id=f"b{len(result)}",
                        text=table_rows[0] + "\n" + row if len(table_rows) > 1 else row,
                        type="table_row",
                        locator={**loc, "table_row": n + 1, "header_row": 0},
                        heading_path=list(heading),
                        hard_boundary=True,
                    )
                )
        else:
            result.append(
                Block(
                    block_id=f"b{len(result)}",
                    text=value.strip(),
                    type=kind or "text",
                    locator=loc,
                    heading_path=list(heading),
                    hard_boundary=hard,
                )
            )
    return result


def parse_pdf(path, settings, artifacts_dir):
    if settings.pdf_backend == "test_text":
        result = []
        for page, obj in enumerate(PdfReader(path).pages, 1):
            result.extend(text_blocks(obj.extract_text() or "", page=page))
        return result, ["TEST_ONLY: pypdf text extraction, no MinerU layout validation"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    command = [
        arg.format(input=str(path.resolve()), output=str(artifacts_dir.resolve()))
        for arg in settings.mineru_command
    ]
    registry = os.environ.get(REGISTRY_ENV)
    if registry:
        command = [sys.executable, "-m", "xiaoying.managed_process", str(os.getpid()), *command]
    try:
        with (artifacts_dir / "mineru.log").open("wb") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=settings.pdf_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise AppError("PDF_TIMEOUT", "MinerU 解析超时", 504, True) from None
            finally:
                if registry and proc.poll() is not None:
                    (Path(registry) / f"{proc.pid}.json").unlink(missing_ok=True)
    except FileNotFoundError:
        raise AppError(
            "MINERU_UNAVAILABLE", "请在独立环境安装 MinerU 2.x 并配置 MINERU_COMMAND", 503
        ) from None
    if code:
        raise AppError("PDF_PARSE_FAILED", "MinerU 解析失败，见本地解析日志", 422)
    paths = list(artifacts_dir.rglob("*_content_list.json"))
    if len(paths) != 1:
        raise AppError("MINERU_SCHEMA", "未得到唯一 content_list.json", 422)
    blocks = mineru_blocks(json.loads(paths[0].read_text()))
    warnings = (
        ["PDF 部分块缺少 bbox，使用显式阅读顺序定位"]
        if any(not b.locator.get("bbox") for b in blocks)
        else []
    )
    return blocks, warnings


@document_errors
def parse_document(path: Path, metadata, settings, artifacts_dir=None):
    validate_file(path, settings)
    warnings = []
    ext = path.suffix.lower()
    if ext == ".md":
        blocks = parse_markdown(path)
    elif ext == ".docx":
        blocks = parse_docx(path)
    elif ext == ".html":
        blocks = parse_html(path)
    elif ext == ".txt":
        blocks = text_blocks(decode(path.read_bytes()))
    else:
        if artifacts_dir is None:
            with tempfile.TemporaryDirectory() as tmp:
                blocks, warnings = parse_pdf(path, settings, Path(tmp))
        else:
            blocks, warnings = parse_pdf(path, settings, artifacts_dir)
    if not blocks:
        raise AppError("EMPTY_DOCUMENT", "未解析出可用正文", 422)
    for n, block in enumerate(blocks):
        block.order = n
        block.block_id = f"b{n}"
    return DocumentIR(metadata=metadata, blocks=blocks, parser_version=f"{ext}-v1", warnings=warnings)
