"""Read-only tools over a local corpus of Russian legal documents (Markdown).

Five tools:
  1. list_documents()         — list available corpus documents
  2. list_sections(doc_id)    — list articles/points in a document
  3. search_by_article(num)   — get full text of a specific article by number
  4. search_documents(query)  — full-text search across all documents
  5. read_document(doc_id, start, length) — read a bounded excerpt

All tools are read-only, path-confined, and bounded in output size.
dispatch() never raises — it always returns a string (for feeding back to the model).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"

# --- Harness configuration (constants, not JSONL — one harness, one policy) ---

HARNESS = {
    "name": "local-docs",
    "document_root": str(CORPUS_DIR),
    "tools": ["list_documents", "list_sections", "search_by_article",
              "search_documents", "read_document"],
    "max_tool_calls": 8,
    "max_tool_result_chars": 4000,
    "tool_timeout_s": 10,
}

ORACLE_HARNESS = {
    "name": "oracle",
    "document_root": str(CORPUS_DIR),
    # no tools — prompt-only condition
}

# --- Tool schemas (OpenAI function format; flattened for Responses API) ---

TOOL_SCHEMAS_FLAT = [
    {
        "type": "function",
        "name": "list_documents",
        "description": "List all available legal documents in the corpus. Returns document IDs, titles, and sizes.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "list_sections",
        "description": "List all articles (Статья N) or points (Пункт N) in a specific document. Use this to see the table of contents before reading specific articles.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID (filename without .md), e.g. 'zhk-rf', 'gk-rf', 'pp-rf-491'"},
            },
            "required": ["document_id"],
        },
    },
    {
        "type": "function",
        "name": "search_by_article",
        "description": "Get the full text of a specific article by its number. Works for codes (Статья N) and decrees (Пункт N). This is the most precise way to find a specific legal provision.",
        "parameters": {
            "type": "object",
            "properties": {
                "article": {"type": "string", "description": "Article or point number, e.g. '143.1', '161', '10', '47.1'"},
                "code": {"type": "string", "description": "Optional: document ID or code name to search in, e.g. 'zhk-rf', 'ЖК РФ', 'pp-rf-491'"},
            },
            "required": ["article"],
        },
    },
    {
        "type": "function",
        "name": "search_documents",
        "description": "Search across all legal documents for a text query. Returns matching snippets with document IDs and article references. Use this when you don't know the exact article number.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in Russian, e.g. 'содержание общего имущества', 'лицензирование УК'"},
                "max_results": {"type": "integer", "description": "Maximum number of results to return (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "read_document",
        "description": "Read a bounded excerpt from a document by character offset. Use this for reading large sections that don't match a single article.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID (filename without .md)"},
                "start": {"type": "integer", "description": "Starting character offset (default 0)", "default": 0},
                "length": {"type": "integer", "description": "Number of characters to read (default 4000)", "default": 4000},
            },
            "required": ["document_id"],
        },
    },
]

# OpenAI chat format (nested under "function")
TOOL_SCHEMAS_CHAT = [
    {"type": "function", "function": {k: v for k, v in s.items() if k != "type"}}
    for s in TOOL_SCHEMAS_FLAT
]


# --- Helpers ---

_DOC_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ARTICLE_NUM_RE = re.compile(r"^[0-9]+(?:[\.\-][0-9]+)*$")
# Article header: ## Статья 143.1. Title  OR  ### Пункт 10  OR  ### Пункт 10. Title  (2+ hashes, period+title optional)
_HEADER_RE = re.compile(r"^##+\s+(?:Статья|Пункт)\s+(\d+(?:[\.\-]\d+)*)\.?\s*(.*)$", re.MULTILINE)


def _validate_doc_id(doc_id: str, root: Path) -> Path | None:
    """Validate document_id and return safe path, or None if invalid."""
    if not doc_id or not _DOC_ID_RE.match(doc_id):
        return None
    path = (root / f"{doc_id}.md").resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    if not path.exists():
        return None
    return path


def _parse_sections(text: str) -> list[dict]:
    """Extract article/point sections from Markdown text."""
    sections = []
    for m in _HEADER_RE.finditer(text):
        sections.append({"article": m.group(1), "title": m.group(2).strip()})
    return sections


def _find_article(text: str, article: str) -> str | None:
    """Find and return the full text of a specific article/point by number."""
    # Build pattern: ## Статья 143.1.  or  ### Пункт 10  (2+ hashes, period optional)
    pattern = rf"^##+\s+(?:Статья|Пункт)\s+{re.escape(article)}\.?\s"
    header_re = re.compile(pattern, re.MULTILINE)
    m = header_re.search(text)
    if not m:
        # Try with different number formats (e.g., 143.1 vs 143-1)
        alt = article.replace("-", ".")
        if alt != article:
            pattern_alt = rf"^##+\s+(?:Статья|Пункт)\s+{re.escape(alt)}\.?\s"
            m = re.search(pattern_alt, text, re.MULTILINE)

    if not m:
        return None

    # Find the next article/point header or end of file
    start = m.end()
    next_header = re.search(r"^##+\s+(?:Статья|Пункт)\s+\d", text[start:], re.MULTILINE)
    if next_header:
        end = start + next_header.start()
    else:
        end = len(text)

    # Return header + body
    body = text[m.start():end].strip()
    return body


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding a notice if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"


# --- Timeout handler (thread-safe via ThreadPoolExecutor) ---

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

_TOOL_POOL = ThreadPoolExecutor(max_workers=2)


def _with_timeout(fn, timeout_s: int, *args, **kwargs):
    """Run fn with a timeout. Returns result or error string.

    Uses a ThreadPoolExecutor instead of signals so it works in worker threads.
    """
    try:
        future = _TOOL_POOL.submit(fn, *args, **kwargs)
        result = future.result(timeout=timeout_s)
        return result
    except FuturesTimeoutError:
        return "ERROR: tool timed out"
    except Exception as e:
        return f"ERROR: {e}"


# --- Tool implementations ---

def tool_list_documents(ctx: dict) -> str:
    root = Path(ctx["document_root"])
    docs = []
    for f in sorted(root.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        sections = _parse_sections(text)
        title_line = text.split("\n")[0].lstrip("# ").strip()
        docs.append({
            "id": f.stem,
            "title": title_line,
            "articles": len(sections),
            "chars": len(text),
        })
    return json.dumps(docs, ensure_ascii=False)


def tool_list_sections(document_id: str, ctx: dict) -> str:
    root = Path(ctx["document_root"])
    path = _validate_doc_id(document_id, root)
    if path is None:
        return f"ERROR: invalid or unknown document_id '{document_id}'"
    text = path.read_text(encoding="utf-8")
    sections = _parse_sections(text)
    if not sections:
        return json.dumps({"document_id": document_id, "sections": []}, ensure_ascii=False)
    return json.dumps({"document_id": document_id, "sections": sections}, ensure_ascii=False)


def tool_search_by_article(article: str, code: str = "", ctx: dict = "") -> str:
    root = Path(ctx["document_root"])
    if not article or not _ARTICLE_NUM_RE.match(article):
        return f"ERROR: invalid article number '{article}'. Use a number like '143.1' or '161'."

    # Determine which documents to search
    if code:
        # Try to match code to a document ID
        code_lower = code.lower().strip()
        # Map common names to file stems
        code_map = {
            "жк рф": "zhk-rf", "жк": "zhk-rf", "zhk-rf": "zhk-rf", "zhk": "zhk-rf",
            "гк рф": "gk-rf", "гк": "gk-rf", "gk-rf": "gk-rf", "gk": "gk-rf",
            "грк рф": "grk-rf", "грк": "grk-rf", "grk-rf": "grk-rf",
            "коап рф": "koap-rf", "коап": "koap-rf", "koap-rf": "koap-rf",
            "пп-491": "pp-rf-491", "пп 491": "pp-rf-491", "pp-rf-491": "pp-rf-491",
            "пп-47": "pp-rf-47", "пп 47": "pp-rf-47", "pp-rf-47": "pp-rf-47",
            "пп-290": "pp-rf-290", "пп 290": "pp-rf-290", "pp-rf-290": "pp-rf-290",
            "пп-354": "pp-rf-354", "пп 354": "pp-rf-354", "pp-rf-354": "pp-rf-354",
        }
        doc_id = code_map.get(code_lower, code_lower)
        path = _validate_doc_id(doc_id, root)
        if path is None:
            return f"ERROR: unknown code '{code}'. Use list_documents to see available document IDs."
        text = path.read_text(encoding="utf-8")
        result = _find_article(text, article)
        if result:
            return _truncate(result, ctx["max_tool_result_chars"])
        return f"Article {article} not found in {doc_id}."

    # No code specified — search all documents
    results = []
    for f in sorted(root.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        found = _find_article(text, article)
        if found:
            results.append({"document_id": f.stem, "text": found})
    if not results:
        return f"Article {article} not found in any document."
    if len(results) == 1:
        return _truncate(results[0]["text"], ctx["max_tool_result_chars"])
    # Multiple matches — return first with note
    out = f"Found article {article} in {len(results)} documents. Showing first match ({results[0]['document_id']}):\n\n"
    out += _truncate(results[0]["text"], ctx["max_tool_result_chars"] - len(out) - 100)
    return out


def tool_search_documents(query: str, max_results: int = 5, ctx: dict = None) -> str:
    root = Path(ctx["document_root"])
    if not query.strip():
        return "ERROR: query is empty"

    results = []
    query_lower = query.lower()
    # Also try regex if query looks like a pattern
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = None

    for f in sorted(root.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        lines = text.split("\n")
        current_article = ""
        for i, line in enumerate(lines):
            # Track current article
            hm = _HEADER_RE.match(line)
            if hm:
                current_article = f"Статья {hm.group(1)}"
                continue

            # Check for match
            matched = query_lower in line.lower()
            if not matched and pattern:
                matched = bool(pattern.search(line))
            if matched:
                # Get context: 1 line before, 2 lines after
                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                snippet = "\n".join(lines[start:end])
                results.append({
                    "document_id": f.stem,
                    "article": current_article,
                    "line": i + 1,
                    "snippet": snippet[:300],
                })
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break

    if not results:
        return f"No results found for '{query}'."

    return json.dumps(results, ensure_ascii=False)


def tool_read_document(document_id: str, start: int = 0, length: int = 4000, ctx: dict = None) -> str:
    root = Path(ctx["document_root"])
    path = _validate_doc_id(document_id, root)
    if path is None:
        return f"ERROR: invalid or unknown document_id '{document_id}'"
    text = path.read_text(encoding="utf-8")
    start = max(0, start)
    length = min(length, ctx["max_tool_result_chars"])
    excerpt = text[start:start + length]
    if start + length < len(text):
        excerpt += f"\n\n[... {len(text) - start - length} more chars ...]"
    return excerpt


# --- Dispatch ---

_TOOL_FUNCS = {
    "list_documents": lambda args, ctx: tool_list_documents(ctx),
    "list_sections": lambda args, ctx: tool_list_sections(args.get("document_id", ""), ctx),
    "search_by_article": lambda args, ctx: tool_search_by_article(
        args.get("article", ""), args.get("code", ""), ctx),
    "search_documents": lambda args, ctx: tool_search_documents(
        args.get("query", ""), args.get("max_results", 5), ctx),
    "read_document": lambda args, ctx: tool_read_document(
        args.get("document_id", ""), args.get("start", 0), args.get("length", 4000), ctx),
}


def dispatch(name: str, args: dict, ctx: dict) -> str:
    """Execute a tool by name with the given arguments. Never raises."""
    if name not in _TOOL_FUNCS:
        return f"ERROR: unknown tool '{name}'. Available: {list(_TOOL_FUNCS.keys())}"
    if not isinstance(args, dict):
        try:
            args = json.loads(args) if isinstance(args, str) else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
    timeout_s = ctx.get("tool_timeout_s", 10)
    return _with_timeout(_TOOL_FUNCS[name], timeout_s, args, ctx)


def make_ctx(harness: dict) -> dict:
    """Build a context dict from a harness config."""
    return {
        "document_root": harness["document_root"],
        "max_tool_result_chars": harness.get("max_tool_result_chars", 4000),
        "tool_timeout_s": harness.get("tool_timeout_s", 10),
    }


# --- Oracle: citation parser ---

# Patterns observed in expert answers:
#   Статья 259.4
#   часть 3 ст. 143.1 Жилищного кодекса
#   ч. 1 ст. 174
#   ст. 166 Жилищного кодекса РФ
#   ст. 53 Градостроительного кодекса РФ
#   ч. 8 ст. 10 149-ФЗ
CITATION_RE = re.compile(
    r"(?:стать[ьеюяи]|ст\.?)\s*(\d+(?:\.\d+)*)"
    r"(?:\s+((?:Жилищного|ЖК|Гражданского|ГК|Градостроительного|ГрК|КоАП|Уголовного|УК)\s*(?:кодекса)?\s*РФ?))?",
    re.IGNORECASE,
)

# Map code name fragments to document IDs
_CODE_NAME_MAP = {
    "жилищн": "zhk-rf", "жк": "zhk-rf",
    "гражданск": "gk-rf", "гк": "gk-rf",
    "градостроительн": "grk-rf", "грк": "grk-rf",
    "коап": "koap-rf", "административ": "koap-rf",
}


def _resolve_code_id(code_text: str) -> str | None:
    """Map a code name fragment to a corpus document ID."""
    code_lower = code_text.lower().strip()
    for fragment, doc_id in _CODE_NAME_MAP.items():
        if fragment in code_lower:
            return doc_id
    return None


def parse_citations(text: str) -> list[dict]:
    """Extract article citations from text (e.g., expert_answer).

    Returns list of {article, code_id, code_name} dicts.
    """
    citations = []
    for m in CITATION_RE.finditer(text):
        article = m.group(1)
        code_id = None
        code_name = ""

        # Group 2 is the optional code name (e.g. "ЖК РФ", "Гражданского кодекса РФ")
        if m.group(2):
            code_name = m.group(2).strip()
            code_id = _resolve_code_id(code_name)

        # If no code name in the citation, check text after the match
        if not code_id:
            after = text[m.end():m.end() + 40]
            code_id = _resolve_code_id(after)
            if code_id:
                code_name = after[:30].strip()

        citations.append({
            "article": article,
            "code_id": code_id,
            "code_name": code_name,
        })
    return citations


def get_oracle_articles(expert_answer: str, ctx: dict | None = None) -> list[dict]:
    """Parse citations from expert_answer and look up article text in corpus.

    Returns list of {code, article, title, text, chars} dicts.
    """
    if ctx is None:
        ctx = make_ctx(ORACLE_HARNESS)

    citations = parse_citations(expert_answer)
    if not citations:
        return []

    root = Path(ctx["document_root"])
    results = []
    seen = set()  # deduplicate (code_id, article)

    for cite in citations:
        article = cite["article"]
        code_id = cite["code_id"]

        # Determine which documents to search
        search_docs = []
        if code_id:
            path = _validate_doc_id(code_id, root)
            if path:
                search_docs = [(code_id, path)]
        else:
            # Search all documents
            search_docs = [(f.stem, f) for f in sorted(root.glob("*.md"))]

        for doc_id, path in search_docs:
            key = (doc_id, article)
            if key in seen:
                continue
            text = path.read_text(encoding="utf-8")
            found = _find_article(text, article)
            if found:
                seen.add(key)
                # Extract title from the article text
                title_match = re.match(r"^##+\s+(?:Статья|Пункт)\s+\d+(?:[\.\-]\d+)*\.\s*(.+)$", found.split("\n")[0])
                title = title_match.group(1).strip() if title_match else ""
                results.append({
                    "code": doc_id,
                    "article": article,
                    "title": title,
                    "text": _truncate(found, ctx["max_tool_result_chars"]),
                    "chars": len(found),
                })
                break  # Found in this doc, move to next citation

    return results


# --- Self-check ---

if __name__ == "__main__":
    import tempfile, os

    # Create a tiny test corpus
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test-code.md").write_text(
            "# Тестовый кодекс\n\n"
            "## Статья 1. Первая статья\n\nТекст первой статьи.\n\n"
            "## Статья 2.2. Вторая статья\n\nТекст второй статьи с важным словом.\n\n"
            "## Статья 10. Десятая статья\n\nТекст десятой статьи.\n",
            encoding="utf-8",
        )
        (tmpdir / "test-decree.md").write_text(
            "# Тестовое постановление\n\n"
            "## Пункт 1. Первый пункт\n\nТекст первого пункта.\n\n"
            "## Пункт 5. Пятый пункт\n\nТекст пятого пункта.\n",
            encoding="utf-8",
        )

        ctx = {
            "document_root": str(tmpdir),
            "max_tool_result_chars": 4000,
            "tool_timeout_s": 5,
        }

        counts = [0, 0]  # [passed, failed]

        def check(name, condition, detail=""):
            if condition:
                print(f"  ✓ {name}")
                counts[0] += 1
            else:
                print(f"  ✗ {name} {detail}")
                counts[1] += 1

        print("Running self-check...")

        # 1. list_documents
        result = dispatch("list_documents", {}, ctx)
        docs = json.loads(result)
        check("list_documents returns 2 docs", len(docs) == 2, f"got {len(docs)}")
        check("list_documents has test-code", any(d["id"] == "test-code" for d in docs))

        # 2. list_sections
        result = dispatch("list_sections", {"document_id": "test-code"}, ctx)
        data = json.loads(result)
        check("list_sections returns 3 articles", len(data["sections"]) == 3, f"got {len(data['sections'])}")
        check("list_sections has article 2.2", any(s["article"] == "2.2" for s in data["sections"]))

        # 3. search_by_article
        result = dispatch("search_by_article", {"article": "2.2", "code": "test-code"}, ctx)
        check("search_by_article finds 2.2", "Вторая статья" in result, f"got: {result[:100]}")
        result = dispatch("search_by_article", {"article": "10", "code": "test-code"}, ctx)
        check("search_by_article finds 10", "Десятая статья" in result)
        result = dispatch("search_by_article", {"article": "999"}, ctx)
        check("search_by_article returns not found", "not found" in result.lower())

        # 4. search_documents
        result = dispatch("search_documents", {"query": "важным"}, ctx)
        data = json.loads(result)
        check("search_documents finds 'важным'", len(data) >= 1, f"got {len(data)}")
        check("search_documents in test-code", any(r["document_id"] == "test-code" for r in data))

        # 5. read_document
        result = dispatch("read_document", {"document_id": "test-code", "start": 0, "length": 50}, ctx)
        check("read_document returns content", "Тестовый кодекс" in result)
        check("read_document truncates", len(result) <= 200)  # 50 chars + notice

        # 6. Path traversal protection
        result = dispatch("read_document", {"document_id": "../../../etc/passwd"}, ctx)
        check("path traversal rejected", "ERROR" in result)
        result = dispatch("read_document", {"document_id": "nonexistent"}, ctx)
        check("nonexistent doc rejected", "ERROR" in result)

        # 7. Invalid tool name
        result = dispatch("hack_the_planet", {}, ctx)
        check("unknown tool rejected", "ERROR" in result)

        # 8. Invalid article number
        result = dispatch("search_by_article", {"article": "abc"}, ctx)
        check("invalid article rejected", "ERROR" in result)

        # 9. Citation parser
        citations = parse_citations("Согласно ст. 2.2 Тестового кодекса и статья 10")
        check("parse_citations finds 2.2", any(c["article"] == "2.2" for c in citations))
        check("parse_citations finds 10", any(c["article"] == "10" for c in citations))

        # 10. Oracle articles
        oracle = get_oracle_articles("Согласно ст. 2.2 и статья 10", ctx)
        check("get_oracle_articles returns 2 articles", len(oracle) == 2, f"got {len(oracle)}")
        if oracle:
            check("oracle article has text", "text" in oracle[0] and len(oracle[0]["text"]) > 0)

        print(f"\n{counts[0]} passed, {counts[1]} failed")
        if counts[1]:
            exit(1)
        print("All checks passed!")
