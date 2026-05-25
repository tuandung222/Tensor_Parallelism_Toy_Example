#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

TEXT_EXTENSIONS = {".md", ".ts", ".tsx", ".json", ".yml", ".yaml", ".css"}
IGNORED_PARTS = {"node_modules", "build", ".docusaurus", ".git", "__pycache__", ".lightning_studio", ".idea", ".vscode"}
PERSONAL_LEAK_PATTERN = re.compile(
    r"Small[- ]?Qwen|CLIP[- ]?HAR|LLMs?-with-Semantic-Search|Open[- ]?vocabulary-Action-Recognition|tuandung222",
    re.IGNORECASE,
)


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_EXTENSIONS:
            continue
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_readme_empty(errors: list[str]) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        errors.append("README.md is missing")
        return
    if readme.stat().st_size != 0:
        errors.append(f"README.md must stay empty, got {readme.stat().st_size} bytes")


def check_privacy_controls(errors: list[str]) -> None:
    robots = ROOT / "static" / "robots.txt"
    if not robots.exists():
        errors.append("static/robots.txt is missing")
    elif read(robots).strip() != "User-agent: *\nDisallow: /":
        errors.append("static/robots.txt must disallow all crawling")
    config = read(ROOT / "docusaurus.config.ts")
    if "noindex,nofollow,noarchive,nosnippet" not in config:
        errors.append("docusaurus.config.ts is missing noindex robots metadata")
    if "sitemap: false" not in config:
        errors.append("docusaurus.config.ts must disable sitemap generation")


def check_source_hygiene(errors: list[str]) -> None:
    for path in iter_text_files():
        rel = path.relative_to(ROOT)
        if rel == Path("README.md"):
            continue
        text = read(path)
        if "\u2014" in text:
            errors.append(f"em dash found in {rel}")
        if path.is_relative_to(DOCS) and PERSONAL_LEAK_PATTERN.search(text):
            errors.append(f"personal identity leak found in {rel}")


def check_sidebar_doc_ids(errors: list[str]) -> None:
    sidebars = read(ROOT / "sidebars.ts")
    ids: list[str] = []
    for block in re.findall(r"items:\s*\[(.*?)\]", sidebars, re.S):
        ids.extend(re.findall(r"'([^']+)'", block))
    # also include top-level doc ids referenced via {type: 'doc', id: '...'}
    ids.extend(re.findall(r"id:\s*'([^']+)'", sidebars))
    for doc_id in ids:
        candidates = [DOCS / f"{doc_id}.md", DOCS / doc_id / "index.md"]
        if not any(candidate.exists() for candidate in candidates):
            errors.append(f"sidebar doc id does not exist: {doc_id}")


def check_absolute_doc_links(errors: list[str]) -> None:
    for path in DOCS.rglob("*.md"):
        text = read(path)
        for match in re.finditer(r"\]\((/docs/[^)#]+)", text):
            doc_id = match.group(1).replace("/docs/", "")
            candidates = [DOCS / f"{doc_id}.md", DOCS / doc_id / "index.md"]
            if not any(candidate.exists() for candidate in candidates):
                errors.append(f"broken absolute docs link in {path.relative_to(ROOT)}: {doc_id}")


def main() -> int:
    errors: list[str] = []
    check_readme_empty(errors)
    check_privacy_controls(errors)
    check_source_hygiene(errors)
    check_sidebar_doc_ids(errors)
    check_absolute_doc_links(errors)

    doc_files = list(DOCS.rglob("*.md"))
    total_words = sum(len(read(path).split()) for path in doc_files)
    print(f"Docs: {len(doc_files)} markdown files, {total_words} words")
    print(f"README bytes: {(ROOT / 'README.md').stat().st_size}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("QA passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
