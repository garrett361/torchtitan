"""Generate a representative test sample from a 7M-balanced SFT JSONL file.

Scans the source file and fills per-category quotas greedily.  Categories are
checked in priority order (tool cases before multi-turn cases) so each record is
assigned to at most one bucket.  Output is written in category order for
readability.

Usage::

    # directory source (scans part_*.jsonl in sorted order until quota filled)
    python -m torchtitan.models.granite.scripts.gen_test_data \\
        --source /path/to/dataset_dir

    # single-file source (output defaults to <source_parent>/test_sample/<source_name>)
    python -m torchtitan.models.granite.scripts.gen_test_data \\
        --source /path/to/part_00.jsonl

Categories and default quota (2 each):

    tool_with_rc       — has tool message AND only the last assistant has reasoning_content
    tool_no_rc         — has tool message AND no assistant has reasoning_content
    single_turn_with_rc — n_asst==1 AND at least one assistant has reasoning_content
    single_turn_no_rc  — n_asst==1 AND no assistant has reasoning_content
    multi_turn_all_rc  — n_asst>1, no tool, all assistants have reasoning_content
    multi_turn_mixed_rc — n_asst>1, no tool, some have rc and some don't
    multi_turn_no_rc   — n_asst>1, no tool, no assistant has reasoning_content
"""

import argparse
import json
import pathlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Category:
    name: str
    predicate: Callable[[dict[str, Any]], bool]
    quota: int
    records: list[str] = field(default_factory=list)

    @property
    def filled(self) -> int:
        return len(self.records)

    @property
    def done(self) -> bool:
        return self.filled >= self.quota


def _asst_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in record["messages"] if m["role"] == "assistant"]


def _has_tool(record: dict[str, Any]) -> bool:
    return any(m["role"] == "tool" for m in record["messages"])


def _any_rc(record: dict[str, Any]) -> bool:
    return any("reasoning_content" in m for m in _asst_messages(record))


def _all_rc(record: dict[str, Any]) -> bool:
    asst = _asst_messages(record)
    return bool(asst) and all("reasoning_content" in m for m in asst)


def _no_rc(record: dict[str, Any]) -> bool:
    return not _any_rc(record)


def _mixed_rc(record: dict[str, Any]) -> bool:
    return _any_rc(record) and not _all_rc(record)


def _n_asst(record: dict[str, Any]) -> int:
    return sum(1 for m in record["messages"] if m["role"] == "assistant")


def _total_chars(record: dict[str, Any]) -> int:
    return sum(
        len(m.get("content", "")) + len(m.get("reasoning_content", ""))
        for m in record["messages"]
    )


def _only_last_asst_has_rc(record: dict[str, Any]) -> bool:
    """True when exactly the last assistant turn has reasoning_content, none before it.

    The chat template only strips thinking from turns before the last *user* message.
    Tool-call records typically have a single initial user turn, so last_user_idx=0 and
    no intermediate turn ever gets stripped.  Selecting records where only the final
    answer has rc avoids triggering the template's non-stripping path for intermediate
    tool-call turns.
    """
    asst = _asst_messages(record)
    if not asst:
        return False
    return "reasoning_content" in asst[-1] and not any(
        "reasoning_content" in m for m in asst[:-1]
    )


def _build_categories(quota: int) -> list[Category]:
    return [
        Category(
            "tool_with_rc",
            lambda r: _has_tool(r) and _only_last_asst_has_rc(r),
            quota,
        ),
        Category(
            "tool_no_rc",
            lambda r: _has_tool(r) and _no_rc(r),
            quota,
        ),
        Category(
            "single_turn_with_rc",
            lambda r: not _has_tool(r) and _n_asst(r) == 1 and _any_rc(r),
            quota,
        ),
        Category(
            "single_turn_no_rc",
            lambda r: not _has_tool(r) and _n_asst(r) == 1 and _no_rc(r),
            quota,
        ),
        Category(
            "multi_turn_all_rc",
            lambda r: not _has_tool(r) and _n_asst(r) > 1 and _all_rc(r),
            quota,
        ),
        Category(
            "multi_turn_mixed_rc",
            lambda r: not _has_tool(r) and _n_asst(r) > 1 and _mixed_rc(r),
            quota,
        ),
        Category(
            "multi_turn_no_rc",
            lambda r: not _has_tool(r) and _n_asst(r) > 1 and _no_rc(r),
            quota,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a representative test sample from an SFT JSONL file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source .jsonl file or directory of .jsonl files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output .jsonl file (default: <source>/test_sample/part_00.jsonl)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    parser.add_argument("--quota", type=int, default=5, help="Records per category")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=150_000,
        help="Skip records whose total character count exceeds this",
    )
    args = parser.parse_args()

    categories = _build_categories(args.quota)

    source = pathlib.Path(args.source)
    if not source.exists():
        print(f"error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    if source.is_dir():
        output = pathlib.Path(args.output) if args.output else source / "test_sample" / "part_00.jsonl"
        source_files = sorted(p for p in source.glob("*.jsonl") if p != output)
        if not source_files:
            print(f"error: no .jsonl files found in {source}", file=sys.stderr)
            sys.exit(1)
    else:
        output = pathlib.Path(args.output) if args.output else source.parent / "test_sample" / source.name
        source_files = [source]

    if output.exists() and not args.overwrite:
        print(f"error: output already exists: {output} (use --overwrite to replace)", file=sys.stderr)
        sys.exit(1)

    scanned = 0
    for source_file in source_files:
        if all(c.done for c in categories):
            break
        with source_file.open() as f:
            for lineno, raw_line in enumerate(f, 1):
                if all(c.done for c in categories):
                    break
                scanned += 1
                try:
                    record = json.loads(raw_line)
                    if _total_chars(record) > args.max_chars:
                        continue
                    for cat in categories:
                        if not cat.done and cat.predicate(record):
                            cat.records.append(raw_line.rstrip("\n"))
                            break
                except (json.JSONDecodeError, KeyError) as exc:
                    print(
                        f"warning: skipping {source_file.name}:{lineno}: {exc}",
                        file=sys.stderr,
                    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for cat in categories:
            for line in cat.records:
                f.write(line + "\n")

    total = sum(c.filled for c in categories)
    print(f"Scanned {scanned} records, wrote {total} to {output}\n")
    col = max(len(c.name) for c in categories)
    print(f"  {'category':<{col}}  filled / quota")
    print(f"  {'-' * col}  --------------")
    for cat in categories:
        status = "OK" if cat.done else "INCOMPLETE"
        print(f"  {cat.name:<{col}}  {cat.filled:>3} / {cat.quota:<3}  {status}")

    if not all(c.done for c in categories):
        incomplete = [c.name for c in categories if not c.done]
        print(
            f"\nwarning: {len(incomplete)} categories did not reach quota: {incomplete}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
