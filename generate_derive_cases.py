#!/usr/bin/env python3
"""Generate derive-transform parity test cases from a customer rule corpus.

Reads a CSV with one derive-rule JSON per row (column ``rule_json_duplicate``),
batches rules in groups of ``--rules-per-case`` (default 20), and writes one
parity case per batch under ``<out>/NNNN_rule/`` containing:
  - job.json                    (ruleSetList with N derive rules)
  - inputs/suite_data.csv       (header from extracted column refs + sample rows)
  - tags.txt

This version is stricter than the original corpus cleaner: it keeps only rules
that look like valid Spark-SQL-style derive expressions, skips obviously broken
or placeholder formulas, guarantees unique output columns within each case, and
emits only full batches of ``--rules-per-case`` rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT = "derive_cases_derived_dataset.csv"
DEFAULT_OUT = "shared-tests/cases/transform"
DEFAULT_RULES_PER_CASE = 20
DEFAULT_NUM_CASES = 200

SAMPLE_VALUES = [
    "1",
    "2",
    "3",
    "10",
    "100",
    "2024-01-15",
    "abc",
    "sample@example.com",
    "2024-01-15 10:30:00",
    "A-100",
]
COLUMN_REF_RE = re.compile(r"`([^`]+)`")
ADD_FORMULA_PREFIX_RE = re.compile(r"^\s*add\s+formula\b\s*", re.IGNORECASE)
NONDETERMINISTIC_FN_RE = re.compile(
    r"\b(?:current_timestamp|current_date|current_time|current_user|"
    r"current_timezone|now|rand|random|uuid|yesterday)\s*\("
    r"|\bunix_timestamp\s*\(\s*\)",
    re.IGNORECASE,
)
BAD_TOKEN_RE = re.compile(
    r"(?:\$\{|\bDateTimeNow\s*\(|\bToday\b|;|\n\s*[A-Za-z_][A-Za-z0-9_]*\s*=)",
    re.IGNORECASE,
)
ONLY_LITERAL_TEXT_RE = re.compile(r"^[\s\w\-\.:/@#%&+,\u00A0-\uFFFF]+$")

SAFE_FUNCTION_NAMES = {
    "trim",
    "ltrim",
    "rtrim",
    "upper",
    "lower",
    "initcap",
    "proper",
    "concat",
    "concat_ws",
    "substring",
    "substr",
    "substring_index",
    "left",
    "right",
    "replace",
    "regexp_replace",
    "regexp_extract",
    "split",
    "starts_with",
    "ends_with",
    "contains",
    "if",
    "case",
    "when",
    "otherwise",
    "char",
    "index_of",
    "locate",
    "strip",
    "remove_symbols",
    "truncate_char",
    "truncate_words",
    "unbase64",
    "unhex",
    "week_of_month",
    "week_of_year",
    "week_of_year_with_year",
    "weekday",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "trunc",
    "max_date",
    "variance",
    "variance_if",
    "add",
    "add_years",
    "sub_months",
    "convert_timezone",
    "extract_date_with_format",
    "join",
    "nullifempty",
    "sentence_case",
    "regexreplace",
    "replace_regex",
    "toString",
    "mid",
    "sha",
}

POST_EXEC = {
    "inferrer": {
        "output": {
            "writeData": True,
            "oInfo": {
                "data": {"path": "file://outputs/data.csv"},
                "model": {"path": "file://outputs/model.json"},
            },
        },
        "doSchema": True,
    }
}

DS_OPTIONS = {
    "header": "true",
    "delimiter": ",",
    "quote": "\"",
    "escape": "\"",
    "charset": "UTF-8",
    "ignoreLeadingWhiteSpace": "true",
    "ignoreTrailingWhiteSpace": "true",
    "initialRowsToSkip": "0",
    "intCols": "true",
    "initSample": "true",
    "rn": "1",
}


def referenced_columns(formula: str) -> list[str]:
    """Ordered unique backtick-quoted column refs in a formula."""
    seen: dict[str, None] = {}
    for col in COLUMN_REF_RE.findall(formula or ""):
        seen.setdefault(col, None)
    return list(seen)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def balanced(formula: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    in_single = False
    in_double = False
    in_backtick = False
    escape = False

    for ch in formula:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if not in_double and not in_backtick and ch == "'":
            in_single = not in_single
            continue
        if not in_single and not in_backtick and ch == '"':
            in_double = not in_double
            continue
        if not in_single and not in_double and ch == "`":
            in_backtick = not in_backtick
            continue
        if in_single or in_double or in_backtick:
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()

    return not stack and not in_single and not in_double and not in_backtick


def has_empty_function_call(formula: str) -> bool:
    return bool(re.search(r"\b(?:upper|lower|trim|year|month|weekday|week_of_month|week_of_year|unhex)\s*\(\s*\)", formula, re.IGNORECASE))


def has_supported_shape(formula: str) -> bool:
    stripped = formula.strip()
    if not stripped:
        return False
    if BAD_TOKEN_RE.search(stripped):
        return False
    if has_empty_function_call(stripped):
        return False
    if not balanced(stripped):
        return False

    refs = referenced_columns(stripped)
    fn_names = {
        name.lower()
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
    }
    if any(name not in SAFE_FUNCTION_NAMES for name in fn_names):
        return False

    if not refs and not fn_names:
        if ONLY_LITERAL_TEXT_RE.fullmatch(stripped):
            return False

    bad_patterns = [
        r"\bwearpump\.com\b",
        r"^\s*wwweee\s*$",
        r"^\s*\{[^}]+\}\s*$",
        r"\bInterop\s+`",
        r"\bwhen\s*\([^)]*\)\s*otherwise\s*\(",
        r"\byear\s*\([^)]*\)month\s*\(",
        r"\bweek_of_year_with_year\s*\(`[^`]+`\s*\d",
    ]
    return not any(re.search(p, stripped, re.IGNORECASE | re.MULTILINE) for p in bad_patterns)


def extract_input_columns(rules: list[dict]) -> list[str]:
    produced = {r["params"]["as"] for r in rules}
    seen: dict[str, None] = {}

    def add(name: str | None) -> None:
        if name and name not in produced and name not in seen:
            seen[name] = None

    for r in rules:
        params = r["params"]
        for col in referenced_columns(params.get("formula", "") or ""):
            add(col)
        for col in params.get("groupBy") or []:
            add(col)
        for entry in params.get("orderBy") or []:
            if isinstance(entry, dict):
                add(entry.get("col"))
            elif isinstance(entry, str):
                add(entry)
    return list(seen)


def topo_sort_rules(rules: list[dict]) -> list[dict]:
    producer: dict[str, int] = {}
    for i, r in enumerate(rules):
        producer.setdefault(r["params"]["as"], i)

    PENDING, VISITING, DONE = 0, 1, 2
    state = [PENDING] * len(rules)
    out: list[dict] = []

    def visit(i: int) -> bool:
        if state[i] == DONE:
            return True
        if state[i] == VISITING:
            return False
        state[i] = VISITING
        formula = rules[i]["params"].get("formula", "") or ""
        for dep in referenced_columns(formula):
            j = producer.get(dep)
            if j is None or j == i:
                continue
            if not visit(j):
                state[i] = PENDING
                return False
        state[i] = DONE
        out.append(rules[i])
        return True

    for i in range(len(rules)):
        visit(i)
    return out


def clean_rule(corpus_rule: dict) -> dict | None:
    if corpus_rule.get("name") != "derive":
        return None
    params = corpus_rule.get("params")
    if not isinstance(params, dict):
        return None

    formula = params.get("formula")
    as_name = params.get("as")
    if not isinstance(formula, str) or not formula.strip():
        return None
    if not isinstance(as_name, str) or not as_name.strip():
        return None

    formula = formula.replace("\u00A0", " ")
    formula = ADD_FORMULA_PREFIX_RE.sub("", formula.lstrip(), count=1)
    formula = formula.strip()
    if not formula:
        return None
    if NONDETERMINISTIC_FN_RE.search(formula):
        return None
    if not has_supported_shape(formula):
        return None

    out_params: dict[str, Any] = {"formula": formula, "as": as_name.strip()}
    for k in ("type", "groupBy", "orderBy", "index"):
        if k in params:
            out_params[k] = params[k]
    return {"name": "derive", "params": out_params}


def build_job_json(batch_id: str, rules: list[dict]) -> dict:
    rules_out = [dict(r) for r in rules]
    rules_out[-1] = {**rules_out[-1], "postExec": POST_EXEC}
    return {
        "ruleSetList": [
            {
                "id": batch_id,
                "dsInfo": {
                    "dsID": "-1",
                    "alias": "DS",
                    "iInfo": {
                        "data": {
                            "path": "file://inputs/suite_data.csv",
                            "datastore_id": 1,
                            "options": DS_OPTIONS,
                        }
                    },
                    "iType": "source",
                },
                "rules": rules_out,
                "props": {"zs.jobtype": "sample"},
            }
        ],
        "props": {"zs.jobtype": "sample", "zs.manage.exceptions": "false"},
    }


def sample_value_for_column(name: str, row_idx: int) -> str:
    lower = name.lower()
    if any(token in lower for token in ("date", "time", "year", "month", "day")):
        vals = ["2024-01-15", "2024-02-20", "2024-03-25", "2024-04-30"]
        return vals[row_idx % len(vals)]
    if any(token in lower for token in ("email",)):
        vals = ["a@example.com", "b@example.com", "c@example.com", "d@example.com"]
        return vals[row_idx % len(vals)]
    if any(token in lower for token in ("phone", "tel", "mobile")):
        vals = ["1234567890", "9876543210", "5550001111", "8001112222"]
        return vals[row_idx % len(vals)]
    if any(token in lower for token in ("zip", "postal")):
        vals = ["12345", "54321", "10001", "94105"]
        return vals[row_idx % len(vals)]
    if any(token in lower for token in ("price", "amount", "value", "qty", "count", "id", "number")):
        vals = ["1", "20", "300", "4000"]
        return vals[row_idx % len(vals)]
    return SAMPLE_VALUES[row_idx % len(SAMPLE_VALUES)]


def write_csv(path: Path, columns: list[str], n_rows: int) -> None:
    if not columns:
        columns = ["_placeholder"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(columns)
        for r in range(n_rows):
            w.writerow(sample_value_for_column(col, r) for col in columns)


def write_case(
    out_root: Path,
    idx: int,
    rules: list[dict],
    sample_rows: int,
    overwrite: bool,
) -> bool:
    case_dir = out_root / f"{idx:04d}_rule"
    if case_dir.exists() and not overwrite:
        return False

    sorted_rules = topo_sort_rules(rules)
    if len(sorted_rules) != len(rules):
        return False
    if len(sorted_rules) != 20 and len(sorted_rules) != len(rules):
        return False

    case_dir.mkdir(parents=True, exist_ok=True)
    batch_id = f"derive_batch_{idx:04d}"
    columns = extract_input_columns(sorted_rules)
    write_csv(case_dir / "inputs" / "suite_data.csv", columns, sample_rows)

    job = build_job_json(batch_id, sorted_rules)
    (case_dir / "job.json").write_text(
        json.dumps(job, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (case_dir / "tags.txt").write_text(
        f"derive batch-{idx:04d} p0\n", encoding="utf-8"
    )
    return True


def stream_corpus(path: Path) -> Iterable[dict]:
    csv.field_size_limit(sys.maxsize)
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "rule_json_duplicate" not in (reader.fieldnames or []):
            sys.exit(
                f"Expected column 'rule_json_duplicate' in {path}; got: {reader.fieldnames}"
            )
        for row_idx, row in enumerate(reader, start=2):
            raw = (row.get("rule_json_duplicate") or "").strip()
            if not raw:
                continue
            try:
                rule = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"row {row_idx}: skip — JSON parse error: {e}", file=sys.stderr)
                continue
            cleaned = clean_rule(rule)
            if cleaned is None:
                continue
            key = (
                normalize_whitespace(cleaned["params"]["formula"]),
                cleaned["params"]["as"],
            )
            if key in seen:
                continue
            seen.add(key)
            yield cleaned


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    p.add_argument("--out", default=DEFAULT_OUT, type=Path)
    p.add_argument("--rules-per-case", type=int, default=DEFAULT_RULES_PER_CASE)
    p.add_argument("--num-cases", type=int, default=DEFAULT_NUM_CASES)
    p.add_argument("--sample-rows", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-idx", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=20)
    args = p.parse_args(argv)

    if not args.input.is_file():
        sys.exit(f"Input CSV not found: {args.input}")
    if args.rules_per_case != 20:
        sys.exit("This generator now enforces exactly 20 valid rules per case; pass --rules-per-case 20")
    if args.num_cases < 0:
        sys.exit("--num-cases must be >= 0")

    args.out.mkdir(parents=True, exist_ok=True)

    rule_budget: int | None = None
    if args.num_cases > 0:
        rule_budget = args.num_cases * args.rules_per_case
    if args.limit is not None:
        rule_budget = args.limit if rule_budget is None else min(rule_budget, args.limit)

    consumed = 0
    case_idx = args.start_idx
    batch: list[dict] = []
    created = 0
    skipped_existing = 0
    skipped_invalid_batch = 0

    def flush() -> None:
        nonlocal created, skipped_existing, skipped_invalid_batch, case_idx, batch
        if len(batch) != args.rules_per_case:
            batch = []
            return
        case_dir = args.out / f"{case_idx:04d}_rule"
        result = write_case(args.out, case_idx, batch, args.sample_rows, args.overwrite)
        if result:
            created += 1
            if args.progress_every > 0 and created % args.progress_every == 0:
                print(f"  ...{created} cases written (last: {case_idx:04d}_rule)")
        elif case_dir.exists():
            skipped_existing += 1
        else:
            skipped_invalid_batch += 1
        case_idx += 1
        batch = []

    for rule in stream_corpus(args.input):
        if rule_budget is not None and consumed >= rule_budget:
            break
        if args.num_cases > 0 and created >= args.num_cases:
            break

        as_name = rule["params"]["as"]
        if any(r["params"]["as"] == as_name for r in batch):
            continue

        batch.append(rule)
        consumed += 1
        if len(batch) == args.rules_per_case:
            flush()
            if args.num_cases > 0 and created >= args.num_cases:
                break

    print(
        f"consumed {consumed} valid distinct rules → {created} cases created"
        + (f", {skipped_existing} existing skipped" if skipped_existing else "")
        + (f", {skipped_invalid_batch} invalid batches dropped" if skipped_invalid_batch else "")
        + "."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
