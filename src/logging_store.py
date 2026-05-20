from __future__ import annotations

import csv
from pathlib import Path


CHAT_LOG_FIELDS = [
    "id",
    "question",
    "answer",
    "category",
    "source_used",
    "helpful",
    "created_at",
]

UNANSWERED_FIELDS = ["id", "question", "reason", "created_at"]


def ensure_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()


def append_chat_log(data_dir: Path, row: dict[str, str]) -> None:
    path = data_dir / "chat_logs.csv"
    ensure_csv(path, CHAT_LOG_FIELDS)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHAT_LOG_FIELDS)
        writer.writerow({field: row.get(field, "") for field in CHAT_LOG_FIELDS})


def append_unanswered_question(data_dir: Path, row: dict[str, str]) -> None:
    path = data_dir / "unanswered_questions.csv"
    ensure_csv(path, UNANSWERED_FIELDS)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNANSWERED_FIELDS)
        writer.writerow({field: row.get(field, "") for field in UNANSWERED_FIELDS})


def update_helpfulness(data_dir: Path, log_id: str, helpful: bool) -> None:
    path = data_dir / "chat_logs.csv"
    ensure_csv(path, CHAT_LOG_FIELDS)

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if row.get("id") == log_id:
            row["helpful"] = "true" if helpful else "false"
            break

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHAT_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
