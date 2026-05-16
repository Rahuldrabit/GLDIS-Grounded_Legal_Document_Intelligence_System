"""
One-time migration: add draft_type column to drafts table.
Safe to run multiple times — skips if the column already exists.

Usage:
    python scripts/migrate_add_draft_type.py
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings

settings = get_settings()
db_url = settings.database_url

# Strip SQLite prefix(es)
for prefix in ("sqlite:///./", "sqlite:///", "sqlite://"):
    if db_url.startswith(prefix):
        db_path = db_url[len(prefix):]
        break
else:
    db_path = db_url

# Resolve relative path from project root
db_path = Path(__file__).parent.parent / db_path
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()

cur.execute("PRAGMA table_info(drafts)")
existing_cols = [row[1] for row in cur.fetchall()]

if "draft_type" not in existing_cols:
    cur.execute(
        "ALTER TABLE drafts ADD COLUMN draft_type TEXT DEFAULT 'case_summary_memo'"
    )
    conn.commit()
    print("Migration complete: draft_type column added to drafts table.")
else:
    print("Column draft_type already exists; skipping.")

conn.close()
