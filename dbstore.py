"""Attached company databases for the analysis agent.

SQLite files are stored on disk. Remote engines (Postgres, MySQL, …) are
opened via SQLAlchemy from a connection URL. Every query is checked to be
a single read-only SELECT/WITH before it runs.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_WRITE = re.compile(
    r"\b(insert\s+into|update\s+[A-Za-z0-9_`\"\[\]]+\s+set|delete\s+from|drop\s+(table|index|view|schema|database)|"
    r"alter\s+(table|view|database|schema)|create\s+(table|index|view|schema|database)|truncate\s+|grant\s+|revoke\s+|pragma\s+|vacuum\b|"
    r"attach\s+|detach\s+|reindex\b|copy\s+|merge\s+into)\b",
    re.I,
)
_READ_START = re.compile(r"^\s*(with|select)\b", re.I)


def _strip_sql_comments_and_strings(sql: str) -> str:
    """Remove comments and quoted string literals so keyword checks don't produce false positives."""
    # Remove block comments
    s = re.sub(r"/\*[\s\S]*?\*/", " ", sql)
    # Remove line comments
    s = re.sub(r"--[^\n]*", " ", s)
    # Replace single-quoted string literals with empty strings
    s = re.sub(r"'(?:''|\\'|[^'])*'", "''", s)
    # Replace double-quoted string literals (if used for literals)
    s = re.sub(r'"(?:""|\\"|[^"])*"', '""', s)
    return s


def is_read_only_sql(sql: str) -> str | None:
    """Return an error message if `sql` is not a single read-only statement."""
    if not sql or not sql.strip():
        return "Empty query"
    clean = _strip_sql_comments_and_strings(sql.strip())
    stripped = clean.rstrip(";").strip()
    if ";" in stripped:
        return "Multiple statements are not allowed"
    if not _READ_START.search(stripped):
        return "Only SELECT / WITH queries are allowed"
    if _WRITE.search(stripped):
        return "Write or schema-changing statements are not allowed"
    return None


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def redact_url(url: str) -> str:
    return re.sub(r"(://[^:/?#]+):([^@/]+)@", r"\1:***@", url)


class DatabaseHub:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.data_dir / "dbs"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.data_dir / "databases.json"
        self._lock = threading.RLock()
        self.connections: dict[str, dict] = {}
        self._engines: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.meta_path.exists():
            try:
                with self.meta_path.open("r", encoding="utf-8") as fh:
                    self.connections = json.load(fh)
            except Exception:
                self.connections = {}

    def _save(self) -> None:
        with self.meta_path.open("w", encoding="utf-8") as fh:
            json.dump(self.connections, fh, ensure_ascii=False, indent=2)

    def _resolve_sqlite_path(self, cid: str, meta: dict) -> Path:
        """Find the sqlite file path dynamically, handling absolute or relative paths."""
        raw_path = meta.get("path")
        if raw_path:
            p = Path(raw_path)
            if p.is_file():
                return p
            # Check files_dir by filename
            fallback = self.files_dir / p.name
            if fallback.is_file():
                return fallback
        # Check by cid
        by_cid = self.files_dir / f"{cid}.sqlite"
        if by_cid.is_file():
            return by_cid
        raise FileNotFoundError(f"Database file for connection {cid} could not be found.")

    def _public(self, cid: str, meta: dict) -> dict:
        out = {
            "id": cid,
            "name": meta.get("name") or "Database",
            "engine": meta.get("engine") or "sqlite",
            "added_at": meta.get("added_at"),
            "kind": meta.get("kind") or "file",
        }
        if meta.get("url"):
            out["url"] = redact_url(meta["url"])
        return out

    def list_connections(self) -> list[dict]:
        with self._lock:
            items = [self._public(cid, m) for cid, m in self.connections.items()]
            items.sort(key=lambda d: d.get("added_at") or 0, reverse=True)
            return items

    def get(self, cid: str) -> dict | None:
        with self._lock:
            meta = self.connections.get(cid)
            if not meta:
                return None
            return self._public(cid, meta)

    def add_sqlite_bytes(self, name: str, data: bytes) -> dict:
        if not data:
            raise ValueError("Empty database file")
        # Validate it is actually SQLite before keeping it.
        tmp = self.files_dir / f".probe-{uuid.uuid4().hex}.sqlite"
        tmp.write_bytes(data)
        try:
            conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
            try:
                conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            finally:
                conn.close()
        except sqlite3.Error as exc:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"Not a readable SQLite database: {exc}") from exc

        cid = uuid.uuid4().hex
        dest = self.files_dir / f"{cid}.sqlite"
        tmp.replace(dest)
        meta = {
            "name": name or "company.db",
            "engine": "sqlite",
            "kind": "file",
            "path": f"{cid}.sqlite",
            "added_at": time.time(),
        }
        with self._lock:
            self.connections[cid] = meta
            self._save()
        return self._public(cid, meta)

    def add_url(self, name: str, url: str) -> dict:
        url = (url or "").strip()
        if not url:
            raise ValueError("Connection string is required")
        engine_name = _engine_from_url(url)
        _probe_url(url)
        cid = uuid.uuid4().hex
        meta = {
            "name": name or engine_name,
            "engine": engine_name,
            "kind": "url",
            "url": url,
            "added_at": time.time(),
        }
        with self._lock:
            self.connections[cid] = meta
            self._save()
        return self._public(cid, meta)

    def add_sample(self) -> dict:
        cid = uuid.uuid4().hex
        dest = self.files_dir / f"{cid}.sqlite"
        seed_northline(dest)
        meta = {
            "name": "Northline (sample)",
            "engine": "sqlite",
            "kind": "sample",
            "path": f"{cid}.sqlite",
            "added_at": time.time(),
        }
        with self._lock:
            self.connections[cid] = meta
            self._save()
        return self._public(cid, meta)

    def remove(self, cid: str) -> bool:
        with self._lock:
            meta = self.connections.pop(cid, None)
            if not meta:
                return False
            if meta.get("kind") in ("file", "sample"):
                try:
                    p = self._resolve_sqlite_path(cid, meta)
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            url = meta.get("url")
            if url and url in self._engines:
                # Check if any other connection uses this url
                if not any(c.get("url") == url for c in self.connections.values()):
                    eng = self._engines.pop(url, None)
                    if eng:
                        try:
                            eng.dispose()
                        except Exception:
                            pass
            self._save()
            return True

    def _get_engine(self, url: str):
        with self._lock:
            if url not in self._engines:
                self._engines[url] = _sa_engine(url)
            return self._engines[url]

    def _connect(self, cid: str):
        meta = self.connections.get(cid)
        if not meta:
            raise KeyError(cid)
        if meta.get("kind") in ("file", "sample") or meta.get("path"):
            path = self._resolve_sqlite_path(cid, meta)
            return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return self._get_engine(meta["url"])

    def schema(self, cid: str) -> dict:
        if cid not in self.connections:
            raise KeyError(cid)
        meta = self.connections[cid]
        if meta.get("kind") in ("file", "sample") or meta.get("path"):
            path = self._resolve_sqlite_path(cid, meta)
            return _sqlite_schema(str(path))
        return _sa_schema(meta["url"])

    def schema_text(self, cid: str, max_tables: int = 40) -> str:
        spec = self.schema(cid)
        lines = [f"Database: {self.connections[cid].get('name')} ({spec.get('engine')})"]
        for table in spec.get("tables", [])[:max_tables]:
            cols = ", ".join(
                f"{c['name']} {c['type']}" + (" PK" if c.get("pk") else "")
                for c in table.get("columns", [])
            )
            rows = table.get("rows")
            suffix = f" — ~{rows} rows" if rows is not None else ""
            lines.append(f"- {table['name']} ({cols}){suffix}")
        return "\n".join(lines)

    def preview(self, cid: str, table: str, limit: int = 8) -> dict:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table or ""):
            raise ValueError("Invalid table name")
        ident = _quote_ident(table, self.connections[cid].get("engine") or "sqlite")
        return self.query(cid, f"SELECT * FROM {ident} LIMIT {int(limit)}")

    def query(self, cid: str, sql: str, row_cap: int = 200) -> dict:
        err = is_read_only_sql(sql)
        if err:
            raise ValueError(err)
        cleaned = sql.strip().rstrip(";").strip()
        if not re.search(r"\blimit\b", cleaned, re.I):
            cleaned = f"{cleaned} LIMIT {int(row_cap)}"
        meta = self.connections.get(cid)
        if not meta:
            raise KeyError(cid)
        if meta.get("kind") in ("file", "sample") or meta.get("path"):
            path = self._resolve_sqlite_path(cid, meta)
            return _sqlite_query(str(path), cleaned, row_cap)
        return _sa_query(meta["url"], cleaned, row_cap)


def _engine_from_url(url: str) -> str:
    head = url.split(":", 1)[0].lower()
    if head.startswith("postgres"):
        return "postgres"
    if head.startswith("mysql") or head.startswith("mariadb"):
        return "mysql"
    if head.startswith("sqlite"):
        return "sqlite"
    return head or "sql"


def _quote_ident(name: str, engine: str) -> str:
    if engine in {"mysql"}:
        return f"`{name}`"
    return f'"{name}"'


def _sqlite_schema(path: str) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = []
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        for (tname,) in cur.fetchall():
            cols = []
            for row in conn.execute(f"PRAGMA table_info({_quote_ident(tname, 'sqlite')})"):
                cols.append({"name": row[1], "type": row[2] or "TEXT", "pk": bool(row[5])})
            try:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM {_quote_ident(tname, 'sqlite')}"
                ).fetchone()[0]
            except sqlite3.Error:
                n = None
            tables.append({"name": tname, "columns": cols, "rows": n})
        return {"engine": "sqlite", "tables": tables}
    finally:
        conn.close()


def _sqlite_query(path: str, sql: str, row_cap: int) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(row_cap + 1)
        truncated = len(fetched) > row_cap
        rows = [[json_safe(r[c]) for c in columns] for r in fetched[:row_cap]]
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
        }
    finally:
        conn.close()


def _sa_engine(url: str):
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError(
            "SQLAlchemy is required for remote databases. pip install sqlalchemy"
        ) from exc
    return create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=0)


def _probe_url(url: str) -> None:
    engine = _sa_engine(url)
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        engine.dispose()


def _sa_connect(url: str):
    return _sa_engine(url)


def _sa_schema(url: str) -> dict:
    from sqlalchemy import inspect, text

    engine = _sa_engine(url)
    try:
        insp = inspect(engine)
        tables = []
        for tname in insp.get_table_names():
            cols = []
            pk = set(insp.get_pk_constraint(tname).get("constrained_columns") or [])
            for col in insp.get_columns(tname):
                cols.append(
                    {
                        "name": col["name"],
                        "type": str(col.get("type") or "TEXT"),
                        "pk": col["name"] in pk,
                    }
                )
            rows = None
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text(f"SELECT COUNT(*) FROM {_quote_ident(tname, _engine_from_url(url))}")).scalar()
            except Exception:
                rows = None
            tables.append({"name": tname, "columns": cols, "rows": rows})
        return {"engine": _engine_from_url(url), "tables": tables}
    finally:
        engine.dispose()


def _sa_query(url: str, sql: str, row_cap: int) -> dict:
    from sqlalchemy import text

    engine = _sa_engine(url)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            fetched = result.fetchmany(row_cap + 1)
            truncated = len(fetched) > row_cap
            rows = [[json_safe(v) for v in row] for row in fetched[:row_cap]]
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
            }
    finally:
        engine.dispose()


def seed_northline(path: Path) -> None:
    """A small fictional company warehouse for demos and first-run exploration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget INTEGER NOT NULL
            );
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                title TEXT NOT NULL,
                department_id INTEGER NOT NULL REFERENCES departments(id),
                location TEXT NOT NULL,
                salary INTEGER NOT NULL,
                hired_on TEXT NOT NULL
            );
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                industry TEXT NOT NULL,
                region TEXT NOT NULL,
                account_owner TEXT NOT NULL
            );
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                sku TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_price REAL NOT NULL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                ordered_on TEXT NOT NULL,
                status TEXT NOT NULL,
                channel TEXT NOT NULL
            );
            CREATE TABLE order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                qty INTEGER NOT NULL,
                unit_price REAL NOT NULL
            );
            CREATE TABLE refunds (
                id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(id),
                refunded_on TEXT NOT NULL,
                amount REAL NOT NULL,
                reason TEXT NOT NULL
            );
            """
        )
        departments = [
            (1, "Operations", 1_240_000),
            (2, "Sales", 980_000),
            (3, "Finance", 610_000),
            (4, "People", 420_000),
            (5, "Engineering", 1_870_000),
        ]
        employees = [
            (1, "Mara Chen", "COO", 1, "Denver", 248000, "2018-03-12"),
            (2, "Ibrahim Okonkwo", "VP Sales", 2, "Austin", 214000, "2019-07-01"),
            (3, "Elena Voss", "Controller", 3, "Denver", 176000, "2020-01-20"),
            (4, "Jonah Patel", "People Partner", 4, "Remote", 128000, "2021-04-08"),
            (5, "Sofia Alvarez", "Staff Engineer", 5, "Seattle", 198000, "2019-11-15"),
            (6, "Chris Lang", "Account Executive", 2, "Chicago", 142000, "2022-02-14"),
            (7, "Priya Nair", "Account Executive", 2, "Austin", 138000, "2022-06-01"),
            (8, "Owen Blake", "Warehouse Lead", 1, "Denver", 92000, "2017-09-30"),
            (9, "Hannah Cho", "Data Analyst", 3, "Remote", 118000, "2023-01-09"),
            (10, "Luis Romero", "Backend Engineer", 5, "Seattle", 164000, "2021-08-23"),
            (11, "Amina Farouk", "Frontend Engineer", 5, "Remote", 158000, "2022-10-03"),
            (12, "Theo Berg", "Recruiter", 4, "Denver", 98000, "2023-05-16"),
        ]
        customers = [
            (1, "Helios Transit", "Logistics", "West", "Chris Lang"),
            (2, "Cedar & Pine", "Retail", "Midwest", "Priya Nair"),
            (3, "North Harbor Health", "Healthcare", "Northeast", "Chris Lang"),
            (4, "Kite & Loom", "Retail", "West", "Priya Nair"),
            (5, "Fieldstone Energy", "Energy", "South", "Ibrahim Okonkwo"),
            (6, "Amberline Schools", "Education", "Midwest", "Priya Nair"),
            (7, "Quarry & Co", "Manufacturing", "Northeast", "Chris Lang"),
            (8, "Lumen Labs", "Technology", "West", "Ibrahim Okonkwo"),
        ]
        products = [
            (1, "NL-410", "Atlas pallet jack", "Equipment", 1840.00),
            (2, "NL-220", "Harbor tote (pack of 20)", "Supplies", 96.00),
            (3, "NL-880", "Meridian scanner", "Equipment", 2620.00),
            (4, "NL-105", "Field vest", "Apparel", 64.00),
            (5, "NL-550", "Ledger annual license", "Software", 4800.00),
            (6, "NL-330", "Northline service plan", "Services", 1200.00),
        ]
        orders = [
            (1, 1, "2025-10-03", "fulfilled", "direct"),
            (2, 1, "2025-11-18", "fulfilled", "direct"),
            (3, 2, "2025-09-12", "fulfilled", "partner"),
            (4, 3, "2025-12-02", "fulfilled", "direct"),
            (5, 3, "2026-01-14", "open", "direct"),
            (6, 4, "2025-08-22", "cancelled", "web"),
            (7, 5, "2025-10-29", "fulfilled", "direct"),
            (8, 5, "2026-02-06", "fulfilled", "direct"),
            (9, 6, "2025-11-03", "fulfilled", "partner"),
            (10, 7, "2026-01-22", "open", "direct"),
            (11, 8, "2025-12-19", "fulfilled", "web"),
            (12, 8, "2026-03-01", "fulfilled", "direct"),
            (13, 2, "2026-02-17", "fulfilled", "web"),
            (14, 4, "2026-03-11", "refunded", "web"),
            (15, 7, "2025-07-08", "fulfilled", "direct"),
        ]
        order_items = [
            (1, 1, 1, 4, 1840.00),
            (2, 1, 2, 12, 96.00),
            (3, 2, 3, 2, 2620.00),
            (4, 2, 6, 2, 1200.00),
            (5, 3, 2, 40, 96.00),
            (6, 3, 4, 20, 64.00),
            (7, 4, 5, 6, 4800.00),
            (8, 5, 5, 2, 4800.00),
            (9, 6, 4, 8, 64.00),
            (10, 7, 1, 10, 1840.00),
            (11, 7, 6, 10, 1200.00),
            (12, 8, 3, 4, 2620.00),
            (13, 9, 2, 18, 96.00),
            (14, 10, 1, 6, 1840.00),
            (15, 11, 5, 3, 4800.00),
            (16, 12, 5, 8, 4800.00),
            (17, 12, 6, 8, 1200.00),
            (18, 13, 2, 24, 96.00),
            (19, 14, 4, 30, 64.00),
            (20, 15, 3, 1, 2620.00),
            (21, 15, 6, 1, 1200.00),
        ]
        refunds = [
            (1, 14, "2026-03-20", 1920.00, "Sizing run returned after pilot"),
            (2, 6, "2025-08-24", 512.00, "Order cancelled before ship"),
            (3, 3, "2025-10-01", 320.00, "Damaged totes — partial credit"),
        ]
        conn.executemany("INSERT INTO departments VALUES (?,?,?)", departments)
        conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?)", employees)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", order_items)
        conn.executemany("INSERT INTO refunds VALUES (?,?,?,?,?)", refunds)
        conn.commit()
    finally:
        conn.close()
