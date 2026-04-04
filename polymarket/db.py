import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import Market, OrderBook, OrderLevel

DEFAULT_DB = "markets.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    condition_id TEXT NOT NULL UNIQUE,
    slug TEXT,
    description TEXT,
    outcomes TEXT NOT NULL,
    outcome_prices TEXT NOT NULL,
    clob_token_ids TEXT NOT NULL,
    volume REAL,
    volume_24hr REAL,
    liquidity REAL,
    active INTEGER,
    closed INTEGER,
    end_date TEXT,
    spread REAL,
    best_bid REAL,
    best_ask REAL,
    last_trade_price REAL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_condition_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    bids TEXT NOT NULL,
    asks TEXT NOT NULL,
    midpoint REAL,
    spread REAL,
    last_trade_price REAL,
    timestamp TEXT,
    fetched_at TEXT NOT NULL,
    FOREIGN KEY (market_condition_id) REFERENCES markets(condition_id)
);

CREATE INDEX IF NOT EXISTS idx_markets_volume ON markets(volume DESC);
CREATE INDEX IF NOT EXISTS idx_markets_liquidity ON markets(liquidity DESC);
CREATE INDEX IF NOT EXISTS idx_order_books_condition ON order_books(market_condition_id);
CREATE INDEX IF NOT EXISTS idx_order_books_fetched ON order_books(fetched_at DESC);
"""


def init_db(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_markets(conn: sqlite3.Connection, markets: list[Market]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO markets
        (id, question, condition_id, slug, description, outcomes, outcome_prices,
         clob_token_ids, volume, volume_24hr, liquidity, active, closed, end_date,
         spread, best_bid, best_ask, last_trade_price, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                m.id,
                m.question,
                m.condition_id,
                m.slug,
                m.description,
                json.dumps(m.outcomes),
                json.dumps(m.outcome_prices),
                json.dumps(m.clob_token_ids),
                m.volume,
                m.volume_24hr,
                m.liquidity,
                int(m.active),
                int(m.closed),
                m.end_date,
                m.spread,
                m.best_bid,
                m.best_ask,
                m.last_trade_price,
                m.fetched_at.isoformat(),
            )
            for m in markets
        ],
    )
    conn.commit()


def insert_order_books(conn: sqlite3.Connection, books: list[OrderBook]) -> None:
    conn.executemany(
        """INSERT INTO order_books
        (market_condition_id, asset_id, outcome, bids, asks, midpoint, spread,
         last_trade_price, timestamp, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                b.market_condition_id,
                b.asset_id,
                b.outcome,
                json.dumps([{"price": l.price, "size": l.size} for l in b.bids]),
                json.dumps([{"price": l.price, "size": l.size} for l in b.asks]),
                b.midpoint,
                b.spread,
                b.last_trade_price,
                b.timestamp,
                b.fetched_at.isoformat(),
            )
            for b in books
        ],
    )
    conn.commit()


def _row_to_market(row: sqlite3.Row) -> Market:
    return Market(
        id=row["id"],
        question=row["question"],
        condition_id=row["condition_id"],
        slug=row["slug"],
        description=row["description"],
        outcomes=json.loads(row["outcomes"]),
        outcome_prices=json.loads(row["outcome_prices"]),
        clob_token_ids=json.loads(row["clob_token_ids"]),
        volume=row["volume"],
        volume_24hr=row["volume_24hr"],
        liquidity=row["liquidity"],
        active=bool(row["active"]),
        closed=bool(row["closed"]),
        end_date=row["end_date"],
        spread=row["spread"],
        best_bid=row["best_bid"],
        best_ask=row["best_ask"],
        last_trade_price=row["last_trade_price"],
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
    )


def get_top_markets(
    conn: sqlite3.Connection,
    limit: int = 20,
    order_by: str = "volume",
) -> list[Market]:
    valid_cols = {"volume", "liquidity", "volume_24hr"}
    col = order_by if order_by in valid_cols else "volume"
    rows = conn.execute(
        f"SELECT * FROM markets WHERE active = 1 AND closed = 0 ORDER BY {col} DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_market(r) for r in rows]


def get_latest_order_book(
    conn: sqlite3.Connection, condition_id: str, outcome: str
) -> OrderBook | None:
    row = conn.execute(
        """SELECT * FROM order_books
        WHERE market_condition_id = ? AND outcome = ?
        ORDER BY fetched_at DESC LIMIT 1""",
        (condition_id, outcome),
    ).fetchone()
    if row is None:
        return None
    return OrderBook(
        market_condition_id=row["market_condition_id"],
        asset_id=row["asset_id"],
        outcome=row["outcome"],
        bids=[OrderLevel(**l) for l in json.loads(row["bids"])],
        asks=[OrderLevel(**l) for l in json.loads(row["asks"])],
        midpoint=row["midpoint"],
        spread=row["spread"],
        last_trade_price=row["last_trade_price"],
        timestamp=row["timestamp"],
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
    )
