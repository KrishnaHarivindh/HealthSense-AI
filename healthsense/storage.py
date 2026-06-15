from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from healthsense.config import DATABASE_PATH, STORAGE_ROOT, ensure_platform_dirs


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    ensure_platform_dirs()
    resolved = db_path or DATABASE_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(resolved)


def init_storage(db_path: Path | None = None) -> Path:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                module TEXT NOT NULL,
                created_at TEXT NOT NULL,
                input_summary_json TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                report_path TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT NOT NULL,
                run_mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                artifact_version TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics_path TEXT,
                config_json TEXT
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                session_id INTEGER,
                created_at TEXT NOT NULL,
                report_type TEXT NOT NULL,
                path TEXT NOT NULL,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id),
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )
        connection.commit()
    return db_path or DATABASE_PATH


def create_session(name: str, db_path: Path | None = None) -> int:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO sessions (name, created_at) VALUES (?, ?)",
            (name, created_at),
        )
        connection.commit()
        return int(cursor.lastrowid)


def record_prediction(
    module: str,
    input_summary: dict[str, Any],
    prediction_payload: dict[str, Any],
    model_version: str,
    session_id: int | None = None,
    report_path: str | None = None,
    db_path: Path | None = None,
) -> int:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (session_id, module, created_at, input_summary_json, prediction_json, model_version, report_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                module,
                created_at,
                json.dumps(input_summary),
                json.dumps(prediction_payload),
                model_version,
                report_path,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def record_experiment(
    module: str,
    run_mode: str,
    artifact_version: str,
    status: str,
    metrics_path: str | None,
    config: dict[str, Any],
    db_path: Path | None = None,
) -> int:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (module, run_mode, created_at, artifact_version, status, metrics_path, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                module,
                run_mode,
                created_at,
                artifact_version,
                status,
                metrics_path,
                json.dumps(config),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def record_report(
    report_type: str,
    path: str,
    prediction_id: int | None = None,
    session_id: int | None = None,
    db_path: Path | None = None,
) -> int:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO reports (prediction_id, session_id, created_at, report_type, path) VALUES (?, ?, ?, ?, ?)",
            (prediction_id, session_id, created_at, report_type, path),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_prediction_history(limit: int = 50, db_path: Path | None = None) -> pd.DataFrame:
    init_storage(db_path)
    with _connect(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT p.id, p.session_id, s.name AS session_name, p.module, p.created_at, p.model_version, p.report_path,
                   p.input_summary_json, p.prediction_json
            FROM predictions p
            LEFT JOIN sessions s ON s.id = p.session_id
            ORDER BY p.id DESC
            LIMIT ?
            """,
            connection,
            params=[int(limit)],
        )


def list_experiments(limit: int = 50, db_path: Path | None = None) -> pd.DataFrame:
    init_storage(db_path)
    with _connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?",
            connection,
            params=[int(limit)],
        )


def get_session_predictions(session_id: int, db_path: Path | None = None) -> pd.DataFrame:
    init_storage(db_path)
    with _connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM predictions WHERE session_id = ? ORDER BY id ASC",
            connection,
            params=[int(session_id)],
        )
