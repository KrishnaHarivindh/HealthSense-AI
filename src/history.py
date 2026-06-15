from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    image_name TEXT,
    target_label TEXT NOT NULL,
    patient_age INTEGER,
    patient_gender TEXT,
    view_position TEXT,
    follow_up INTEGER,
    cnn_probability REAL,
    random_forest_probability REAL,
    svm_probability REAL,
    final_probability REAL,
    threshold REAL,
    diagnosis TEXT,
    severity TEXT
)
"""


def init_history_db(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(CREATE_TABLE_SQL)
        connection.commit()
    return path


def save_prediction_record(db_path: str | Path, record: dict) -> int:
    path = init_history_db(db_path)
    payload = {
        "created_at": record.get("created_at") or datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_name": record.get("image_name"),
        "target_label": record.get("target_label", "Pneumonia"),
        "patient_age": int(record.get("patient_age", 0)),
        "patient_gender": record.get("patient_gender", "Unknown"),
        "view_position": record.get("view_position", "Unknown"),
        "follow_up": int(record.get("follow_up", 0)),
        "cnn_probability": float(record.get("cnn_probability", 0.0)),
        "random_forest_probability": float(record.get("random_forest_probability", 0.0)),
        "svm_probability": float(record.get("svm_probability", 0.0)),
        "final_probability": float(record.get("final_probability", 0.0)),
        "threshold": float(record.get("threshold", 0.5)),
        "diagnosis": record.get("diagnosis", ""),
        "severity": record.get("severity", ""),
    }

    columns = list(payload.keys())
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO prediction_history ({', '.join(columns)}) VALUES ({placeholders})"

    with sqlite3.connect(path) as connection:
        cursor = connection.execute(sql, [payload[column] for column in columns])
        connection.commit()
        return int(cursor.lastrowid)


def load_prediction_history(db_path: str | Path, limit: int = 25) -> pd.DataFrame:
    path = init_history_db(db_path)
    query = """
        SELECT id, created_at, image_name, target_label, patient_age, patient_gender,
               view_position, follow_up, cnn_probability, random_forest_probability,
               svm_probability, final_probability, threshold, diagnosis, severity
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    """
    with sqlite3.connect(path) as connection:
        history = pd.read_sql_query(query, connection, params=[int(limit)])
    return history
