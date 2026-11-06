from __future__ import annotations

import os
import time
import joblib
import pandas as pd

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import func

from app.database import (
    SessionLocal,
    TelemetryRecord,
    PredictionRecord,
    init_db,
)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]

# Configuración del modelo de Machine Learning
MODEL_FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[1] / "models" / "model.joblib")
    )
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL = None

class Telemetry(BaseModel):
    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)


class PredictionRequest(Telemetry):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL

    init_db()

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"No se encontró el modelo: {MODEL_PATH}. "
            "Ejecuta python ml/train_model.py"
        )

    MODEL = joblib.load(MODEL_PATH)

    print(f"Modelo cargado correctamente: {MODEL_PATH}")

    yield


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0",
    description="Cloud service for virtual robot fleet telemetry.",
    lifespan=lifespan,
)

REQUEST_COUNT = 0


def serialize_record(record: TelemetryRecord) -> dict:
    """Convertir un registro de base de datos a diccionario."""
    return {
        "robot_id": record.robot_id,
        "timestamp": record.timestamp,
        "x": record.x,
        "y": record.y,
        "speed": record.speed,
        "battery": record.battery,
        "motor_temp": record.motor_temp,
        "vibration": record.vibration,
        "motor_current": record.motor_current,
        "obstacle_distance": record.obstacle_distance,
    }


def run_prediction(sample: Telemetry) -> dict:
    """Realizar una predicción con el modelo cargado."""

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de Machine Learning no disponible."
        )

    X = pd.DataFrame([{
        feature: getattr(sample, feature)
        for feature in MODEL_FEATURES
    }], columns=MODEL_FEATURES)

    start_time = time.perf_counter()

    predicted_status = str(MODEL.predict(X)[0])
    probabilities = MODEL.predict_proba(X)[0]
    confidence = float(max(probabilities))

    inference_ms = (time.perf_counter() - start_time) * 1000

    return {
        "predicted_status": predicted_status,
        "confidence": round(confidence, 4),
        "model_version": MODEL_VERSION,
        "inference_ms": round(inference_ms, 3),
    }


def save_telemetry_and_prediction(sample: Telemetry) -> tuple[int, dict]:
    """Guardar la medición y su predicción en una misma operación."""

    prediction = run_prediction(sample)

    with SessionLocal() as db:
        # Guardar la medición original.
        record = TelemetryRecord(**sample.model_dump())
        db.add(record)
        db.flush()

        # Guardar la predicción relacionada con esa medición.
        prediction_record = PredictionRecord(
            telemetry_id=record.id,
            predicted_status=prediction["predicted_status"],
            confidence=prediction["confidence"],
            model_version=prediction["model_version"],
            inference_ms=prediction["inference_ms"],
        )

        db.add(prediction_record)
        db.commit()
        db.refresh(record)

        telemetry_id = record.id

    return telemetry_id, prediction


@app.get("/health")
def health() -> dict:
    with SessionLocal() as db:
        robot_count = (
            db.query(func.count(func.distinct(TelemetryRecord.robot_id)))
            .scalar() or 0
        )
        record_count = (
            db.query(func.count(TelemetryRecord.id)).scalar() or 0
        )

    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "robots_in_database": robot_count,
        "telemetry_records": record_count,
        "utc": datetime.now(timezone.utc).isoformat(),
    }



@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    telemetry_id, prediction = save_telemetry_and_prediction(sample)

    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "record_id": telemetry_id,
        **prediction,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "note": "Telemetry and prediction stored successfully.",
    }




@app.get("/robots")
def list_robots() -> list[dict]:
    with SessionLocal() as db:

        # Obtener la última medición de cada robot.
        latest_ids = (
            db.query(
                func.max(TelemetryRecord.id).label("latest_id")
            )
            .group_by(TelemetryRecord.robot_id)
            .subquery()
        )

        records = (
            db.query(TelemetryRecord)
            .join(
                latest_ids,
                TelemetryRecord.id == latest_ids.c.latest_id
            )
            .order_by(TelemetryRecord.robot_id)
            .all()
        )

        result = []

        for record in records:

            # Buscar la predicción asociada a esa medición.
            prediction = (
                db.query(PredictionRecord)
                .filter(
                    PredictionRecord.telemetry_id == record.id
                )
                .order_by(PredictionRecord.id.desc())
                .first()
            )

            robot = serialize_record(record)

            if prediction:
                robot.update({
                    "predicted_status": prediction.predicted_status,
                    "confidence": prediction.confidence,
                    "model_version": prediction.model_version,
                    "inference_ms": prediction.inference_ms,
                })
            else:
                robot.update({
                    "predicted_status": None,
                    "confidence": None,
                    "model_version": None,
                    "inference_ms": None,
                })

            result.append(robot)

        return result



@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:
    with SessionLocal() as db:
        records = (
            db.query(TelemetryRecord)
            .filter(TelemetryRecord.robot_id == robot_id)
            .order_by(TelemetryRecord.id.asc())
            .all()
        )

        if not records:
            raise HTTPException(
                status_code=404,
                detail="Robot not found"
            )

        return [serialize_record(record) for record in records]



@app.post("/predict")
def predict(sample: PredictionRequest) -> dict:
    telemetry_id, prediction = save_telemetry_and_prediction(sample)

    return {
        "robot_id": sample.robot_id,
        **prediction,
    }



@app.get("/metrics")
def metrics() -> dict:
    with SessionLocal() as db:
        robot_count = (
            db.query(func.count(func.distinct(TelemetryRecord.robot_id)))
            .scalar() or 0
        )
        record_count = (
            db.query(func.count(TelemetryRecord.id)).scalar() or 0
        )

    return {
        "requests_received": REQUEST_COUNT,
        "robots_in_database": robot_count,
        "telemetry_records": record_count,
        "persistence": "SQLITE_LOCAL",
        "prediction_service": "NOT_IMPLEMENTED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
