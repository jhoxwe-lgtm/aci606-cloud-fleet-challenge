
from __future__ import annotations

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
    init_db,
)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


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
    # Crear las tablas al iniciar el servicio.
    init_db()
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

    record = TelemetryRecord(**sample.model_dump())

    with SessionLocal() as db:
        db.add(record)
        db.commit()
        db.refresh(record)

    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "record_id": record.id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "note": "Telemetry stored in persistent database.",
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    with SessionLocal() as db:
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

        return [serialize_record(record) for record in records]


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
def predict(_: PredictionRequest) -> dict:
    # Pendiente: integrar el modelo real de Machine Learning.
    raise HTTPException(
        status_code=501,
        detail="/predict is not implemented yet.",
    )


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
