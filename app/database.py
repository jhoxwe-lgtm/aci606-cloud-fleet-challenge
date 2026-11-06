
import os
from pathlib import Path
import time

from sqlalchemy import Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


# Base de datos local para desarrollo.
# En Render utilizaremos DATABASE_URL de Neon.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/fleet.db"
)

# Adaptar la URL de PostgreSQL para utilizar psycopg 3.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://", "postgresql+psycopg://", 1
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

# Crear la carpeta local si todavía no existe.
if DATABASE_URL.startswith("sqlite"):
    Path("data").mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

# Crear el motor de conexión.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args
)


class Base(DeclarativeBase):
    pass


class TelemetryRecord(Base):
    """Tabla que almacena las mediciones de cada robot."""

    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    robot_id: Mapped[str] = mapped_column(String(20), index=True)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    speed: Mapped[float] = mapped_column(Float, nullable=False)
    battery: Mapped[float] = mapped_column(Float, nullable=False)
    motor_temp: Mapped[float] = mapped_column(Float, nullable=False)
    vibration: Mapped[float] = mapped_column(Float, nullable=False)
    motor_current: Mapped[float] = mapped_column(Float, nullable=False)
    obstacle_distance: Mapped[float] = mapped_column(Float, nullable=False)


class PredictionRecord(Base):
    """Tabla que almacena las predicciones del modelo."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    # Relaciona la predicción con su medición original.
    telemetry_id: Mapped[int] = mapped_column(
        ForeignKey("telemetry.id"),
        nullable=False,
        index=True
    )

    predicted_status: Mapped[str] = mapped_column(
        String(10), nullable=False
    )

    confidence: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    model_version: Mapped[str] = mapped_column(
        String(20), nullable=False
    )

    inference_ms: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    created_at: Mapped[float] = mapped_column(
        Float, nullable=False, default=time.time
    )

# Crear sesiones para consultar y guardar mediciones.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)


def init_db() -> None:
    """Crear las tablas si todavía no existen."""
    Base.metadata.create_all(bind=engine)
