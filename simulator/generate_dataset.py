
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from core import make_robot, step_robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate labelled data from independent robot fleets."
    )
    parser.add_argument("--samples", type=int, default=3000,
                        help="Total number of observations.")
    parser.add_argument("--robots", type=int, default=8,
                        help="Robots per fleet.")
    parser.add_argument("--fleets", type=int, default=12,
                        help="Number of independent fleets.")
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--output", default="data/training.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.samples < args.fleets:
        raise SystemExit("Samples must be greater than or equal to fleets.")

    if args.robots < 1 or args.fleets < 2:
        raise SystemExit("Use at least 1 robot and 2 independent fleets.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # Repartir las muestras de forma equilibrada entre las flotas.
    base_samples, remainder = divmod(args.samples, args.fleets)

    for fleet_index in range(args.fleets):

        fleet_id = f"F{fleet_index + 1:02d}"

        # Cada flota tiene su propio generador aleatorio
        # y sus propios robots independientes.
        rng = random.Random(args.seed + fleet_index)

        robots = [
            make_robot(index + 1, rng)
            for index in range(args.robots)
            
        ]
        
        # Variar la batería inicial en el dataset de entrenamiento.
        # Así incluimos robots con batería alta, media y baja.
        for robot in robots:
            robot.battery = rng.uniform(0.0, 100.0)

        fleet_samples = base_samples + (
            1 if fleet_index < remainder else 0
        )

        for sample_index in range(fleet_samples):

            robot = robots[sample_index % args.robots]

            sample, status = step_robot(robot, rng)

            rows.append({
                "fleet_id": fleet_id,
                **sample,
                "status": status,
            })

    # Guardar el dataset.
    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Mostrar resultados.
    class_counts = Counter(row["status"] for row in rows)
    fleet_counts = Counter(row["fleet_id"] for row in rows)

    print("\nDataset generado correctamente.")
    print(f"Archivo: {output_path}")
    print(f"Total de registros: {len(rows)}")
    print(f"Flotas independientes: {args.fleets}")
    print(f"Robots por flota: {args.robots}")

    print("\nDistribución de clases:")
    for status in ["NORMAL", "WARNING", "CRITICAL"]:
        print(f"{status}: {class_counts.get(status, 0)}")

    print("\nRegistros por flota:")
    for fleet_id, count in sorted(fleet_counts.items()):
        print(f"{fleet_id}: {count}")


if __name__ == "__main__":
    main()
