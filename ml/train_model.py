
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Archivos
DATA = Path("data/training.csv")
MODEL_OUT = Path("models/model.joblib")
REPORT_OUT = Path("models/training_report.json")

# Variables que utilizará el modelo
FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]

TARGET = "status"
GROUP = "fleet_id"

CLASS_ORDER = ["NORMAL", "WARNING", "CRITICAL"]
RANDOM_STATE = 42


def main() -> None:

    # 1. Comprobar dataset
    if not DATA.exists():
        raise SystemExit(
            "Dataset no encontrado. Ejecuta primero "
            "python simulator/generate_dataset.py"
        )

    df = pd.read_csv(DATA)

    required_columns = FEATURES + [TARGET, GROUP]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise SystemExit(f"Faltan columnas: {missing_columns}")

    if df[required_columns].isnull().any().any():
        raise SystemExit(
            "Hay valores faltantes. Revisa el dataset antes de entrenar."
        )

    print("=" * 60)
    print("ACI606 - CLOUD FLEET CHALLENGE")
    print("ENTRENAMIENTO DE MACHINE LEARNING")
    print("=" * 60)

    print(f"\nRegistros: {len(df)}")
    print(f"Flotas: {df[GROUP].nunique()}")

    print("\nDistribución de clases:")
    print(df[TARGET].value_counts())

    # 2. Preparar variables
    X = df[FEATURES].copy()
    y = df[TARGET].astype(str)
    groups = df[GROUP].astype(str)

    if set(y.unique()) != set(CLASS_ORDER):
        raise SystemExit(
            "El dataset debe contener NORMAL, WARNING y CRITICAL."
        )

    # 3. Separar por flotas completas
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    train_fleets = sorted(groups_train.unique())
    test_fleets = sorted(groups.iloc[test_idx].unique())

    if set(train_fleets).intersection(test_fleets):
        raise RuntimeError("Error: hay flotas compartidas entre Train y Test.")

    if set(y_train.unique()) != set(CLASS_ORDER):
        raise SystemExit("Train no contiene todas las clases.")

    if set(y_test.unique()) != set(CLASS_ORDER):
        raise SystemExit("Test no contiene todas las clases.")

    print("\nSeparación de flotas:")
    print("Train:", train_fleets)
    print("Test:", test_fleets)

    print(f"\nRegistros Train: {len(X_train)}")
    print(f"Registros Test: {len(X_test)}")

    # 4. Crear los clasificadores
    models = {
        "Regresion Logistica": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    # 5. Comparar modelos mediante validación cruzada por flotas
    n_splits = min(5, len(train_fleets))
    cv = GroupKFold(n_splits=n_splits)

    results = []

    print("\n" + "=" * 60)
    print("COMPARACION DE MODELOS")
    print("=" * 60)

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )

        mean_score = scores.mean()
        std_score = scores.std()

        results.append({
            "model": name,
            "mean_f1_macro": mean_score,
            "std_f1_macro": std_score,
        })

        print(f"\nModelo: {name}")
        print(f"F1 macro por fold: {scores.round(4)}")
        print(f"F1 macro promedio: {mean_score:.4f}")
        print(f"Desviación: {std_score:.4f}")

    # 6. Seleccionar el modelo según CV, no según Test
    results_df = pd.DataFrame(results).sort_values(
        "mean_f1_macro",
        ascending=False,
    )

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]

    print("\nRanking de validación cruzada:")
    print(results_df.to_string(index=False))

    print(f"\nModelo seleccionado: {best_name}")

    # 7. Entrenar el modelo elegido con todo Train
    best_model.fit(X_train, y_train)

    # 8. Evaluar una sola vez sobre las flotas no vistas
    y_pred = best_model.predict(X_test)

    test_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
    )

    print("\n" + "=" * 60)
    print("EVALUACION FINAL EN TEST")
    print("=" * 60)

    print(f"\nModelo: {best_name}")
    print(f"F1 macro Test: {test_f1:.4f}")

    print("\nMatriz de confusion:")
    print(pd.DataFrame(
        matrix,
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    ))

    print("\nReporte por clase:")
    print(classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    ))

    # 9. Guardar el Pipeline completo
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_OUT)

    report = {
        "model_name": best_name,
        "model_version": "v1",
        "features": FEATURES,
        "target": TARGET,
        "class_order": CLASS_ORDER,
        "train_fleets": train_fleets,
        "test_fleets": test_fleets,
        "cv_f1_macro": float(results_df.iloc[0]["mean_f1_macro"]),
        "test_f1_macro": float(test_f1),
        "confusion_matrix": matrix.tolist(),
            "model_comparison": [
        {
            "model": result["model"],
            "cv_f1_macro_mean": float(result["mean_f1_macro"]),
            "cv_f1_macro_std": float(result["std_f1_macro"]),
        }
        for result in results
    ],
    }

    with REPORT_OUT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)

    print(f"Modelo guardado: {MODEL_OUT}")
    print(f"Reporte guardado: {REPORT_OUT}")


if __name__ == "__main__":
    main()
