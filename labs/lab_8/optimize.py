import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Ruta del dataset
DATA_PATH = "water_potability.csv"

# Nombre del experimento en MLflow
EXPERIMENT_NAME = "optuna_xgboost_water_potability"


def get_best_model(experiment_id):
    # Obtener todos los runs del experimento
    runs = mlflow.search_runs(experiment_id)
    # Seleccionar el run con mejor F1
    best_model_id = runs.sort_values(
        "metrics.valid_f1",
        ascending=False,
    )["run_id"].iloc[0]
    # Cargar el modelo asociado a ese run
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model


def optimize_model():
    # Desactivar autologging para evitar runs automáticos
    mlflow.autolog(disable=True)
    mlflow.xgboost.autolog(disable=True)
    # Crear carpeta de salida
    Path("models").mkdir(exist_ok=True)
    # Cargar dataset
    df = pd.read_csv(DATA_PATH)
    # Separar variables predictoras y objetivo
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    # Dividir datos en entrenamiento y validación
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    # Crear o seleccionar experimento
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    def objective(trial):
        # Espacio de búsqueda de hiperparámetros
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        }
        # Nombre descriptivo del run
        run_name = f"XGBoost lr={params['learning_rate']:.3f}, depth={params['max_depth']}"
        # Iniciar run en MLflow
        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        ):
            # Forzar nombre y metadata del run
            mlflow.set_tag("mlflow.runName", run_name)
            mlflow.set_tag("model", "XGBoost")
            mlflow.set_tag("optimizer", "Optuna")
            # Crear modelo
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
            # Entrenar modelo
            model.fit(X_train, y_train)
            # Realizar predicciones
            y_pred = model.predict(X_valid)
            # Calcular F1-score
            valid_f1 = f1_score(y_valid, y_pred)
            # Registrar parámetros y métrica
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", valid_f1)
            # Guardar modelo en MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )
        return valid_f1

    # Crear estudio Optuna
    study = optuna.create_study(
        direction="maximize",
        study_name="optuna_xgboost_water_potability_study",
    )
    # Ejecutar optimización
    study.optimize(objective, n_trials=10)
    # Guardar mejores parámetros
    pd.DataFrame([study.best_params]).to_csv(
        "models/best_params.csv",
        index=False,
    )
    # Recuperar mejor modelo registrado
    best_model = get_best_model(experiment.experiment_id)
    # Serializar mejor modelo
    with open("models/best_model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    # Guardar versiones utilizadas
    versions = {
        "mlflow": mlflow.__version__,
        "optuna": optuna.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    pd.DataFrame.from_dict(
        versions,
        orient="index",
        columns=["version"],
    ).to_csv(
        "models/library_versions.csv",
        index_label="library",
    )
    return best_model


def main():
    optimize_model()


if __name__ == "__main__":
    main()
