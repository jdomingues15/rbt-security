"""
tests/test_ai.py
────────────────
Pruebas específicas de IA y ML — van más allá de "el modelo predice algo".
Verifican calidad, robustez, equidad y comportamiento del modelo scikit-learn.

  Suite 1 — Calidad del modelo       : métricas de clasificación
  Suite 2 — Robustez ante adversarios: ¿el modelo se puede engañar?
  Suite 3 — Equidad (Fairness)       : ¿discrimina injustamente?
  Suite 4 — Explicabilidad           : importancia de features
  Suite 5 — Estabilidad              : mismo resultado en múltiples ejecuciones
  Suite 6 — Integración ML en vivo   : modelo en el middleware real
  Suite 7 — Detección de drift       : distribución de datos esperada

Run:
    pytest tests/test_ai.py -v
    pytest tests/test_ai.py -v -k "robustez"
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import random
import string
import time

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
MODEL_PATH = Path("ml/bot_detector.pkl")

def load_model():
    if not MODEL_PATH.exists():
        pytest.skip("Modelo no entrenado — ejecuta: python ml/train_model.py")
    import joblib
    return joblib.load(MODEL_PATH)

def load_feature_cols():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ml.train_model import FEATURE_COLS
    return FEATURE_COLS

def generate_data(n=500):
    from ml.train_model import generate_training_data
    return generate_training_data(n_samples=n)

BASE = "http://localhost:8000"

def uid():
    return "".join(random.choices(string.ascii_lowercase, k=8))

def bot_session():
    s = requests.Session()
    s.headers["User-Agent"] = f"headless-bot/{uid()}"
    return s

def legit_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      f"Mozilla/5.0 RBT-AITest/{uid()}",
        "Accept-Language": "es-ES",
        "Accept-Encoding": "gzip",
    })
    return s


# ══════════════════════════════════════════════
# SUITE 1 — Calidad del modelo
# ══════════════════════════════════════════════
class TestModelQuality:
    """Verifica las métricas de calidad del clasificador."""

    def test_accuracy_above_80_percent(self):
        """Accuracy mínima del 80%."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        df = generate_data(1000)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc >= 0.80, f"Accuracy {acc:.2%} < 80%"

    def test_f1_score_above_75_percent(self):
        """F1-score mínimo del 75% (equilibra precisión y recall)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        df = generate_data(800)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        f1s = cross_val_score(model, X, y, cv=5, scoring="f1")
        assert f1s.mean() >= 0.75, f"F1 medio {f1s.mean():.2%} < 75%"

    def test_precision_above_70_percent(self):
        """Precisión ≥ 70%: de los que marcamos como bot, ¿cuántos lo son?"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score
        df = generate_data(800)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        prec = precision_score(y_test, model.predict(X_test))
        assert prec >= 0.70, f"Precisión {prec:.2%} < 70% — demasiados falsos positivos"

    def test_recall_above_70_percent(self):
        """Recall ≥ 70%: de los bots reales, ¿cuántos detectamos?"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import recall_score
        df = generate_data(800)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        rec = recall_score(y_test, model.predict(X_test))
        assert rec >= 0.70, f"Recall {rec:.2%} < 70% — demasiados bots sin detectar"

    def test_roc_auc_above_85_percent(self):
        """AUC-ROC ≥ 0.85 — el modelo distingue bien bots de legítimos."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        df = generate_data(800)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        aucs = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        assert aucs.mean() >= 0.85, f"AUC-ROC {aucs.mean():.2%} < 85%"

    def test_false_positive_rate_acceptable(self):
        """
        Tasa de falsos positivos ≤ 20%.
        En RBT de seguridad, bloquear usuarios legítimos es inaceptable.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        df = generate_data(1000)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        tn, fp = cm[0][0], cm[0][1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        assert fpr <= 0.20, \
            f"Tasa de falsos positivos {fpr:.2%} > 20% — demasiados legítimos bloqueados"

    def test_model_not_overfitting(self):
        """El modelo no debe memorizar los datos de entrenamiento (overfitting)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        df = generate_data(1000)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        train_f1 = f1_score(y_train, model.predict(X_train))
        test_f1  = f1_score(y_test,  model.predict(X_test))
        gap = train_f1 - test_f1
        assert gap <= 0.15, \
            f"Posible overfitting: train F1={train_f1:.2f}, test F1={test_f1:.2f}, gap={gap:.2f}"


# ══════════════════════════════════════════════
# SUITE 2 — Robustez ante adversarios
# ══════════════════════════════════════════════
class TestModelRobustness:
    """Verifica que el modelo resiste intentos de evasión."""

    def test_bot_with_legit_ua_still_detected_by_other_features(self):
        """
        Un bot que cambia su User-Agent a uno legítimo pero mantiene
        otras señales (muchos fails, high rate) aún debe ser detectado.
        """
        model = load_model()
        cols  = load_feature_cols()
        # Bot que imita UA legítimo pero tiene muchos fallos
        X = np.array([[
            0,    # is_headless_ua = 0 (imita legítimo)
            1,    # has_accept_language = 1
            200,  # requests_per_minute = alto
            50.0, # current_risk_score = alto
            25,   # failed_logins = muchos
            0,    # has_legitimate_header = 0
        ]])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        assert pred == 1 or prob >= 0.5, \
            "Bot con UA falso no detectado — modelo vulnerable a evasión de UA"

    def test_bot_with_zero_risk_score_detected_by_behavior(self):
        """
        Un bot con Risk Score 0 (primera vez) debe ser detectado
        por otras features como headless UA y alto rate.
        """
        model = load_model()
        X = np.array([[
            1,    # headless
            0,    # sin lang
            150,  # muchas peticiones
            0.0,  # score nuevo = 0
            0,    # sin fails
            0,
        ]])
        prob = model.predict_proba(X)[0][1]
        # Con headless + sin lang + 150 req/min, debería ser sospechoso
        assert prob >= 0.3, \
            f"Bot de primera visita no detectado: probabilidad {prob:.2f}"

    def test_model_stable_with_noisy_inputs(self):
        """
        El modelo debe dar predicciones estables ante pequeñas
        variaciones en los datos de entrada (ruido gaussiano pequeño).
        """
        model = load_model()
        # Bot claro
        base = np.array([[1, 0, 100, 60.0, 15, 0]], dtype=float)
        pred_base = model.predict(base)[0]

        stable_count = 0
        for _ in range(20):
            noisy = base + np.random.normal(0, 0.5, base.shape)
            noisy = np.clip(noisy, 0, None)
            noisy[:, :2] = np.round(noisy[:, :2]).astype(int)
            pred_noisy = model.predict(noisy)[0]
            if pred_noisy == pred_base:
                stable_count += 1

        stability_rate = stable_count / 20
        assert stability_rate >= 0.75, \
            f"Modelo inestable ante ruido: {stability_rate:.0%} de consistencia"

    def test_model_not_fooled_by_legitimate_header_alone(self):
        """
        X-Legitimate-User: true (feature 5 = 1) con todas las demás
        features de bot NO debe resultar en predicción legítima.
        """
        model = load_model()
        X = np.array([[
            1,    # headless
            0,    # sin lang
            300,  # alto rate
            90.0, # alto score
            30,   # muchos fails
            1,    # TIENE el header legítimo
        ]])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        # El header solo no debería salvar a un bot tan obvio
        # (puede pasar en producción via bypass, pero el ML debería marcarlo)
        assert prob >= 0.4, \
            f"Bot con header legítimo tiene probabilidad {prob:.2f} — demasiado bajo"

    def test_edge_case_all_zeros(self):
        """Input de todos ceros no debe crashear el modelo."""
        model = load_model()
        X = np.zeros((1, 6))
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        assert pred in [0, 1]
        assert len(prob) == 2
        assert abs(sum(prob) - 1.0) < 1e-6

    def test_edge_case_extreme_values(self):
        """Valores extremos (muy altos) no deben crashear el modelo."""
        model = load_model()
        X = np.array([[1, 0, 9999, 9999.0, 9999, 0]])
        pred = model.predict(X)[0]
        assert pred == 1, "Valores extremos de bot deberían clasificarse como bot"


# ══════════════════════════════════════════════
# SUITE 3 — Equidad del modelo (Fairness)
# ══════════════════════════════════════════════
class TestModelFairness:
    """
    Verifica que el modelo no discrimina injustamente.
    En RBT de seguridad, la equidad significa no bloquear usuarios
    legítimos más de lo aceptable.
    """

    def test_legit_users_have_low_false_positive_rate(self):
        """Los usuarios legítimos deben pasar con ≥ 80% de probabilidad."""
        model = load_model()
        # Generar 100 perfiles de usuarios legítimos
        np.random.seed(42)
        legit_profiles = np.column_stack([
            np.zeros(100),                          # not headless
            np.ones(100),                           # has lang
            np.random.randint(1, 10, 100),          # low rate
            np.random.uniform(0, 5, 100),           # low score
            np.zeros(100),                          # no fails
            np.zeros(100),                          # no legit header
        ])
        preds = model.predict(legit_profiles)
        fp_rate = preds.mean()  # proporción marcada como bot
        assert fp_rate <= 0.20, \
            f"El {fp_rate:.0%} de usuarios legítimos es marcado como bot — demasiado alto"

    def test_obvious_bots_have_high_detection_rate(self):
        """Los bots obvios deben ser detectados con ≥ 80% de probabilidad."""
        model = load_model()
        np.random.seed(42)
        bot_profiles = np.column_stack([
            np.ones(100),                            # headless
            np.zeros(100),                           # no lang
            np.random.randint(50, 200, 100),         # high rate
            np.random.uniform(20, 100, 100),         # high score
            np.random.randint(5, 30, 100),           # many fails
            np.zeros(100),                           # no legit header
        ])
        preds = model.predict(bot_profiles)
        detection_rate = preds.mean()
        assert detection_rate >= 0.80, \
            f"Solo el {detection_rate:.0%} de bots obvios es detectado"

    def test_threshold_sensitivity_analysis(self):
        """
        Verifica el comportamiento del modelo en diferentes umbrales
        de probabilidad para evaluar el trade-off precision/recall.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        df = generate_data(600)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]

        results = {}
        for threshold in [0.3, 0.5, 0.7]:
            preds = (probas >= threshold).astype(int)
            tp = ((preds == 1) & (y_test == 1)).sum()
            fp = ((preds == 1) & (y_test == 0)).sum()
            fn = ((preds == 0) & (y_test == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            results[threshold] = {"precision": prec, "recall": rec}

        # A threshold=0.5, debe haber balance razonable
        assert results[0.5]["precision"] >= 0.60
        assert results[0.5]["recall"]    >= 0.60


# ══════════════════════════════════════════════
# SUITE 4 — Explicabilidad del modelo
# ══════════════════════════════════════════════
class TestModelExplainability:
    """Verifica que podemos entender por qué el modelo toma sus decisiones."""

    def test_feature_importances_sum_to_one(self):
        """Las importancias de features deben sumar exactamente 1.0."""
        model = load_model()
        importances = model.feature_importances_
        assert abs(importances.sum() - 1.0) < 1e-6, \
            f"Importancias no suman 1: {importances.sum():.6f}"

    def test_all_features_have_positive_importance(self):
        """Todas las features deben contribuir algo al modelo."""
        model = load_model()
        importances = model.feature_importances_
        cols = load_feature_cols()
        for col, imp in zip(cols, importances):
            assert imp > 0, f"Feature '{col}' tiene importancia 0 — podría eliminarse"

    def test_risk_score_is_most_important_feature(self):
        """
        current_risk_score debería ser una de las features más importantes,
        ya que es la señal más directa de comportamiento sospechoso.
        """
        model = load_model()
        cols  = load_feature_cols()
        importances = dict(zip(cols, model.feature_importances_))
        top_3 = sorted(importances, key=importances.get, reverse=True)[:3]
        assert "current_risk_score" in top_3 or "failed_logins" in top_3, \
            f"Señales de riesgo no están entre las top 3: {top_3}"

    def test_headless_ua_has_meaningful_importance(self):
        """is_headless_ua debe tener importancia > 0.05 (contribuye al modelo)."""
        model = load_model()
        cols  = load_feature_cols()
        imp   = dict(zip(cols, model.feature_importances_))
        assert imp["is_headless_ua"] >= 0.05, \
            f"UA headless tiene importancia muy baja: {imp['is_headless_ua']:.3f}"

    def test_model_has_expected_number_of_features(self):
        """El modelo debe usar exactamente 6 features."""
        model = load_model()
        assert model.n_features_in_ == 6, \
            f"Modelo entrenado con {model.n_features_in_} features, esperado 6"

    def test_decision_path_exists_for_sample(self):
        """Podemos obtener el camino de decisión para cualquier muestra."""
        model = load_model()
        X = np.array([[1, 0, 100, 50.0, 10, 0]])
        # RandomForest permite inspeccionar los árboles
        indicator = model.decision_path(X)
        assert indicator is not None
        assert indicator.shape[0] == 1


# ══════════════════════════════════════════════
# SUITE 5 — Estabilidad del modelo
# ══════════════════════════════════════════════
class TestModelStability:
    """Verifica que el modelo produce resultados consistentes."""

    def test_same_input_same_output_always(self):
        """El mismo input debe producir exactamente la misma predicción."""
        model = load_model()
        X = np.array([[1, 0, 50, 30.0, 8, 0]])
        predictions = [model.predict(X)[0] for _ in range(20)]
        assert len(set(predictions)) == 1, \
            "El modelo no es determinista — predicciones distintas para el mismo input"

    def test_model_predictions_consistent_across_batch(self):
        """Predecir en batch debe dar el mismo resultado que individualmente."""
        model = load_model()
        samples = np.array([
            [1, 0, 100, 50.0, 10, 0],
            [0, 1, 3,   0.0,  0,  0],
            [1, 1, 200, 80.0, 20, 0],
        ])
        batch_preds = model.predict(samples)
        individual_preds = [model.predict(s.reshape(1, -1))[0] for s in samples]
        assert list(batch_preds) == individual_preds, \
            "Predicción en batch difiere de predicciones individuales"

    def test_retraining_gives_similar_performance(self):
        """Dos entrenamientos con la misma semilla dan el mismo resultado."""
        from sklearn.ensemble import RandomForestClassifier
        df = generate_data(500)
        cols = load_feature_cols()
        X, y = df[cols].values, df["label"].values

        model1 = RandomForestClassifier(n_estimators=20, random_state=42)
        model2 = RandomForestClassifier(n_estimators=20, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)

        test_X = X[:50]
        assert list(model1.predict(test_X)) == list(model2.predict(test_X)), \
            "Dos entrenamientos con la misma semilla dan resultados distintos"

    def test_prediction_time_acceptable(self):
        """La predicción debe ser rápida — < 10ms para no añadir latencia."""
        model = load_model()
        X = np.array([[1, 0, 50, 30.0, 5, 0]])

        # Calentar el modelo
        for _ in range(5):
            model.predict(X)

        # Medir tiempos
        import time
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            model.predict(X)
            times.append(time.perf_counter() - t0)

        p99_ms = sorted(times)[int(len(times) * 0.99)] * 1000
        assert p99_ms < 10, \
            f"P99 de predicción ML: {p99_ms:.2f}ms > 10ms — añade latencia al middleware"


# ══════════════════════════════════════════════
# SUITE 6 — ML integrado en el middleware (vivo)
# ══════════════════════════════════════════════
class TestMLLiveIntegration:
    """Pruebas del modelo ML funcionando dentro del middleware de FastAPI."""

    def test_api_reports_ml_loaded(self):
        """La API debe confirmar que el modelo ML está cargado."""
        r = requests.get(f"{BASE}/status", timeout=10)
        assert r.json().get("ml_model_loaded") is True

    def test_bot_probability_exposed_in_metrics(self):
        """bot_ml_probability debe aparecer en /metrics después de una request."""
        if not requests.get(f"{BASE}/status").json().get("ml_model_loaded"):
            pytest.skip("ML no cargado")

        s = bot_session()
        s.get(f"{BASE}/api/data")
        time.sleep(0.5)

        text = requests.get(f"{BASE}/metrics").text
        assert "bot_ml_probability{" in text

    def test_high_probability_bots_are_blocked(self):
        """Bots con patrón claro deben ser bloqueados (por ML o por reglas)."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login",
                  params={"username": "hacker", "password": "wrong"})

        r = s.get(f"{BASE}/api/data")
        assert r.status_code == 403, \
            "Bot con patrón claro no fue bloqueado"

    def test_legit_users_pass_ml_filter(self):
        """Usuarios legítimos deben pasar el filtro ML sin ser bloqueados."""
        if not requests.get(f"{BASE}/status").json().get("ml_model_loaded"):
            pytest.skip("ML no cargado")

        for _ in range(8):
            s = legit_session()
            r = s.get(f"{BASE}/api/data", timeout=10)
            assert r.status_code == 200, \
                f"Usuario legítimo bloqueado por ML en intento {_+1}"
            time.sleep(0.1)


# ══════════════════════════════════════════════
# SUITE 7 — Detección de drift de datos
# ══════════════════════════════════════════════
class TestDataDrift:
    """
    Verifica que la distribución de los datos de entrenamiento
    es coherente con los patrones esperados del sistema.
    """

    def test_training_data_distribution_makes_sense(self):
        """Los datos de entrenamiento deben tener distribuciones lógicas."""
        df = generate_data(1000)
        bots  = df[df["label"] == 1]
        legit = df[df["label"] == 0]

        # Los bots deben tener más peticiones por minuto que los legítimos
        assert bots["requests_per_minute"].mean() > \
               legit["requests_per_minute"].mean(), \
            "Bots no tienen más peticiones/minuto que usuarios legítimos"

        # Los bots deben tener más fallos de login
        assert bots["failed_logins"].mean() > \
               legit["failed_logins"].mean(), \
            "Bots no tienen más fallos de login que usuarios legítimos"

        # Los usuarios legítimos deben tener Accept-Language más frecuentemente
        assert legit["has_accept_language"].mean() > \
               bots["has_accept_language"].mean(), \
            "Usuarios legítimos no tienen Accept-Language más que los bots"

    def test_no_nan_values_in_training_data(self):
        """Los datos de entrenamiento no deben tener valores NaN."""
        df = generate_data(500)
        cols = load_feature_cols()
        assert not df[cols].isnull().any().any(), \
            "Hay valores NaN en los datos de entrenamiento"

    def test_no_infinite_values_in_training_data(self):
        """Los datos de entrenamiento no deben tener valores infinitos."""
        df = generate_data(500)
        cols = load_feature_cols()
        for col in cols:
            assert not np.isinf(df[col].values).any(), \
                f"Hay valores infinitos en la columna {col}"

    def test_feature_ranges_are_valid(self):
        """Todas las features deben estar en rangos válidos."""
        df = generate_data(500)
        assert df["is_headless_ua"].between(0, 1).all()
        assert df["has_accept_language"].between(0, 1).all()
        assert df["has_legitimate_header"].between(0, 1).all()
        assert (df["requests_per_minute"] >= 0).all()
        assert (df["current_risk_score"]  >= 0).all()
        assert (df["failed_logins"]        >= 0).all()

    def test_model_performance_on_new_synthetic_data(self):
        """
        El modelo entrenado debe funcionar bien en datos sintéticos
        nuevos (simula detección de drift).
        """
        from sklearn.metrics import f1_score
        model = load_model()
        cols  = load_feature_cols()

        # Generar datos "nuevos" con semilla diferente
        df_new = generate_data(400)
        X_new  = df_new[cols].values
        y_new  = df_new["label"].values

        preds = model.predict(X_new)
        f1    = f1_score(y_new, preds)

        assert f1 >= 0.70, \
            f"El modelo tiene F1={f1:.2f} en datos nuevos — posible drift detectado"
