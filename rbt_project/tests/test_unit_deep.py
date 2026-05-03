"""
tests/test_unit_deep.py
───────────────────────
Pruebas unitarias profundas — cada función interna del proyecto
probada en aislamiento total usando mocks.

No necesita Docker ni API levantada.

Run:
    pytest tests/test_unit_deep.py -v
    pytest tests/test_unit_deep.py -v -k "fingerprint"
"""

import pytest
import hashlib
import numpy as np
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fake_request(ua="Mozilla/5.0", lang="en-US", encoding="gzip",
                 ip="1.2.3.4", path="/api/data",
                 extra_headers=None):
    """Construye un Request falso sin levantar FastAPI."""
    req = MagicMock()
    headers = {
        "User-Agent":      ua,
        "Accept-Language": lang,
        "Accept-Encoding": encoding,
    }
    if extra_headers:
        headers.update(extra_headers)
    req.headers = headers
    req.client      = MagicMock()
    req.client.host = ip
    req.url         = MagicMock()
    req.url.path    = path
    return req


def fake_redis(risk=0.0, rate_count=5, fails=0):
    """Redis mock con respuestas predefinidas."""
    r = MagicMock()
    r.get  = MagicMock(side_effect=lambda k: (
        str(risk)  if k.startswith("risk:")  else
        str(fails) if k.startswith("fails:") else None
    ))
    r.zcard             = MagicMock(return_value=rate_count)
    r.zadd              = MagicMock()
    r.zremrangebyscore  = MagicMock()
    r.expire            = MagicMock()
    r.set               = MagicMock()
    r.incr              = MagicMock()
    r.delete            = MagicMock()
    return r


# ══════════════════════════════════════════════
# SUITE 1 — get_fingerprint()
# ══════════════════════════════════════════════
class TestGetFingerprint:
    """Verifica la generación de fingerprints únicos por headers."""

    def test_same_headers_produce_same_fingerprint(self):
        from main import get_fingerprint
        r1 = fake_request(ua="Chrome/99", lang="en", encoding="gzip")
        r2 = fake_request(ua="Chrome/99", lang="en", encoding="gzip")
        assert get_fingerprint(r1) == get_fingerprint(r2)

    def test_different_ua_produces_different_fingerprint(self):
        from main import get_fingerprint
        assert get_fingerprint(fake_request(ua="Chrome/99")) != \
               get_fingerprint(fake_request(ua="Firefox/100"))

    def test_different_language_produces_different_fingerprint(self):
        from main import get_fingerprint
        assert get_fingerprint(fake_request(lang="en-US")) != \
               get_fingerprint(fake_request(lang="es-ES"))

    def test_fingerprint_is_valid_md5_hex(self):
        from main import get_fingerprint
        fp = get_fingerprint(fake_request())
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)

    def test_missing_headers_fallback_to_unknown(self):
        from main import get_fingerprint
        req = MagicMock()
        req.headers = {}
        expected = hashlib.md5("unknown|unknown|unknown".encode()).hexdigest()
        assert get_fingerprint(req) == expected

    def test_fingerprint_is_deterministic_across_calls(self):
        from main import get_fingerprint
        req = fake_request(ua="TestBot/1.0", lang="fr-FR")
        fps = [get_fingerprint(req) for _ in range(10)]
        assert len(set(fps)) == 1, "Fingerprint no es determinista"

    def test_encoding_change_changes_fingerprint(self):
        from main import get_fingerprint
        r1 = fake_request(encoding="gzip")
        r2 = fake_request(encoding="br")
        assert get_fingerprint(r1) != get_fingerprint(r2)


# ══════════════════════════════════════════════
# SUITE 2 — get_identifier()
# ══════════════════════════════════════════════
class TestGetIdentifier:
    """Verifica la construcción del identificador fingerprint:ip."""

    def test_format_is_fingerprint_colon_ip(self):
        from main import get_identifier, get_fingerprint
        req = fake_request(ip="10.0.0.1")
        identifier = get_identifier(req)
        fp = get_fingerprint(req)
        assert identifier == f"{fp}:10.0.0.1"

    def test_uses_x_forwarded_for_when_present(self):
        from main import get_identifier
        req = fake_request(ip="10.0.0.1",
                           extra_headers={"X-Forwarded-For": "203.0.113.5"})
        identifier = get_identifier(req)
        assert "203.0.113.5" in identifier

    def test_falls_back_to_client_host(self):
        from main import get_identifier
        req = fake_request(ip="192.168.1.1")
        identifier = get_identifier(req)
        assert "192.168.1.1" in identifier

    def test_identifier_contains_colon_separator(self):
        from main import get_identifier
        identifier = get_identifier(fake_request())
        assert ":" in identifier
        parts = identifier.split(":")
        assert len(parts) >= 2


# ══════════════════════════════════════════════
# SUITE 3 — analyze_behavioral_ai()
# ══════════════════════════════════════════════
class TestAnalyzeBehavioralAI:
    """Verifica la puntuación basada en comportamiento del User-Agent."""

    def test_normal_browser_returns_zero(self):
        from main import analyze_behavioral_ai
        req = fake_request(ua="Mozilla/5.0 Chrome/99")
        assert analyze_behavioral_ai(req) == 0.0

    def test_headless_ua_returns_15(self):
        from main import analyze_behavioral_ai
        for ua in ["headless-chrome", "selenium/4.0",
                   "puppeteer/19", "playwright/1.0"]:
            req = fake_request(ua=ua)
            assert analyze_behavioral_ai(req) == 15.0, f"Falló para UA: {ua}"

    def test_no_accept_language_adds_5(self):
        from main import analyze_behavioral_ai
        req = MagicMock()
        req.headers = {"User-Agent": "Mozilla/5.0"}  # sin Accept-Language
        assert analyze_behavioral_ai(req) == 5.0

    def test_headless_plus_no_language_returns_20(self):
        from main import analyze_behavioral_ai
        req = MagicMock()
        req.headers = {"User-Agent": "selenium-webdriver"}
        assert analyze_behavioral_ai(req) == 20.0

    def test_case_insensitive_ua_detection(self):
        from main import analyze_behavioral_ai
        for ua in ["HEADLESS-CHROME", "Selenium/4", "Puppeteer"]:
            req = fake_request(ua=ua)
            score = analyze_behavioral_ai(req)
            assert score >= 15.0, f"No detectó UA en mayúsculas: {ua}"

    def test_python_requests_ua_is_flagged(self):
        from main import analyze_behavioral_ai
        req = fake_request(ua="python-requests/2.31.0")
        score = analyze_behavioral_ai(req)
        assert score >= 15.0, "python-requests UA no fue detectado como bot"

    def test_legitimate_browser_with_all_headers_is_zero(self):
        from main import analyze_behavioral_ai
        req = fake_request(
            ua="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            lang="es-ES,es;q=0.9",
            encoding="gzip, deflate, br"
        )
        assert analyze_behavioral_ai(req) == 0.0


# ══════════════════════════════════════════════
# SUITE 4 — extract_features()
# ══════════════════════════════════════════════
class TestExtractFeatures:
    """Verifica la extracción de features para el modelo ML."""

    def _extract(self, req, risk=0.0, rate=5, fails=0):
        r = fake_redis(risk=risk, rate_count=rate, fails=fails)
        from main import extract_features, get_identifier
        identifier = get_identifier(req)
        with patch("main.r", r):
            return extract_features(req, identifier)

    def test_returns_numpy_array_shape_1_by_6(self):
        features = self._extract(fake_request())
        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 6)

    def test_feature_0_headless_ua_is_1(self):
        features = self._extract(fake_request(ua="headless-chrome"))
        assert features[0][0] == 1

    def test_feature_0_legit_ua_is_0(self):
        features = self._extract(fake_request(ua="Mozilla/5.0"))
        assert features[0][0] == 0

    def test_feature_1_with_lang_header_is_1(self):
        features = self._extract(fake_request(lang="en-US"))
        assert features[0][1] == 1

    def test_feature_2_reflects_rate_count(self):
        features = self._extract(fake_request(), rate=42)
        assert features[0][2] == 42

    def test_feature_3_reflects_risk_score(self):
        features = self._extract(fake_request(), risk=27.5)
        assert features[0][3] == 27.5

    def test_feature_4_reflects_failed_logins(self):
        features = self._extract(fake_request(), fails=8)
        assert features[0][4] == 8.0

    def test_feature_5_legit_header_true_is_1(self):
        req = fake_request(extra_headers={"X-Legitimate-User": "true"})
        features = self._extract(req)
        assert features[0][5] == 1

    def test_feature_5_legit_header_absent_is_0(self):
        features = self._extract(fake_request())
        assert features[0][5] == 0

    def test_all_features_are_numeric(self):
        features = self._extract(fake_request())
        assert all(isinstance(v, (int, float)) for v in features[0])

    def test_feature_values_are_non_negative(self):
        features = self._extract(fake_request())
        assert all(v >= 0 for v in features[0])


# ══════════════════════════════════════════════
# SUITE 5 — update_risk_score()
# ══════════════════════════════════════════════
class TestUpdateRiskScore:
    """Verifica la actualización del Risk Score en Redis y Prometheus."""

    def _update(self, identifier="test:user", points=10.0, current=0.0):
        r = fake_redis(risk=current)
        mock_gauge = MagicMock()
        with patch("main.r", r), \
             patch("main.RISK_SCORE_METRIC") as mock_metric:
            mock_metric.labels.return_value = mock_gauge
            from main import update_risk_score
            result = update_risk_score(identifier, points)
        return result, r, mock_gauge

    def test_adds_points_to_zero_score(self):
        result, _, _ = self._update(points=15.0, current=0.0)
        assert result == 15.0

    def test_adds_points_to_existing_score(self):
        result, _, _ = self._update(points=10.0, current=25.0)
        assert result == 35.0

    def test_stores_result_in_redis_with_ttl(self):
        _, r, _ = self._update(identifier="user:abc", points=20.0)
        r.set.assert_called_once_with("risk:user:abc", 20.0, ex=10000)

    def test_exports_to_prometheus_gauge(self):
        _, _, mock_gauge = self._update(points=15.0)
        mock_gauge.set.assert_called_once_with(15.0)

    def test_fractional_points_work(self):
        result, _, _ = self._update(points=7.5, current=2.5)
        assert result == 10.0

    def test_zero_points_doesnt_change_score(self):
        result, _, _ = self._update(points=0.0, current=30.0)
        assert result == 30.0

    def test_large_points_work(self):
        result, _, _ = self._update(points=1000.0, current=0.0)
        assert result == 1000.0


# ══════════════════════════════════════════════
# SUITE 6 — ML Model (train_model.py)
# ══════════════════════════════════════════════
class TestMLModelUnit:
    """Verifica el entrenamiento y comportamiento del modelo ML."""

    def test_generate_data_has_correct_shape(self):
        from ml.train_model import generate_training_data, FEATURE_COLS
        df = generate_training_data(n_samples=500)
        assert len(df) == 500
        assert "label" in df.columns
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_binary_features_are_only_0_or_1(self):
        from ml.train_model import generate_training_data
        df = generate_training_data(n_samples=300)
        for col in ["is_headless_ua", "has_accept_language", "has_legitimate_header"]:
            assert df[col].isin([0, 1]).all(), f"{col} tiene valores no binarios"

    def test_continuous_features_are_non_negative(self):
        from ml.train_model import generate_training_data
        df = generate_training_data(n_samples=300)
        for col in ["requests_per_minute", "current_risk_score", "failed_logins"]:
            assert (df[col] >= 0).all(), f"{col} tiene valores negativos"

    def test_both_classes_are_represented(self):
        from ml.train_model import generate_training_data
        df = generate_training_data(n_samples=500)
        assert 0 in df["label"].values, "No hay muestras legítimas (label=0)"
        assert 1 in df["label"].values, "No hay muestras bot (label=1)"

    def test_class_ratio_is_balanced(self):
        from ml.train_model import generate_training_data
        df = generate_training_data(n_samples=1000)
        ratio = df["label"].mean()
        assert 0.3 <= ratio <= 0.7, f"Clases muy desbalanceadas: ratio={ratio:.2f}"

    def test_model_trains_without_error(self):
        from ml.train_model import generate_training_data, FEATURE_COLS
        from sklearn.ensemble import RandomForestClassifier
        df = generate_training_data(n_samples=300)
        X = df[FEATURE_COLS].values
        y = df["label"].values
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert hasattr(model, "feature_importances_")

    def test_model_predicts_binary_labels(self):
        from ml.train_model import generate_training_data, FEATURE_COLS
        from sklearn.ensemble import RandomForestClassifier
        df = generate_training_data(n_samples=300)
        X = df[FEATURE_COLS].values
        y = df["label"].values
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X[:20])
        assert set(preds).issubset({0, 1})

    def test_model_f1_score_above_threshold(self):
        from ml.train_model import generate_training_data, FEATURE_COLS
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        df = generate_training_data(n_samples=800)
        X = df[FEATURE_COLS].values
        y = df["label"].values
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring="f1")
        assert scores.mean() > 0.75, \
            f"F1-score demasiado bajo: {scores.mean():.2f} (mínimo 0.75)"

    def test_obvious_bot_classified_correctly(self):
        """Un bot obvio debe ser clasificado como bot."""
        model_path = Path("ml/bot_detector.pkl")
        if not model_path.exists():
            pytest.skip("Modelo no entrenado — ejecuta: python ml/train_model.py")
        import joblib
        model = joblib.load(model_path)
        # headless=1, no_lang=0, 200 req/min, score=80, 20 fails, no_legit=0
        X = np.array([[1, 0, 200, 80.0, 20, 0]])
        assert model.predict(X)[0] == 1, "Bot obvio clasificado como legítimo"

    def test_obvious_legit_classified_correctly(self):
        """Un usuario legítimo obvio debe ser clasificado como humano."""
        model_path = Path("ml/bot_detector.pkl")
        if not model_path.exists():
            pytest.skip("Modelo no entrenado")
        import joblib
        model = joblib.load(model_path)
        # legit=1, has_lang=1, 2 req/min, score=0, 0 fails, no_legit=0
        X = np.array([[0, 1, 2, 0.0, 0, 0]])
        assert model.predict(X)[0] == 0, "Usuario legítimo clasificado como bot"

    def test_probability_output_is_valid(self):
        """predict_proba debe devolver valores entre 0 y 1."""
        model_path = Path("ml/bot_detector.pkl")
        if not model_path.exists():
            pytest.skip("Modelo no entrenado")
        import joblib
        model = joblib.load(model_path)
        X = np.array([[1, 0, 50, 25.0, 3, 0]])
        proba = model.predict_proba(X)[0]
        assert len(proba) == 2
        assert all(0.0 <= p <= 1.0 for p in proba)
        assert abs(sum(proba) - 1.0) < 1e-6, "Las probabilidades no suman 1"
