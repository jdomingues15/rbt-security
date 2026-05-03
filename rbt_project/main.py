from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, generate_latest
from contextlib import asynccontextmanager
import redis, time, os, hashlib, joblib, numpy as np
from collections import deque
from pathlib import Path

# ─────────────────────────────────────────────
# REDIS CONNECTION
# ─────────────────────────────────────────────
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

# ─────────────────────────────────────────────
# ML MODEL — load at startup if exists
# ─────────────────────────────────────────────
MODEL_PATH = Path("ml/bot_detector.pkl")
bot_model = None

def load_model():
    global bot_model
    if MODEL_PATH.exists():
        bot_model = joblib.load(MODEL_PATH)
        print("✅ ML model loaded from", MODEL_PATH)
    else:
        print("⚠️  No ML model found — using Risk Score rules only")
        print("   Run: python ml/train_model.py to train the model")

# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    RISK_SCORE_METRIC.labels(identifier="system_startup").set(0)
    LOGIN_FAILURES.labels(method="password", reason="none").inc(0)
    yield
    # Shutdown
    r.close()

app = FastAPI(lifespan=lifespan)

# ─────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────
REQUESTS          = Counter("http_requests_total",         "Total requests",           ["method", "endpoint"])
BLOCKED           = Counter("blocked_requests_total",      "Blocked requests",         ["reason", "identifier"])
FALSE_POSITIVES   = Counter("false_positive_blocks_total", "False positive detections",["identifier"])
RISK_SCORE_METRIC = Gauge  ("current_risk_score",          "Risk score per user",      ["identifier"])
LOGIN_FAILURES    = Counter("login_failures_total",        "Failed login attempts",    ["method", "reason"])
BOT_PROBABILITY   = Gauge  ("bot_ml_probability",          "ML bot probability 0-1",   ["identifier"])
ML_BLOCKED        = Counter("ml_blocked_total",            "Requests blocked by ML",   ["identifier"])

# ─────────────────────────────────────────────
# SECURITY CONFIG
# ─────────────────────────────────────────────
WINDOW    = 100     # Rate limit window in seconds
LIMIT     = 10000   # Max requests in window
THRESHOLD = 30      # Risk score threshold for blocking

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def get_fingerprint(request: Request) -> str:
    """Generate a unique fingerprint from request headers."""
    ua       = request.headers.get("User-Agent", "unknown")
    lang     = request.headers.get("Accept-Language", "unknown")
    encoding = request.headers.get("Accept-Encoding", "unknown")
    return hashlib.md5(f"{ua}|{lang}|{encoding}".encode()).hexdigest()


def get_identifier(request: Request) -> str:
    """Return fingerprint:ip string."""
    client_id = get_fingerprint(request)
    ip = request.headers.get("X-Forwarded-For", request.client.host)
    return f"{client_id}:{ip}"


def update_risk_score(identifier: str, points: float) -> float:
    """Add points to the risk score stored in Redis."""
    key = f"risk:{identifier}"
    current = float(r.get(key) or 0)
    new_score = current + points
    r.set(key, new_score, ex=10000)
    RISK_SCORE_METRIC.labels(identifier=identifier).set(new_score)
    return new_score


def extract_features(request: Request, identifier: str) -> np.ndarray:
    """
    Extract numeric features from the request for the ML model.
    Returns a (1, 6) numpy array.

    Features:
        0 - is_headless_ua       : 1 if UA contains bot keywords
        1 - has_accept_language  : 1 if header present
        2 - requests_per_minute  : count in rate window
        3 - current_risk_score   : accumulated score in Redis
        4 - failed_logins        : failed logins counter in Redis
        5 - has_legitimate_header: 1 if X-Legitimate-User: true
    """
    ua = request.headers.get("User-Agent", "").lower()
    bot_keywords = ["headless", "selenium", "puppeteer", "playwright", "python-requests"]

    is_headless     = int(any(kw in ua for kw in bot_keywords))
    has_lang        = int("accept-language" in request.headers)
    rate_key        = f"rate:{identifier}"
    req_count       = int(r.zcard(rate_key) or 0)
    risk_score      = float(r.get(f"risk:{identifier}") or 0)
    failed_logins   = int(r.get(f"fails:{identifier}") or 0)
    has_legit       = int(request.headers.get("X-Legitimate-User") == "true")

    return np.array([[is_headless, has_lang, req_count,
                      risk_score, failed_logins, has_legit]])


def analyze_behavioral_ai(request: Request) -> float:
    """Rule-based behavioral scoring (fallback when no ML model)."""
    score = 0.0
    ua = request.headers.get("User-Agent", "").lower()
    if any(b in ua for b in ["headless", "selenium", "puppeteer", "playwright"]):
        score += 15
    if "accept-language" not in request.headers:
        score += 5
    return score


# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────
@app.middleware("http")
async def security_middleware(request: Request, call_next):

    # Allow health checks and metrics without inspection
    if request.url.path in ["/", "/metrics"]:
        return await call_next(request)

    REQUESTS.labels(method=request.method, endpoint=request.url.path).inc()

    identifier = get_identifier(request)

    # ── Rate limiting ──────────────────────────────────────────
    rate_key = f"rate:{identifier}"
    now = int(time.time())
    r.zadd(rate_key, {now: now})
    r.zremrangebyscore(rate_key, 0, now - WINDOW)
    r.expire(rate_key, WINDOW + 10)

    # ── ML-based detection (if model available) ───────────────
    if bot_model is not None:
        features = extract_features(request, identifier)
        prediction  = int(bot_model.predict(features)[0])
        probability = float(bot_model.predict_proba(features)[0][1])

        BOT_PROBABILITY.labels(identifier=identifier).set(probability)

        if prediction == 1:
            ML_BLOCKED.labels(identifier=identifier).inc()
            BLOCKED.labels(reason="ml_bot_detected", identifier=identifier).inc()
            return Response(
                content=f"Access Denied: ml_bot_detected (p={probability:.2f})",
                status_code=403
            )

    # ── Rule-based scoring (always active as second layer) ─────
    ai_risk = analyze_behavioral_ai(request)
    if ai_risk > 0:
        update_risk_score(identifier, ai_risk)

    final_score = float(r.get(f"risk:{identifier}") or 0)

    if final_score > THRESHOLD:
        if request.headers.get("X-Legitimate-User") == "true":
            FALSE_POSITIVES.labels(identifier=identifier).inc()
            return await call_next(request)

        BLOCKED.labels(reason="risk_score_exceeded", identifier=identifier).inc()
        return Response(
            content=f"Access Denied: risk_score_exceeded (score={final_score})",
            status_code=403
        )

    return await call_next(request)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "RBT Security Layer Active", "ml_model_loaded": bot_model is not None}


@app.get("/api/data")
def protected_data():
    return {"data": "secure_content"}


@app.post("/login")
@app.get("/login")
async def login(request: Request, username: str = None, password: str = None):
    identifier = get_identifier(request)

    if username == "admin" and password == "secret123":
        # Reset fail counter on successful login
        r.delete(f"fails:{identifier}")
        return {"message": "Welcome", "status": "success"}

    # Track failed login in Redis for ML feature
    r.incr(f"fails:{identifier}")
    r.expire(f"fails:{identifier}", 600)   # 10 min TTL

    update_risk_score(identifier, 10)
    LOGIN_FAILURES.labels(method="password", reason="invalid_credentials").inc()
    return Response(content="Invalid credentials", status_code=401)


@app.get("/status")
def status():
    """Returns system status including ML model info."""
    return {
        "status": "running",
        "ml_model_loaded": bot_model is not None,
        "model_path": str(MODEL_PATH),
        "threshold": THRESHOLD,
    }


@app.get("/metrics")
def metrics():
    """Prometheus scraping endpoint."""
    return Response(generate_latest(), media_type="text/plain")
