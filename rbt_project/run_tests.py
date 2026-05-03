#!/usr/bin/env python3
"""
run_tests.py — Master runner actualizado con todas las suites
─────────────────────────────────────────────────────────────

Suites disponibles:
  unit        → test_unit.py       (tests originales)
  unit-deep   → test_unit_deep.py  (funciones internas + ML unit)
  integration → test_integration.py (FastAPI ↔ Redis ↔ ML ↔ Prometheus)
  security    → test_security.py   (Risk Score, autenticación, falsos positivos)
  advanced    → test_advanced.py   (contrato API, chaos, concurrencia, boundary)
  pentest     → test_penetration.py (SQL, XSS, fuerza bruta, evasión)
  load        → test_load.py       (rendimiento, concurrencia, carga sostenida)
  e2e         → test_e2e.py        (journeys completos, Grafana, Prometheus)
  ai          → test_ai.py         (calidad ML, robustez, fairness, explainability)
  all         → todas las suites

Uso:
    python run_tests.py                    → security + advanced + integration
    python run_tests.py --unit             → tests sin API
    python run_tests.py --unit-deep        → tests unitarios profundos
    python run_tests.py --integration      → integración de componentes
    python run_tests.py --ai               → pruebas de IA/ML
    python run_tests.py --pentest          → penetración
    python run_tests.py --load             → carga (sin test sostenido)
    python run_tests.py --e2e              → end-to-end
    python run_tests.py --all              → todo
    python run_tests.py --quick            → solo tests rápidos
    python run_tests.py --grafana          → genera datos para Grafana
    python run_tests.py --no-api           → solo tests sin API (unit + unit-deep + ai-unit)
"""

import subprocess
import sys
import argparse
import time
import os

BASE = "http://localhost:8000"


def _check(url, name, timeout=5):
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=timeout)
        print(f"  ✅ {name}: OK")
        return True
    except Exception:
        print(f"  ❌ {name}: no responde")
        return False


def check_services():
    import urllib.request, json
    print("\nVerificando servicios...")
    api_ok = _check(f"{BASE}/", "FastAPI")
    _check("http://localhost:9090/-/healthy", "Prometheus")
    _check("http://localhost:3000/api/health", "Grafana")

    if api_ok:
        try:
            data = json.loads(
                urllib.request.urlopen(f"{BASE}/status", timeout=5).read()
            )
            ml = data.get("ml_model_loaded", False)
            print(f"  {'✅' if ml else '⚠️ '} ML Model: {'cargado' if ml else 'NO cargado'}")
        except Exception:
            pass
    return api_ok


def run_pytest(args_list, label):
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}\n")
    os.makedirs("reports", exist_ok=True)
    cmd = [sys.executable, "-m", "pytest"] + args_list
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="🛡 RBT Security — Master Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--unit",        action="store_true", help="Unit tests originales (test_unit.py)")
    parser.add_argument("--unit-deep",   action="store_true", help="Unit tests profundos (test_unit_deep.py)")
    parser.add_argument("--integration", action="store_true", help="Integration tests (FastAPI↔Redis↔ML↔Prometheus)")
    parser.add_argument("--security",    action="store_true", help="Security tests (test_security.py)")
    parser.add_argument("--advanced",    action="store_true", help="Advanced tests (contrato API, chaos, boundary)")
    parser.add_argument("--pentest",     action="store_true", help="Penetration tests (SQL, XSS, brute force)")
    parser.add_argument("--load",        action="store_true", help="Load tests (sin test sostenido)")
    parser.add_argument("--e2e",         action="store_true", help="End-to-end tests")
    parser.add_argument("--ai",          action="store_true", help="AI/ML tests (calidad, fairness, robustez)")
    parser.add_argument("--all",         action="store_true", help="Todas las suites")
    parser.add_argument("--quick",       action="store_true", help="Solo tests rápidos (sin load/e2e)")
    parser.add_argument("--no-api",      action="store_true", help="Solo tests sin API (unit + unit-deep + ai sin live)")
    parser.add_argument("--grafana",     action="store_true", help="Generar datos para Grafana")
    args = parser.parse_args()

    nothing_selected = not any(vars(args).values())

    print("\n🛡  RBT Security — Master Test Runner")
    print("━" * 60)

    # Para tests sin API
    if args.no_api:
        api_ok = False
        print("\n⚙️  Modo sin API — solo tests en aislamiento")
    else:
        api_ok = check_services()
        if not api_ok and not args.unit and not args.unit_deep and not args.ai:
            print("\n❌ API no disponible. Ejecuta: docker compose up -d")
            print("   O usa: python run_tests.py --no-api")
            sys.exit(1)

    results = {}
    start   = time.time()

    # ── Unit Tests (originales) ───────────────────────────────
    if args.unit or args.all or args.no_api or nothing_selected:
        try:
            from ml.train_model import generate_training_data
        except Exception:
            subprocess.run([sys.executable, "ml/train_model.py"], check=False)

        ok = run_pytest([
            "tests/test_unit.py", "-v", "--tb=short",
            "--html=reports/unit_report.html", "--self-contained-html"
        ], "UNIT TESTS — funciones internas con mocks")
        results["Unit (original)"] = ok

    # ── Unit Tests Deep ───────────────────────────────────────
    if args.unit_deep or args.all or args.no_api:
        ok = run_pytest([
            "tests/test_unit_deep.py", "-v", "--tb=short",
            "--html=reports/unit_deep_report.html", "--self-contained-html"
        ], "UNIT TESTS DEEP — get_fingerprint, extract_features, update_risk_score, ML")
        results["Unit Deep"] = ok

    # ── AI Tests ─────────────────────────────────────────────
    if args.ai or args.all:
        k_filter = ["-k", "not Live"] if args.no_api else []
        ok = run_pytest([
            "tests/test_ai.py", "-v", "--tb=short",
        ] + k_filter + [
            "--html=reports/ai_report.html", "--self-contained-html"
        ], "AI/ML TESTS — calidad, robustez, fairness, explicabilidad, drift")
        results["AI/ML"] = ok

    # ── Integration Tests ─────────────────────────────────────
    if (args.integration or args.all or nothing_selected) and api_ok:
        ok = run_pytest([
            "tests/test_integration.py", "-v", "--tb=short",
            "--html=reports/integration_report.html", "--self-contained-html"
        ], "INTEGRATION TESTS — FastAPI ↔ Redis ↔ ML ↔ Prometheus")
        results["Integration"] = ok

    # ── Security Tests ────────────────────────────────────────
    if (args.security or args.all or nothing_selected) and api_ok:
        ok = run_pytest([
            "tests/test_security.py", "-v", "--tb=short",
            "--html=reports/security_report.html", "--self-contained-html"
        ], "SECURITY TESTS — Risk Score, autenticación, falsos positivos, ML")
        results["Security"] = ok

    # ── Advanced Tests ────────────────────────────────────────
    if (args.advanced or args.all or nothing_selected) and api_ok:
        ok = run_pytest([
            "tests/test_advanced.py", "-v", "--tb=short",
            "--html=reports/advanced_report.html", "--self-contained-html"
        ], "ADVANCED TESTS — contrato API, chaos, concurrencia, boundary values")
        results["Advanced"] = ok

    # ── Penetration Tests ─────────────────────────────────────
    if (args.pentest or args.all) and api_ok:
        ok = run_pytest([
            "tests/test_penetration.py", "-v", "-s", "--tb=short",
            "--html=reports/pentest_report.html", "--self-contained-html"
        ], "PENETRATION TESTS — SQL injection, XSS, fuerza bruta, evasión")
        results["Penetration"] = ok

    # ── Load Tests ────────────────────────────────────────────
    if (args.load or args.all) and api_ok:
        ok = run_pytest([
            "tests/test_load.py", "-v", "-s", "--tb=short",
            "-k", "not sustained",
            "--html=reports/load_report.html", "--self-contained-html"
        ], "LOAD TESTS — rendimiento, P99, concurrencia (sin test sostenido)")
        results["Load"] = ok

    # ── E2E Tests ─────────────────────────────────────────────
    if (args.e2e or args.all) and api_ok:
        ok = run_pytest([
            "tests/test_e2e.py", "-v", "-s", "--tb=short",
            "--html=reports/e2e_report.html", "--self-contained-html"
        ], "E2E TESTS — journeys completos, Grafana, Prometheus, ML pipeline")
        results["E2E"] = ok

    # ── Grafana data generation ───────────────────────────────
    if args.grafana and api_ok:
        ok = run_pytest([
            "tests/test_security.py::TestLoadGeneration", "-v", "-s",
        ], "GRAFANA DATA — generando tráfico para el dashboard")
        results["Grafana Data"] = ok

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'═'*60}")
    print(f"  RESUMEN FINAL  ({elapsed:.0f}s total)")
    print(f"{'═'*60}")

    all_passed = True
    for suite, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {suite}")
        if not passed:
            all_passed = False

    if results:
        print(f"\n  📄 Reportes HTML en: reports/")
        print(f"  📊 Grafana: http://localhost:3000")
    print()

    if all_passed:
        print("  🎉 Todas las suites pasaron correctamente")
    else:
        print("  ⚠️  Algunas suites fallaron — revisa el output anterior")
        sys.exit(1)


if __name__ == "__main__":
    main()
