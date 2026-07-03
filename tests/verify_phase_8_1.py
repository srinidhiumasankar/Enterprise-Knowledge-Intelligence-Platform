# tests/verify_phase_8_1.py
# -------------------------
# Verification script for Phase 8.1: Frontend Authentication Module.
# Tests that page-rendering routes are registered and static assets are accessible.

import os
import sys
import logging
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_1")


def test_html_pages_rendering():
    """
    Verify that frontend HTML view endpoints are registered and respond with 200 OK.
    """
    logger.info("Verifying frontend page-rendering endpoints...")
    client = TestClient(app)

    # 1. Test Login page
    logger.info("  Testing GET /login...")
    res_login = client.get("/login")
    assert res_login.status_code == 200, f"Failed loading login page: {res_login.status_code}"
    assert "Sign In - Enterprise Knowledge" in res_login.text, "Login page missing custom title tag context"
    assert "auth.redirectIfLoggedIn()" in res_login.text, "Login page missing login state redirects script"
    logger.info("  ✓ Login page loaded successfully.")

    # 2. Test Register page
    logger.info("  Testing GET /register...")
    res_register = client.get("/register")
    assert res_register.status_code == 200, f"Failed loading register page: {res_register.status_code}"
    assert "Register - Enterprise Knowledge" in res_register.text, "Register page missing custom title tag context"
    assert "strengthBar" in res_register.text, "Register page missing strengthBar progress bar"
    logger.info("  ✓ Register page loaded successfully.")

    # 3. Test Landing page / Dashboard page
    logger.info("  Testing GET /...")
    res_dashboard = client.get("/")
    assert res_dashboard.status_code == 200, f"Failed loading dashboard: {res_dashboard.status_code}"
    assert "auth.protectRoute()" in res_dashboard.text, "Dashboard page missing route protection script"
    logger.info("  ✓ Dashboard page loaded successfully.")


def test_static_js_assets():
    """
    Verify that static JS libraries exist and are accessible from static URLs.
    """
    logger.info("Verifying static JS assets availability...")
    client = TestClient(app)

    # 1. Verify api.js
    logger.info("  Testing GET /static/js/api.js...")
    res_api = client.get("/static/js/api.js")
    assert res_api.status_code == 200, f"Static js/api.js missing: {res_api.status_code}"
    assert "const api = {" in res_api.text, "Static js/api.js does not contain central api object"
    logger.info("  ✓ Centralized api.js utility library verified.")

    # 2. Verify auth.js
    logger.info("  Testing GET /static/js/auth.js...")
    res_auth = client.get("/static/js/auth.js")
    assert res_auth.status_code == 200, f"Static js/auth.js missing: {res_auth.status_code}"
    assert "const auth = {" in res_auth.text, "Static js/auth.js does not contain central auth object"
    logger.info("  ✓ Authentication auth.js utility library verified.")


def run_all_tests():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.1 AUTHENTICATION MODULE VERIFICATION")
    logger.info("==========================================================")

    test_html_pages_rendering()
    test_static_js_assets()

    logger.info("\n==========================================================")
    logger.info("ALL PHASE 8.1 AUTHENTICATION VERIFICATION TESTS PASSED!")
    logger.info("==========================================================")


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as ae:
        logger.error(f"Assertion failed: {ae}")
        sys.exit(1)
    except Exception as ex:
        logger.error(f"Test run encountered unexpected error: {ex}", exc_info=True)
        sys.exit(1)
