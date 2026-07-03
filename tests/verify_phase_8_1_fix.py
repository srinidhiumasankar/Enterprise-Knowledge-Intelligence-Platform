# tests/verify_phase_8_1_fix.py
import sys
import os
import uuid
import logging
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_1_fix")


def test_complete_auth_workflow():
    client = TestClient(app)
    
    # 1. Register a new user
    suffix = uuid.uuid4().hex[:6]
    email = f"user_{suffix}@ekip-rag.com"
    password = "Password123!"
    full_name = "Auth Verification User"
    
    logger.info(f"Step 1: Registering user '{email}'...")
    res_reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "full_name": full_name}
    )
    assert res_reg.status_code == 201, f"Expected 201, got {res_reg.status_code}: {res_reg.text}"
    logger.info("✓ User registered successfully.")

    # 2. Duplicate registration check
    logger.info("Step 2: Checking duplicate registration...")
    res_dup = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "full_name": full_name}
    )
    assert res_dup.status_code in (400, 409, 422), f"Expected error on duplicate, got {res_dup.status_code}: {res_dup.text}"
    logger.info("✓ Duplicate registration correctly rejected.")

    # 3. Invalid password complexity check
    logger.info("Step 3: Checking invalid password complexity...")
    res_pw = client.post(
        "/api/auth/register",
        json={"email": f"bad_{suffix}@ekip-rag.com", "password": "weak", "full_name": "Bad Password User"}
    )
    assert res_pw.status_code == 422, f"Expected 422 for weak password, got {res_pw.status_code}: {res_pw.text}"
    logger.info("✓ Weak password correctly validation failed on backend.")

    # 4. Wrong credentials login check
    logger.info("Step 4: Logging in with wrong credentials...")
    res_log_fail = client.post(
        "/api/auth/login",
        json={"email": email, "password": "WrongPassword!"}
    )
    assert res_log_fail.status_code == 401, f"Expected 401, got {res_log_fail.status_code}: {res_log_fail.text}"
    logger.info("✓ Wrong credentials login correctly rejected.")

    # 5. Successful login
    logger.info("Step 5: Logging in with correct credentials...")
    res_log_ok = client.post(
        "/api/auth/login",
        json={"email": email, "password": password}
    )
    assert res_log_ok.status_code == 200, f"Expected 200, got {res_log_ok.status_code}: {res_log_ok.text}"
    token_data = res_log_ok.json()
    assert "access_token" in token_data, "Response missing access_token"
    token = token_data["access_token"]
    logger.info("✓ Login successful and token returned.")

    # 6. Retrieve profile using JWT
    logger.info("Step 6: Retrieving user profile with JWT...")
    headers = {"Authorization": f"Bearer {token}"}
    res_me = client.get("/api/auth/me", headers=headers)
    assert res_me.status_code == 200, f"Expected 200, got {res_me.status_code}: {res_me.text}"
    profile_data = res_me.json()
    assert profile_data["email"] == email, f"Expected email {email}, got {profile_data['email']}"
    logger.info("✓ Profile successfully loaded with valid JWT.")

    # 7. Logout
    logger.info("Step 7: Logging out...")
    res_logout = client.post("/api/auth/logout", headers=headers)
    assert res_logout.status_code == 200, f"Expected 200, got {res_logout.status_code}: {res_logout.text}"
    logger.info("✓ Logout successful.")


def run_all_tests():
    logger.info("==========================================================")
    logger.info("RUNNING AUTOMATED FRONTEND AUTHENTICATION FLOW TESTS")
    logger.info("==========================================================")
    
    test_complete_auth_workflow()
    
    logger.info("\n==========================================================")
    logger.info("ALL FRONTEND AUTHENTICATION FLOW TESTS COMPLETED SUCCESSFULLY!")
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
