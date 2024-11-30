import httpx
import json
from datetime import datetime, timedelta
import random

# Base URL for local testing
BASE_URL = "http://localhost:8000"


def test_health_detailed():
    """Detailed test of the health check endpoint"""
    try:
        print("\nTesting health check with detailed output...")
        response = httpx.get(f"{BASE_URL}/healthcheck")
        print(f"Status code: {response.status_code}")
        print(f"Response content: {response.text}")
        print(f"Response headers: {dict(response.headers)}")
        return response.status_code == 200
    except httpx.RequestError as e:
        print(f"Request failed: {str(e)}")
        return False


def run_extended_test():
    """Run extended feed testing"""
    subscribers, posts = generate_test_data()

    try:
        # Test health check with more detailed output
        print("\n1. Testing health check...")
        health_response = httpx.get(f"{BASE_URL}/healthcheck")
        print(f"Health check status code: {health_response.status_code}")
        print(f"Health check response: {health_response.text}")

        if health_response.status_code != 200:
            print("\n❌ Health check failed - checking server status...")
            try:
                # Try the root endpoint
                root_response = httpx.get(f"{BASE_URL}/")
                print(f"Root endpoint status: {root_response.status_code}")
                print(f"Root response: {root_response.text}")
            except Exception as e:
                print(f"Could not connect to root endpoint: {str(e)}")

            raise AssertionError(
                f"Health check failed with status {health_response.status_code}"
            )

        print("✓ Health check passed")

        # Continue with the rest of your test...
        # [Previous test code remains the same]

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise


if __name__ == "__main__":
    # First, test just the health check
    print("Testing health check endpoint...")
    if test_health_detailed():
        print("Health check OK - proceeding with full test...")
        run_extended_test()
    else:
        print("\nHealth check failed - please check if:")
        print("1. The FastAPI server is running")
        print("2. The server is accessible at", BASE_URL)
        print("3. The healthcheck endpoint is properly defined in main.py")
        print(
            "\nTry accessing",
            f"{BASE_URL}/docs",
            "in your browser to check if the server is running.",
        )
