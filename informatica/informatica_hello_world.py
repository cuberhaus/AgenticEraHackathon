import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
POD_URL = "https://dm-em.informaticacloud.com"
USERNAME = os.getenv("INFORMATICA_USERNAME")
PASSWORD = os.getenv("INFORMATICA_PASSWORD")

def informatica_hello_world():
    # 1. Authentication (Login)
    login_url = f"{POD_URL}/identity-service/api/v1/Login"
    login_payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        response = requests.post(login_url, json=login_payload, headers=headers)
        response.raise_for_status()
        
        login_data = response.json()
        session_id = login_data.get("sessionId")

        if not session_id:
            print("Login failed: Session ID not found in response.")
            print(f"Response: {json.dumps(login_data, indent=2)}")
            return

        print(f"Successfully authenticated!")
        print(f"User: {login_data.get('name')}")
        print(f"Organization: {login_data.get('orgName')}")
        print(f"Session ID: {session_id[:10]}...")
        print(f"\nYour effective roles:")
        for role in login_data.get('effectiveRoles', {}).keys():
            print(f"  - {role}")
        
        print(f"\n✓ Hello World from Informatica IDMC!")
        print(f"✓ Successfully connected to {login_data.get('orgName')}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    informatica_hello_world()