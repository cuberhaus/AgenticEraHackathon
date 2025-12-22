import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
LOGIN_URL = "https://dm-em.informaticacloud.com"
MARKETPLACE_URL = "https://idmc-api.dm-em.informaticacloud.com/data360/marketplace"
USERNAME = os.getenv("INFORMATICA_USERNAME")
PASSWORD = os.getenv("INFORMATICA_PASSWORD")

def login():
    """Authenticate and get session ID and org ID"""
    login_url = f"{LOGIN_URL}/identity-service/api/v1/Login"
    login_payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    response = requests.post(login_url, json=login_payload, headers=headers)
    response.raise_for_status()
    
    login_data = response.json()
    session_id = login_data.get("sessionId")
    org_id = login_data.get("orgId")
    
    if not session_id or not org_id:
        raise Exception("Login failed: Session ID or Org ID not found")
    
    print(f"✓ Authenticated as: {login_data.get('name')}")
    print(f"✓ Organization: {login_data.get('orgName')}")
    print(f"✓ Org ID: {org_id}")
    
    return session_id, org_id

def generate_jwt_token(session_id):
    """Generate JWT access token from session ID"""
    jwt_url = f"{LOGIN_URL}/identity-service/api/v1/jwt/Token?client_id=idmc_api&nonce=1234"
    
    headers = {
        "IDS-SESSION-ID": session_id
    }
    
    response = requests.post(jwt_url, headers=headers)
    response.raise_for_status()
    
    jwt_data = response.json()
    access_token = jwt_data.get("jwt_token")  # Changed from "access_token" to "jwt_token"
    
    if not access_token:
        print(f"JWT Response data: {json.dumps(jwt_data, indent=2)}")
        raise Exception("Failed to generate JWT token")
    
    print(f"✓ JWT Token generated successfully")
    
    return access_token

def create_category(jwt_token, org_id, session_id, category_name, category_description=""):
    """Create a new category in Data Marketplace"""
    
    # Data Marketplace API endpoint for categories - using v1 based on testing
    marketplace_url = f"{MARKETPLACE_URL}/api/v1/categories"
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "X-INFA-ORG-ID": org_id,
        "IDS-SESSION-ID": session_id,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Category payload
    category_payload = {
        "name": category_name,
        "description": category_description,
        "isActive": True
    }
    
    print(f"\nCreating category: {category_name}")
    
    response = requests.post(marketplace_url, json=category_payload, headers=headers)
    response.raise_for_status()
    
    category_data = response.json()
    print(f"✓ Category created successfully!")
    print(f"  ID: {category_data.get('id')}")
    print(f"  Name: {category_data.get('name')}")
    print(f"  Description: {category_data.get('description')}")
    
    return category_data

def list_categories(jwt_token, org_id, session_id):
    """List all existing categories"""
    
    # Try different possible endpoints from documentation
    test_endpoints = [
        # v1 endpoints that returned 401 (they exist!)
        (f"{MARKETPLACE_URL}/api/v1/categories", "Categories v1"),
        (f"{MARKETPLACE_URL}/api/v1/data-categories", "Data Categories v1"),
    ]
    
    # Try with different header combinations
    header_combinations = [
        {
            "Authorization": f"Bearer {jwt_token}",
            "X-INFA-ORG-ID": org_id,
            "IDS-SESSION-ID": session_id,
            "Accept": "application/json"
        },
        {
            "Authorization": f"Bearer {jwt_token}",
            "X-INFA-ORG-ID": org_id,
            "icSessionId": session_id,
            "Accept": "application/json"
        },
    ]
    
    print("\nTesting endpoints with different header combinations...")
    
    for endpoint, name in test_endpoints:
        print(f"\n  [{name}]")
        print(f"  URL: {endpoint}")
        
        for idx, headers in enumerate(header_combinations, 1):
            print(f"    Header combo {idx}:", end=" ")
            try:
                response = requests.get(endpoint, headers=headers)
                print(f"Status {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"    ✓ SUCCESS!")
                    
                    if isinstance(data, list):
                        print(f"    Found {len(data)} items")
                        if len(data) > 0:
                            print(f"    Sample: {json.dumps(data[0], indent=6)[:400]}...")
                    elif isinstance(data, dict):
                        print(f"    Response keys: {list(data.keys())}")
                        print(f"    Data: {json.dumps(data, indent=6)[:400]}...")
                    
                    return data
                elif response.status_code == 401:
                    print(f"    Still unauthorized")
                else:
                    print(f"    Response: {response.text[:100]}")
            except Exception as e:
                print(f"    Error: {str(e)[:80]}")
    
    raise Exception("Could not authenticate with any header combination. Check API permissions.")

def main():
    try:
        # 1. Login and get session ID and org ID
        session_id, org_id = login()
        
        # 2. Generate JWT token
        print("\n" + "="*60)
        jwt_token = generate_jwt_token(session_id)
        
        # 3. List existing categories (optional)
        print("\n" + "="*60)
        list_categories(jwt_token, org_id, session_id)
        
        # 4. Create new category
        print("\n" + "="*60)
        category_name = "Test Category"
        category_description = "This is a test category created via API"
        
        # You can modify these values:
        # category_name = "Your Category Name"
        # category_description = "Your category description"
        
        new_category = create_category(jwt_token, org_id, session_id, category_name, category_description)
        
        print("\n" + "="*60)
        print("✓ Process completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
