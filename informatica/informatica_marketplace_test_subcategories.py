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
    access_token = jwt_data.get("jwt_token")
    
    if not access_token:
        raise Exception("Failed to generate JWT token")
    
    print(f"✓ JWT Token generated successfully")
    
    return access_token

def list_categories(jwt_token, org_id, session_id):
    """List all existing categories"""
    
    marketplace_url = f"{MARKETPLACE_URL}/api/v1/categories"
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "X-INFA-ORG-ID": org_id,
        "IDS-SESSION-ID": session_id,
        "Accept": "application/json"
    }
    
    response = requests.get(marketplace_url, headers=headers)
    response.raise_for_status()
    
    categories = response.json()
    
    # Handle different response formats
    if isinstance(categories, dict):
        # If response is a dict, it might have a 'data' or 'items' key
        if 'data' in categories:
            categories = categories['data']
        elif 'items' in categories:
            categories = categories['items']
        else:
            print(f"Response structure: {list(categories.keys())}")
            categories = [categories]
    
    print(f"\nFound {len(categories)} categories:")
    for cat in categories:
        if isinstance(cat, dict):
            print(f"  - {cat.get('name', 'N/A')} (ID: {cat.get('id', 'N/A')})")
        else:
            print(f"  - {cat}")
    
    return categories

def create_category(jwt_token, org_id, session_id, category_name, category_description=""):
    """Create a new parent category in Data Marketplace"""
    
    marketplace_url = f"{MARKETPLACE_URL}/api/v1/categories"
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "X-INFA-ORG-ID": org_id,
        "IDS-SESSION-ID": session_id,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    category_payload = {
        "name": category_name,
        "description": category_description,
        "isActive": True
    }
    
    print(f"\nCreating parent category: {category_name}")
    
    response = requests.post(marketplace_url, json=category_payload, headers=headers)
    response.raise_for_status()
    
    category_data = response.json()
    print(f"✓ Category created successfully!")
    print(f"  ID: {category_data.get('id')}")
    print(f"  Name: {category_data.get('name')}")
    
    return category_data

def create_subcategory(jwt_token, org_id, session_id, parent_category_id, subcategory_name, subcategory_description=""):
    """Create a subcategory under a parent category"""
    
    # Try different possible endpoints for subcategories
    endpoints_to_try = [
        f"{MARKETPLACE_URL}/api/v1/categories/{parent_category_id}/subcategories",
        f"{MARKETPLACE_URL}/api/v1/categories",  # With parentId in payload
        f"{MARKETPLACE_URL}/api/v1/data-categories",  # Alternative endpoint
    ]
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "X-INFA-ORG-ID": org_id,
        "IDS-SESSION-ID": session_id,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    print(f"\nCreating subcategory: {subcategory_name} under parent ID: {parent_category_id}")
    
    # Try first endpoint (nested route)
    try:
        subcategory_payload = {
            "name": subcategory_name,
            "description": subcategory_description,
            "isActive": True
        }
        
        response = requests.post(endpoints_to_try[0], json=subcategory_payload, headers=headers)
        
        if response.status_code in [200, 201]:
            subcategory_data = response.json()
            print(f"✓ Subcategory created successfully!")
            print(f"  ID: {subcategory_data.get('id')}")
            print(f"  Name: {subcategory_data.get('name')}")
            print(f"  Parent ID: {parent_category_id}")
            return subcategory_data
        else:
            print(f"  Endpoint 1 failed with status {response.status_code}")
    except Exception as e:
        print(f"  Endpoint 1 error: {str(e)[:100]}")
    
    # Try second endpoint (with parentId in payload)
    try:
        subcategory_payload = {
            "name": subcategory_name,
            "description": subcategory_description,
            "parentId": parent_category_id,
            "isActive": True
        }
        
        print(f"  Trying with parentId in payload...")
        response = requests.post(endpoints_to_try[1], json=subcategory_payload, headers=headers)
        
        if response.status_code in [200, 201]:
            subcategory_data = response.json()
            print(f"✓ Subcategory created successfully!")
            print(f"  ID: {subcategory_data.get('id')}")
            print(f"  Name: {subcategory_data.get('name')}")
            print(f"  Parent ID: {subcategory_data.get('parentId')}")
            return subcategory_data
        else:
            print(f"  Endpoint 2 failed with status {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  Endpoint 2 error: {str(e)[:100]}")
    
    # Try third endpoint (data-categories)
    try:
        subcategory_payload = {
            "name": subcategory_name,
            "description": subcategory_description,
            "parentCategoryId": parent_category_id,
            "isActive": True
        }
        
        print(f"  Trying data-categories endpoint...")
        response = requests.post(endpoints_to_try[2], json=subcategory_payload, headers=headers)
        
        if response.status_code in [200, 201]:
            subcategory_data = response.json()
            print(f"✓ Subcategory created successfully!")
            print(f"  ID: {subcategory_data.get('id')}")
            print(f"  Name: {subcategory_data.get('name')}")
            return subcategory_data
        else:
            print(f"  Endpoint 3 failed with status {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  Endpoint 3 error: {str(e)[:100]}")
    
    raise Exception("Could not create subcategory with any endpoint method")

def main():
    try:
        # 1. Login
        session_id, org_id = login()
        
        # 2. Generate JWT token
        print("\n" + "="*60)
        jwt_token = generate_jwt_token(session_id)
        
        # 3. List existing categories
        print("\n" + "="*60)
        categories = list_categories(jwt_token, org_id, session_id)
        
        # 4. Create parent category
        print("\n" + "="*60)
        parent_category_name = "Main Category"
        parent_category_description = "This is a parent category for testing subcategories"
        
        parent_category = create_category(jwt_token, org_id, session_id, 
                                         parent_category_name, parent_category_description)
        
        # 5. Create subcategories
        print("\n" + "="*60)
        parent_id = parent_category.get('id')
        
        subcategories = [
            ("Subcategory 1", "First subcategory"),
            ("Subcategory 2", "Second subcategory"),
            ("Subcategory 3", "Third subcategory")
        ]
        
        created_subcategories = []
        for sub_name, sub_desc in subcategories:
            try:
                sub = create_subcategory(jwt_token, org_id, session_id, parent_id, sub_name, sub_desc)
                created_subcategories.append(sub)
            except Exception as e:
                print(f"  ✗ Failed to create '{sub_name}': {e}")
        
        # 6. Summary
        print("\n" + "="*60)
        print("✓ Process completed!")
        print(f"  Parent category: {parent_category_name} (ID: {parent_id})")
        print(f"  Subcategories created: {len(created_subcategories)}")
        for sub in created_subcategories:
            print(f"    - {sub.get('name')} (ID: {sub.get('id')})")
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
