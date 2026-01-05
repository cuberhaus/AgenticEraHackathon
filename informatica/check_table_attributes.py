"""
Informatica Data Governance - Print Table Attributes Script

This script connects to Informatica Data Governance and Catalog,
retrieves all attributes/columns from a scanned table, and prints
a specific field value for each attribute.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LOGIN_URL = "https://dm-em.informaticacloud.com"
# Try different catalog API versions
CATALOG_API_URL = "https://dm-em.informaticacloud.com/ldm-service/api/v2"
USERNAME = os.getenv("INFORMATICA_USERNAME")
PASSWORD = os.getenv("INFORMATICA_PASSWORD")


def login():
    """Authenticate and get session ID"""
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
    
    if not session_id:
        raise Exception("Login failed: Session ID not found")
    
    print(f"✓ Authenticated as: {login_data.get('name')}")
    return session_id


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
    
    print("✓ JWT Token generated")
    return access_token


def search_table(access_token, table_name):
    """Search for a table in the catalog using the Data360 Search API"""
    
    # Use the correct Data360 Search API endpoint
    endpoint = "https://idmc-api.dm-em.informaticacloud.com/data360/search/v1/assets"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Try multiple search strategies
    search_attempts = [
        # 1. Search by exact name
        {
            "params": {"knowledgeQuery": table_name, "segments": "all"},
            "body": {"from": 0, "size": 100}
        },
        # 2. Search tables with name filter
        {
            "params": {"knowledgeQuery": "table", "segments": "all"},
            "body": {
                "from": 0,
                "size": 100,
                "filterSpec": [
                    {"type": "simple", "attribute": "core.name", "values": [table_name]}
                ]
            }
        },
        # 3. Wildcard search
        {
            "params": {"knowledgeQuery": f"*{table_name}*", "segments": "all"},
            "body": {"from": 0, "size": 100}
        },
        # 4. Search all assets, filter by name
        {
            "params": {"knowledgeQuery": "*", "segments": "all"},
            "body": {
                "from": 0,
                "size": 100,
                "filterSpec": [
                    {"type": "simple", "attribute": "core.name", "values": [table_name]}
                ]
            }
        }
    ]
    
    for i, attempt in enumerate(search_attempts, 1):
        try:
            print(f"\nAttempt {i}: {attempt['params']['knowledgeQuery']}")
            
            response = requests.post(endpoint, params=attempt['params'], json=attempt['body'], headers=headers)
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                hits = results.get("hits", [])
                total_hits = results.get("summary", {}).get("total_hits", 0)
                
                print(f"  Total matches: {total_hits}")
                
                if hits:
                    print(f"  ✓ Found {len(hits)} result(s)")
                    
                    # Show all results
                    for idx, hit in enumerate(hits[:5], 1):
                        name = hit.get("summary", {}).get("core.name", "Unknown")
                        class_type = hit.get("systemAttributes", {}).get("core.classType", "Unknown")
                        print(f"    {idx}. {name} ({class_type})")
                    
                    # Return first match
                    table = hits[0]
                    table_id = table.get("core.identity")
                    table_name_found = table.get("summary", {}).get("core.name")
                    table_type = table.get("systemAttributes", {}).get("core.classType")
                    
                    print(f"\n✓ Using: {table_name_found}")
                    print(f"  ID: {table_id}")
                    print(f"  Type: {table_type}")
                    
                    return table
                    
        except Exception as e:
            print(f"  Exception: {e}")
            continue
    
    print(f"\n✗ Table '{table_name}' not found after all attempts")
    print("\nPossible reasons:")
    print("1. Table not scanned/cataloged in Data Governance")
    print("2. Table name is different (check in Informatica UI)")
    print("3. Table in a different schema (try: SCHEMA.TABLE)")
    return None


def get_table_columns(access_token, table):
    """Get all columns/attributes of a table"""
    
    # The table object from search should contain column information
    # or we can fetch detailed asset info using the details URI
    
    table_id = table.get("core.identity")
    details_uri = table.get("details")
    
    # Try to get columns from related assets
    endpoint = "https://idmc-api.dm-em.informaticacloud.com/data360/search/v1/assets"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Search for columns related to this table
    params = {
        "knowledgeQuery": f"columns related to {table.get('summary', {}).get('core.name', '')}",
        "segments": "all"
    }
    
    body = {
        "from": 0,
        "size": 1000  # Get more columns
    }
    
    try:
        response = requests.post(endpoint, params=params, json=body, headers=headers)
        
        if response.status_code == 200:
            results = response.json()
            columns = results.get("hits", [])
            print(f"Found {len(columns)} columns")
            return columns
        else:
            print(f"Failed to get columns: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error getting columns: {e}")
        return []


def get_column_attributes(access_token, column):
    """Get detailed attributes of a column - column data is already in the search result"""
    # The search API with segments=all returns all attributes
    # So we can just return the column as-is
    return column


def print_attribute_field(access_token, table_name, field_name):
    """
    Print a specific field/attribute value for all columns in a table
    
    Args:
        access_token: JWT token for API authentication
        table_name: Name of the table to check
        field_name: The field/attribute to print (e.g., "Nivel de seguridad", "core.classificationNames")
    """
    print(f"\n{'='*60}")
    print(f"Table: {table_name}")
    print(f"Showing field: {field_name}")
    print(f"{'='*60}\n")
    
    # Search for the table
    table = search_table(access_token, table_name)
    if not table:
        print("✗ Table not found")
        return
    
    # Get columns
    columns = get_table_columns(access_token, table)
    
    if not columns:
        print("✗ No columns found")
        return
    
    print(f"\nFound {len(columns)} columns\n")
    
    # Print the field for each column
    for col in columns:
        col_name = (col.get("summary", {}).get("core.name") or 
                   col.get("core.name") or
                   col.get("name", "Unknown"))
        
        # Try to get the field value from different locations
        field_value = None
        
        # Check in selfAttributes
        self_attrs = col.get("selfAttributes", {})
        if field_name in self_attrs:
            field_value = self_attrs[field_name]
        
        # Check in customAttributes
        if field_value is None:
            custom_attrs = col.get("customAttributes", {})
            if field_name in custom_attrs:
                field_value = custom_attrs[field_name]
        
        # Check in systemAttributes
        if field_value is None:
            sys_attrs = col.get("systemAttributes", {})
            if field_name in sys_attrs:
                field_value = sys_attrs[field_name]
        
        # Check at top level
        if field_value is None and field_name in col:
            field_value = col[field_name]
        
        print(f"{col_name}: {field_value if field_value is not None else 'N/A'}")


def main():
    """Main function"""
    try:
        # Authenticate
        session_id = login()
        access_token = generate_jwt_token(session_id)
        
        # Allow user to input table name or use default
        print("\nEnter table name (or press Enter for default 'AF2501T00'):")
        print("Examples: AF2501T00, dbo.AF2501T00, DATABASE.SCHEMA.TABLE")
        user_input = input("> ").strip()
        table_name = user_input if user_input else "AF2501T00"
        
        # Allow user to input field name or use default
        print("\nEnter field/attribute name to display (or press Enter for default 'Nivel de seguridad'):")
        print("Examples: Nivel de seguridad, core.classificationNames, dataType")
        user_field = input("> ").strip()
        field_name = user_field if user_field else "Nivel de seguridad"
        
        # Print the field for all attributes
        print_attribute_field(access_token, table_name, field_name)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
