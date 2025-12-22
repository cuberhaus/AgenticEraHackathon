import requests
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
LOGIN_URL = "https://dm-em.informaticacloud.com"
MARKETPLACE_URL = "https://idmc-api.dm-em.informaticacloud.com/data360/marketplace"
USERNAME = os.getenv("INFORMATICA_USERNAME")
PASSWORD = os.getenv("INFORMATICA_PASSWORD")

# Excel file path - Update this with your file path
EXCEL_FILE = r"C:\Users\pcasacubertagil\Downloads\dominios_subdominios.xlsx"

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

def get_existing_categories(jwt_token, org_id, session_id):
    """Fetch all existing categories from the marketplace"""
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "X-INFA-ORG-ID": org_id,
        "IDS-SESSION-ID": session_id,
        "Accept": "application/json"
    }
    
    try:
        category_map = {}
        offset = 0
        limit = 100
        total_count = None
        
        # Fetch all pages of categories
        while True:
            marketplace_url = f"{MARKETPLACE_URL}/api/v1/categories?offset={offset}&limit={limit}"
            response = requests.get(marketplace_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Get total count from first response
            if total_count is None:
                total_count = data.get('totalCount', 0)
                print(f"  Total categories in marketplace: {total_count}")
            
            # Process objects in this page
            if 'objects' in data and isinstance(data['objects'], list):
                for cat in data['objects']:
                    if cat.get('name'):
                        category_map[cat.get('name')] = cat
                
                # Check if we got all categories
                offset += len(data['objects'])
                if offset >= total_count or len(data['objects']) == 0:
                    break
            else:
                break
        
        print(f"✓ Loaded {len(category_map)} existing categories")
        if len(category_map) > 0:
            print(f"  Sample categories: {list(category_map.keys())[:5]}")
        
        return category_map
    except Exception as e:
        print(f"⚠ Could not fetch existing categories: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_domain(jwt_token, org_id, session_id, domain_name, domain_description=""):
    """Create a domain (parent category)"""
    
    marketplace_url = f"{MARKETPLACE_URL}/api/v1/categories"
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "X-INFA-ORG-ID": org_id,
        "IDS-SESSION-ID": session_id,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    domain_payload = {
        "name": domain_name,
        "description": domain_description,
        "isActive": True
    }
    
    print(f"  Creating domain: {domain_name}")
    
    try:
        response = requests.post(marketplace_url, json=domain_payload, headers=headers)
        response.raise_for_status()
        
        domain_data = response.json()
        print(f"    ✓ Domain created - ID: {domain_data.get('id')}")
        
        return domain_data
    except requests.exceptions.HTTPError as e:
        print(f"    ✗ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"    Response: {e.response.text[:300]}")
        return None
    except Exception as e:
        print(f"    ✗ Error: {str(e)[:100]}")
        return None

def create_subdomain(jwt_token, org_id, session_id, parent_domain_id, subdomain_name, subdomain_description=""):
    """Create a subdomain under a parent domain"""
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "X-INFA-ORG-ID": org_id,
        "IDS-SESSION-ID": session_id,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Try with parentId in payload
    subdomain_payload = {
        "name": subdomain_name,
        "description": subdomain_description,
        "parentId": parent_domain_id,
        "isActive": True
    }
    
    print(f"    Creating subdomain: {subdomain_name}")
    
    try:
        marketplace_url = f"{MARKETPLACE_URL}/api/v1/categories"
        response = requests.post(marketplace_url, json=subdomain_payload, headers=headers)
        response.raise_for_status()
        
        subdomain_data = response.json()
        print(f"      ✓ Subdomain created - ID: {subdomain_data.get('id')}")
        
        return subdomain_data
    except requests.exceptions.HTTPError as e:
        print(f"      ✗ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"      Response: {e.response.text[:300]}")
        return None
    except Exception as e:
        print(f"      ✗ Error: {str(e)[:100]}")
        return None

def read_excel_and_create_hierarchy(jwt_token, org_id, session_id, excel_file):
    """Read Excel file and create domain/subdomain hierarchy"""
    
    print(f"\nReading Excel file: {excel_file}")
    
    try:
        # Try reading with different options
        # First, check all sheets
        excel_data = pd.ExcelFile(excel_file)
        print(f"✓ Found {len(excel_data.sheet_names)} sheet(s): {excel_data.sheet_names}")
        
        # Read from the "Subdomain" sheet
        sheet_name = 'Subdomain'
        print(f"\nReading from sheet: {sheet_name}")
        
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        print(f"✓ Found {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Show first few rows for debugging
        print("\nFirst 5 rows:")
        print(df.head(5).to_string())
        
        # Get existing categories first
        print("\nFetching existing categories...")
        existing_categories = get_existing_categories(jwt_token, org_id, session_id)
        
        # Dictionary to store created/existing domains {domain_name: domain_id}
        created_domains = {}
        
        # Dictionary to track all categories (including subdomains) to avoid duplicates
        all_categories = {}
        
        # Add existing categories to the maps
        for name, cat in existing_categories.items():
            created_domains[name] = cat.get('id')
            all_categories[name] = cat.get('id')
        
        print(f"\n✓ Pre-loaded {len(created_domains)} existing categories into cache")
        
        # Statistics
        stats = {
            "domains_created": 0,
            "subdomains_created": 0,
            "errors": 0,
            "skipped": 0,
            "domains_already_exist": 0,
            "subdomains_already_exist": 0
        }
        
        # Process each row
        print("\n" + "="*60)
        print("Creating domains and subdomains...")
        print("="*60)
        
        for index, row in df.iterrows():
            try:
                name = str(row.get('Name', '')).strip()
                description = str(row.get('Description', '')).strip()
                parent_subdomain = str(row.get('Parent: Subdomain', '')).strip()
                parent_domain = str(row.get('Parent: Domain', '')).strip()
                operation = str(row.get('Operation', 'Create')).strip()
                
                # Replace 'nan' with '-' in description
                if description == 'nan' or description == '':
                    description = '-'
                
                # Skip if operation is not Create
                if operation.lower() != 'create':
                    stats['skipped'] += 1
                    continue
                
                # Skip if name is empty or nan or -
                if not name or name == 'nan' or name == '' or name == '-':
                    stats['skipped'] += 1
                    continue
                
                print(f"\n[Row {index + 1}] {name}")
                print(f"  Parent Domain: {parent_domain if parent_domain and parent_domain not in ['nan', '', '-'] else 'None (this is a domain)'}")
                
                # If there's no parent domain, this IS a domain
                if not parent_domain or parent_domain in ['nan', '', '-']:
                    # Check if domain already exists
                    if name in created_domains:
                        print(f"  ⚠ Domain already exists - ID: {created_domains[name]}")
                        stats["domains_already_exist"] += 1
                    else:
                        # Create as domain (top-level category)
                        domain = create_domain(jwt_token, org_id, session_id, name, description)
                        if domain:
                            created_domains[name] = domain.get('id')
                            all_categories[name] = domain.get('id')
                            stats["domains_created"] += 1
                
                else:
                    # This is a subdomain
                    # First, ensure parent domain exists
                    if parent_domain not in created_domains:
                        print(f"  Creating parent domain first: {parent_domain}")
                        parent = create_domain(jwt_token, org_id, session_id, parent_domain, "-")
                        if parent:
                            created_domains[parent_domain] = parent.get('id')
                            all_categories[parent_domain] = parent.get('id')
                            stats["domains_created"] += 1
                    else:
                        print(f"  ⚠ Parent domain already exists - ID: {created_domains[parent_domain]}")
                    
                    # Check if subdomain already exists
                    if name in all_categories:
                        print(f"  ⚠ Subdomain '{name}' already exists - ID: {all_categories[name]}")
                        stats["subdomains_already_exist"] += 1
                    else:
                        # Now create the subdomain
                        if parent_domain in created_domains:
                            parent_id = created_domains[parent_domain]
                            subdomain = create_subdomain(jwt_token, org_id, session_id, parent_id, name, description)
                            if subdomain:
                                all_categories[name] = subdomain.get('id')
                                stats["subdomains_created"] += 1
                            else:
                                stats["errors"] += 1
                        else:
                            print(f"    ✗ Parent domain '{parent_domain}' could not be created")
                            stats["errors"] += 1
                        
            except Exception as e:
                print(f"  ✗ Error processing row {index + 1}: {e}")
                stats["errors"] += 1
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Domains created:        {stats['domains_created']}")
        print(f"Domains already exist:  {stats['domains_already_exist']}")
        print(f"Subdomains created:     {stats['subdomains_created']}")
        print(f"Subdomains already exist: {stats['subdomains_already_exist']}")
        print(f"Errors:                 {stats['errors']}")
        print(f"Skipped:                {stats['skipped']}")
        print(f"Total processed:        {stats['domains_created'] + stats['subdomains_created']}")
        
    except FileNotFoundError:
        print(f"✗ Excel file not found: {excel_file}")
        print(f"Please update the EXCEL_FILE path at the top of the script")
    except Exception as e:
        print(f"✗ Error reading Excel: {e}")
        import traceback
        traceback.print_exc()

def main():
    try:
        # 1. Login
        session_id, org_id = login()
        
        # 2. Generate JWT token
        print("\n" + "="*60)
        jwt_token = generate_jwt_token(session_id)
        
        # 3. Read Excel and create hierarchy
        print("\n" + "="*60)
        read_excel_and_create_hierarchy(jwt_token, org_id, session_id, EXCEL_FILE)
        
        print("\n" + "="*60)
        print("✓ Process completed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
