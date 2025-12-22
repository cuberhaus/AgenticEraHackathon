import requests
import json
import pandas as pd

# Configuration
LOGIN_URL = "https://dm-em.informaticacloud.com"
MARKETPLACE_URL = "https://idmc-api.dm-em.informaticacloud.com/data360/marketplace"
USERNAME = "pcasacuberta_sandbox"
PASSWORD = "pdeloitte79#"

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
        
        # Dictionary to store created domains {domain_name: domain_id}
        created_domains = {}
        
        # Statistics
        stats = {
            "domains_created": 0,
            "subdomains_created": 0,
            "errors": 0,
            "skipped": 0
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
                    # Create as domain (top-level category)
                    domain = create_domain(jwt_token, org_id, session_id, name, description)
                    if domain:
                        created_domains[name] = domain.get('id')
                        stats["domains_created"] += 1
                
                else:
                    # This is a subdomain
                    # First, ensure parent domain exists
                    if parent_domain not in created_domains:
                        print(f"  Creating parent domain first: {parent_domain}")
                        parent = create_domain(jwt_token, org_id, session_id, parent_domain, "-")
                        if parent:
                            created_domains[parent_domain] = parent.get('id')
                            stats["domains_created"] += 1
                    
                    # Now create the subdomain
                    if parent_domain in created_domains:
                        parent_id = created_domains[parent_domain]
                        subdomain = create_subdomain(jwt_token, org_id, session_id, parent_id, name, description)
                        if subdomain:
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
        print(f"Domains created:    {stats['domains_created']}")
        print(f"Subdomains created: {stats['subdomains_created']}")
        print(f"Errors:             {stats['errors']}")
        print(f"Skipped:            {stats['skipped']}")
        print(f"Total processed:    {stats['domains_created'] + stats['subdomains_created']}")
        
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
