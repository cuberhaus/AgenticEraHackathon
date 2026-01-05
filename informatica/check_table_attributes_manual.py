"""
Alternative: Check Table Attributes from Manual Export

Since the Informatica Catalog APIs are not available in your sandbox,
this script works with manually exported table metadata.

Instructions:
1. In Informatica UI, find your table (AF2501T00)
2. Export the column/attribute information to CSV or Excel
3. Place the file in this directory
4. Update the file path below
5. Run this script
"""

import pandas as pd
import json

# Configuration - Update with your exported file path
METADATA_FILE = r"C:\Users\pcasacubertagil\Downloads\table_metadata.csv"  # or .xlsx

def check_attribute_values(file_path, attribute_name="Nivel de seguridad"):
    """
    Read exported table metadata and check attribute values
    
    Args:
        file_path: Path to CSV or Excel file with column metadata
        attribute_name: The attribute/column to check
    """
    print(f"\n{'='*60}")
    print(f"Reading metadata from: {file_path}")
    print(f"Checking attribute: {attribute_name}")
    print(f"{'='*60}\n")
    
    try:
        # Read the file (auto-detect CSV vs Excel)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("✗ Unsupported file format. Use CSV or Excel.")
            return
        
        print(f"✓ Loaded {len(df)} rows")
        print(f"\nAvailable columns in file:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        # Check if the attribute exists
        if attribute_name not in df.columns:
            print(f"\n✗ Attribute '{attribute_name}' not found in file")
            print(f"\nDid you mean one of these?")
            similar = [col for col in df.columns if attribute_name.lower() in col.lower()]
            for col in similar:
                print(f"  - {col}")
            return
        
        print(f"\n{'='*60}")
        print(f"Values for '{attribute_name}':")
        print(f"{'='*60}\n")
        
        # Assuming first column is the column/field name
        name_col = df.columns[0]
        
        for idx, row in df.iterrows():
            column_name = row[name_col]
            value = row[attribute_name]
            print(f"{column_name}: {value}")
        
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total columns: {len(df)}")
        
        # Value counts
        value_counts = df[attribute_name].value_counts()
        print(f"\nValue distribution:")
        for value, count in value_counts.items():
            print(f"  {value}: {count} columns")
        
        # Check if all have same value
        unique_values = df[attribute_name].nunique()
        if unique_values == 1:
            single_value = df[attribute_name].iloc[0]
            print(f"\n✓ All columns have the same value: '{single_value}'")
        else:
            print(f"\n✗ Columns have {unique_values} different values")
        
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        print("\nSteps to get the file:")
        print("1. Open Informatica Data Governance & Catalog")
        print("2. Find table 'AF2501T00'")
        print("3. Export column metadata to CSV or Excel")
        print("4. Update METADATA_FILE path in this script")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("Alternative approach: Using manually exported metadata")
    print("\nThis script works with CSV/Excel files exported from Informatica UI")
    
    # Check if user wants to specify a different file
    print(f"\nCurrent file path: {METADATA_FILE}")
    user_file = input("Enter different file path (or press Enter to use default): ").strip()
    
    file_path = user_file if user_file else METADATA_FILE
    
    # Ask for attribute name
    print("\nCommon attributes:")
    print("  - Nivel de seguridad")
    print("  - Data Classification")
    print("  - Data Type")
    print("  - Description")
    
    user_attr = input("\nEnter attribute name (or press Enter for 'Nivel de seguridad'): ").strip()
    attribute_name = user_attr if user_attr else "Nivel de seguridad"
    
    # Process the file
    check_attribute_values(file_path, attribute_name)


if __name__ == "__main__":
    main()
