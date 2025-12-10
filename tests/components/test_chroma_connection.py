import os
import sys
# Add project root to path so we can import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from components.db.chroma_db import ChromaDb
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()



def test_connection():
    print("--- Testing ChromaDB Connection ---")
    
    host = os.environ.get("CHROMA_HOST")
    port = int(os.environ.get("CHROMA_PORT", 8000))
    token = os.environ.get("CHROMA_TOKEN")
    ssl = os.environ.get("CHROMA_SSL", "False").lower() == "true"
    tenant = os.environ.get("CHROMA_CLOUD_TENANT") or os.environ.get("CHROMA_TENET")
    database = os.environ.get("CHROMA_CLOUD_DATABASE") or os.environ.get("CHROMA_DATABASE")

    
    
    print("Configuration:")
    print(f"  Host: {host if host else 'Local (PersistentClient)'}")
    print(f"  Port: {port}")
    print(f"  SSL: {ssl}")
    print(f"  Token: {'******' if token else 'None'}")
    print(f"  Tenant: {tenant}")
    print(f"  Database: {database}")


    headers = None
    if token:
        headers = {"X-Chroma-Token": token}
    try:
        # Initialize
        print("\nInitializing ChromaDb client...")
        db = ChromaDb(
            persist_directory="./chroma_store",
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            api_key=token,
            tenant=tenant,
            database=database
        )
        
        # Test 1: Access Collection
        collection_name = "test_connection_check"
        print(f"Creating/Getting collection '{collection_name}'...")
        db.get_or_create(collection_name)
        print("  Success: Collection accessed.")

        # Test 2: Add Data
        print("Adding a test document...")
        # Using a small dimension for dummy embedding (e.g., 10) 
        # Note: If the collection already exists with different dim, this might fail.
        # So we should probably delete it first to be sure.
        try:
            db.delete_collection(collection_name)
            # Re-create
            db.get_or_create(collection_name)
        except Exception:
            pass # Collection might not exist

        dummy_dim = 10
        db.add(
            collection=collection_name,
            ids=["test_id_1"],
            documents=["This is a test document to verify connection."],
            embeddings=[[0.1] * dummy_dim], 
            metadatas=[{"source": "test_script"}]
        )
        print("  Success: Document added.")

        # Test 3: Query
        print("Querying the test document by embedding...")
        results = db.query_by_embedding(
            collection=collection_name,
            query_embedding=[0.1] * dummy_dim,
            n_results=1
        )
        
        if results and results[0].id:
            print(f"  Success: Retrieved document ID {results[0].id}")
        else:
            print("  Warning: Query returned no results.")

        # Cleanup
        print("Cleaning up (deleting test collection)...")
        db.delete_collection(collection_name)
        print("  Success: Collection deleted.")

        print("\n--- CONNECTION TEST PASSED ---")

    except Exception as e:
        print("\n--- CONNECTION TEST FAILED ---")
        print(f"Error: {e}")
        traceback.print_exc()

def test_cloud_connection():
    print("\n--- Testing ChromaDB Cloud Connection ---")
    
    api_key = os.environ.get("CHROMA_TOKEN")
    tenant = os.environ.get("CHROMA_CLOUD_TENANT") or os.environ.get("CHROMA_TENET")
    database = os.environ.get("CHROMA_CLOUD_DATABASE") or os.environ.get("CHROMA_DATABASE")
    
    if not (api_key and tenant and database):
        print("Skipping Cloud test: Missing CHROMA_TOKEN, CHROMA_CLOUD_TENANT, or CHROMA_CLOUD_DATABASE")
        return

    print("Cloud Configuration:")
    print(f"  Tenant: {tenant}")
    print(f"  Database: {database}")
    print(f"  API Key: {'******' if api_key else 'None'}")

    try:
        print("\nInitializing ChromaDb client (Cloud)...")
        db = ChromaDb(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        
        collection_name = "test_cloud_connection_check"
        print(f"Creating/Getting collection '{collection_name}'...")
        db.get_or_create(collection_name)
        print("  Success: Collection accessed.")
        
        # Add a dummy doc to verify write access
        print("Adding a test document...")
        db.add(
            collection=collection_name,
            ids=["cloud_test_id_1"],
            documents=["This is a cloud test document."],
            embeddings=[[0.1] * 10], 
            metadatas=[{"source": "cloud_test_script"}]
        )
        print("  Success: Document added.")

        # Cleanup
        print("Cleaning up (deleting test collection)...")
        db.delete_collection(collection_name)
        print("  Success: Collection deleted.")

        print("\n--- CLOUD CONNECTION TEST PASSED ---")

    except Exception as e:
        print("\n--- CLOUD CONNECTION TEST FAILED ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()
    test_cloud_connection()
    
