#!/usr/bin/env python3
"""
Qdrant Diagnostic Script

This script helps diagnose issues with Qdrant collections, particularly
the dashboard browser error you encountered.
"""

import argparse
import logging
import sys
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("qdrant_diagnostics")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError as e:
    logger.error(f"Required libraries not installed: {e}")
    logger.error("Please run: uv add qdrant-client")
    sys.exit(1)


def diagnose_qdrant_instance(qdrant_url: str = "http://localhost:6333", 
                           qdrant_api_key: str = None):
    """Diagnose Qdrant instance and collections for dashboard issues."""
    
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
            timeout=60  # Extended timeout for diagnostic operations
        )
        
        logger.info(f"Connected to Qdrant at {qdrant_url}")
        
        # Get all collections
        collections = client.get_collections()
        logger.info(f"Total collections: {len(collections.collections)}")
        
        problematic_collections = []
        
        for collection in collections.collections:
            collection_name = collection.name
            logger.info(f"\n=== Analyzing Collection: {collection_name} ===")
            
            try:
                # Get detailed collection info
                collection_info = client.get_collection(collection_name)
                logger.info(f"Status: {collection_info.status}")
                logger.info(f"Points count: {collection_info.points_count}")
                
                if collection_info.points_count == 0:
                    logger.warning(f"Collection '{collection_name}' is empty")
                    continue
                
                # Try to scroll through points to check for data corruption
                try:
                    scroll_result = client.scroll(
                        collection_name=collection_name,
                        limit=1,
                        with_payload=True,
                        with_vectors=True
                    )
                    
                    if scroll_result[0]:
                        point = scroll_result[0][0]
                        logger.info(f"Sample point ID: {point.id}")
                        logger.info(f"Payload size: {len(point.payload) if point.payload else 0} keys")
                        
                        # Check for potential issues
                        issues = []
                        
                        # Check payload
                        if point.payload:
                            for key, value in point.payload.items():
                                if isinstance(value, (bytes, memoryview)):
                                    issues.append(f"Binary data in payload key '{key}'")
                                elif isinstance(value, str) and len(value) > 10000:
                                    issues.append(f"Very large string in payload key '{key}' ({len(value)} chars)")
                        
                        # Check vectors
                        if point.vector:
                            if isinstance(point.vector, dict):
                                for vec_name, vec_data in point.vector.items():
                                    try:
                                        vec_len = len(vec_data)
                                        if vec_len == 0:
                                            issues.append(f"Empty vector '{vec_name}'")
                                    except:
                                        issues.append(f"Invalid vector data in '{vec_name}'")
                            else:
                                try:
                                    vec_len = len(point.vector)
                                    if vec_len == 0:
                                        issues.append("Empty main vector")
                                except:
                                    issues.append("Invalid main vector data")
                        
                        if issues:
                            logger.warning(f"Potential issues found in '{collection_name}':")
                            for issue in issues:
                                logger.warning(f"  - {issue}")
                            problematic_collections.append({
                                'name': collection_name,
                                'issues': issues
                            })
                        else:
                            logger.info(f"Collection '{collection_name}' appears healthy")
                    
                    else:
                        logger.warning(f"Could not retrieve points from '{collection_name}' despite non-zero count")
                        problematic_collections.append({
                            'name': collection_name,
                            'issues': ['Cannot retrieve points despite non-zero count']
                        })
                        
                except Exception as scroll_error:
                    logger.error(f"Error scrolling collection '{collection_name}': {scroll_error}")
                    problematic_collections.append({
                        'name': collection_name,
                        'issues': [f'Scroll error: {scroll_error}']
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing collection '{collection_name}': {e}")
                problematic_collections.append({
                    'name': collection_name,
                    'issues': [f'Analysis error: {e}']
                })
        
        # Summary
        logger.info(f"\n=== DIAGNOSTIC SUMMARY ===")
        logger.info(f"Total collections analyzed: {len(collections.collections)}")
        logger.info(f"Problematic collections: {len(problematic_collections)}")
        
        if problematic_collections:
            logger.warning("Collections with potential issues:")
            for prob_col in problematic_collections:
                logger.warning(f"  - {prob_col['name']}: {len(prob_col['issues'])} issue(s)")
                for issue in prob_col['issues']:
                    logger.warning(f"    * {issue}")
            
            logger.info("\n=== RECOMMENDATIONS ===")
            logger.info("The Qdrant dashboard error you encountered might be due to:")
            logger.info("1. Binary/corrupted data in payloads")
            logger.info("2. Malformed vector data")
            logger.info("3. Empty vectors with non-zero point counts")
            logger.info("4. Very large payload data causing browser timeouts")
            
            logger.info("\nTo fix dashboard issues, consider:")
            logger.info("1. Recreating problematic collections")
            logger.info("2. Re-running unified_embedder.py with --recreate_index")
            logger.info("3. Using this CLI tool instead of the dashboard for debugging")
        else:
            logger.info("All collections appear healthy!")
        
    except Exception as e:
        logger.error(f"Failed to connect or diagnose Qdrant: {e}")
        return False
        
    return True


def clean_collection(client: QdrantClient, collection_name: str):
    """Clean a problematic collection by recreating it."""
    try:
        logger.info(f"Cleaning collection '{collection_name}'...")
        
        # Get collection info first
        collection_info = client.get_collection(collection_name)
        
        # Delete and recreate
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
        
        # Note: This would need to be recreated by running unified_embedder.py again
        logger.info(f"Collection '{collection_name}' deleted. Re-run unified_embedder.py to recreate with clean data.")
        
    except Exception as e:
        logger.error(f"Error cleaning collection '{collection_name}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Qdrant collections and dashboard issues")
    parser.add_argument("--qdrant_url", default="http://localhost:6333", help="Qdrant server URL")
    parser.add_argument("--qdrant_api_key", help="Qdrant API key")
    parser.add_argument("--clean", help="Clean/recreate a specific collection (WARNING: deletes data)")
    
    args = parser.parse_args()
    
    logger.info("Starting Qdrant diagnostics...")
    
    if args.clean:
        logger.warning(f"This will DELETE collection '{args.clean}' and all its data!")
        confirmation = input("Type 'yes' to confirm: ")
        if confirmation.lower() == 'yes':
            client = QdrantClient(
                url=args.qdrant_url,
                api_key=args.qdrant_api_key if args.qdrant_api_key else None,
                timeout=60  # Extended timeout for cleanup operations
            )
            clean_collection(client, args.clean)
        else:
            logger.info("Operation cancelled.")
        return
    
    # Run diagnostics
    success = diagnose_qdrant_instance(args.qdrant_url, args.qdrant_api_key)
    
    if success:
        logger.info("Diagnostics completed successfully!")
    else:
        logger.error("Diagnostics failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()