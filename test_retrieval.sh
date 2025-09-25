#!/opt/homebrew/bin/bash

# Test retrieval from document stores created by unified_embedder.py

# Default collection (can be overridden)
COLLECTION=${1:-"test_fixes"}

echo "Testing retrieval from Qdrant collection: $COLLECTION"
echo "========================================"

# Run diagnostics first
echo "Running collection diagnostics..."
uv run test_retrieval.py --collection "$COLLECTION" --diagnose

echo ""
echo "Running retrieval tests..."
echo "=========================="

# Test with different queries
QUERIES=(
   "anterograde amnesia"
   "concepts"
)

for query in "${QUERIES[@]}"; do
    echo ""
    echo "Testing query: '$query'"
    echo "----------------------------"
    uv run test_retrieval.py \
        --collection "$COLLECTION" \
        --query "$query" \
        --limit 2 \
        --samples 3
    
    echo "Press Enter to continue to next query (or Ctrl+C to exit)..."
    read
done

echo ""
echo "Testing completed!"