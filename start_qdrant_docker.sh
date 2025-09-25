#!/opt/homebrew/bin/bash
# run from /Users/wgehring/Dropbox (Personal)/ai_2025/unified_embedder
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant