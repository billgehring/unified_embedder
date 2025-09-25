#!/bin/bash

# Cross-platform unified embedder script
# Automatically detects OS and uses appropriate paths and configurations

# Load base .env if present (shared values). Host-specific overrides below.
if [[ -f ./.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use existing Dropbox paths
    BASE_DIR='/Users/wgehring/Dropbox (Personal)/ai_2025/LATEST_GREATEST'
    echo "🍎 Running on macOS with Dropbox paths"

    # Configure Tesseract for Homebrew installs to avoid path ambiguity
    if [[ -x "/opt/homebrew/bin/tesseract" ]]; then
        export TESSERACT_CMD="/opt/homebrew/bin/tesseract"
    elif [[ -x "/usr/local/bin/tesseract" ]]; then
        export TESSERACT_CMD="/usr/local/bin/tesseract"
    fi
    # Set tessdata prefix if not already provided in .env
    if [[ -z "${TESSDATA_PREFIX}" ]]; then
        if [[ -d "/opt/homebrew/share/tessdata" ]]; then
            export TESSDATA_PREFIX="/opt/homebrew/share/tessdata/"
        elif [[ -d "/usr/local/share/tessdata" ]]; then
            export TESSDATA_PREFIX="/usr/local/share/tessdata/"
        fi
    fi
    echo "🔎 Tesseract cmd: ${TESSERACT_CMD:-auto}"
    echo "🔎 TESSDATA_PREFIX: ${TESSDATA_PREFIX:-unset}"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Linux - use home directory paths
    BASE_DIR="$HOME/LATEST_GREATEST"
    echo "🐧 Running on Ubuntu/Linux with home directory paths"

    # Prefer system tesseract unless overridden by .env
    if command -v tesseract >/dev/null 2>&1 && [[ -z "${TESSERACT_CMD}" ]]; then
        export TESSERACT_CMD="$(command -v tesseract)"
    fi
    if [[ -z "${TESSDATA_PREFIX}" ]]; then
        if [[ -d "/usr/share/tesseract-ocr/4.00/tessdata" ]]; then
            export TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata/"
        elif [[ -d "/usr/share/tesseract-ocr/tessdata" ]]; then
            export TESSDATA_PREFIX="/usr/share/tesseract-ocr/tessdata/"
        fi
    fi
    # CUDA visibility defaults for dual RTX8000; override via env if needed
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    # Tokenizers parallelism often causes noisy warnings
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
    echo "🔎 Tesseract cmd: ${TESSERACT_CMD:-auto}"
    echo "🔎 TESSDATA_PREFIX: ${TESSDATA_PREFIX:-unset}"
    echo "🔎 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

# Set paths based on detected OS
CANVAS_DATA_DIR="${BASE_DIR}/canvas_scraper_claude/canvas_data"

### Qdrant Database Configuration
# On macOS: Uses local Docker or ./start_qdrant_docker.sh
# On Ubuntu: Uses existing Docker instance with ~/LATEST_GREATEST/qdrant_storage mount
# Start command for Ubuntu: docker run -p 6333:6333 -p 6334:6334 -v "$HOME/LATEST_GREATEST/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

### GPU Optimization Settings
# Automatically detects and configures for:
# - Dual RTX8000 + NVLink (ultimate performance)
# - Single/Multi RTX8000 GPUs
# - Other CUDA GPUs
# - Apple Silicon MPS
# - CPU-only fallback
# Performance optimizer handles all GPU detection and optimization automatically

echo "📁 Using base directory: ${BASE_DIR}"
echo "📚 Canvas data directory: ${CANVAS_DATA_DIR}"

# ColBERT multivector upload tuning (avoid large HTTP payloads/timeouts)
# Use small batch size by default; override by exporting COLBERT_UPLOAD_BATCH
export COLBERT_UPLOAD_BATCH=${COLBERT_UPLOAD_BATCH:-5}
echo "🔧 ColBERT upload batch size: ${COLBERT_UPLOAD_BATCH}"

# Verify paths exist
if [[ ! -d "$BASE_DIR" ]]; then
    echo "❌ Error: Base directory not found: ${BASE_DIR}"
    echo "   Please ensure the LATEST_GREATEST directory exists at the expected location."
    exit 1
fi

if [[ ! -d "$CANVAS_DATA_DIR" ]]; then
    echo "⚠️  Warning: Canvas data directory not found: ${CANVAS_DATA_DIR}"
    echo "   Some commands may fail if data directory doesn't exist."
fi

# Course IDs for reference
# 701917 is Fall 2024 Psychology 240
# 755719 is Spring 2025 Psychology 240  
# 699916 is Fall 2024 Cognitive Science 200

#============================================================================
# EXAMPLE COMMANDS - Uncomment and modify as needed
#============================================================================

# Basic test with small file limit
#uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917" --qdrant --qdrant_collection "psych_240_test" --qdrant_url "http://localhost:6333" --max_files 5 --force_ocr --debug

# Production embedding with ColBERT tokens - Psychology 240 Fall 2024 PDFs
# This command will automatically detect and optimize for your GPU setup

if [[ "$OSTYPE" == "darwin"* ]]; then

# skip troublesome Turing reading
export SKIP_PATTERNS="*OPTIONAL.pdf,*slides.pdf"
export DOCLING_DO_FORMULA_ENRICHMENT=0
export CHUNKER_MAX_TOKENS=512
export CHUNKER_TOKENIZER=sentence-transformers/all-MiniLM-L6-v2

# no longer --force_ocr because tesseract Orientation and Script detection fails on widescreen slides that are exported to PDFs.

# uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_699916" --qdrant --qdrant_collection "COGSCI200_FA2024_699916_ColBERT" --qdrant_url "http://localhost:6333"  --ocr_engine "Tesseract" --max_files 99999 --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0" 

# elif [[ "$OSTYPE" == "linux-gnu"* ]]; then

uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917" --qdrant --qdrant_collection "Psych240_FA2024_701917_ColBERT" --qdrant_url "http://localhost:6333" --ocr_engine "Tesseract" --max_files 99999 --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0" 

fi 

# reduced version for testing, PDFs only
# uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917/PDF readings (numbered according to lecture #)" --qdrant --qdrant_collection "Psych240_FA2024_701917_ColBERT" --qdrant_url "http://localhost:6333" --force_ocr --ocr_engine "Tesseract" --max_files 3 --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0" 



#  - Small end-to-end run (uses chunks, central FastEmbed, and Qdrant writes):
#       - Start Qdrant: ./start_qdrant_docker.sh
#       - Example (adjust path/collection):
#         uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917/
#   PDF readings (numbered according to lecture #)" --qdrant --qdrant_collection
#   "Psych240_FA2024_ColBERT_Prep" --qdrant_url "http://localhost:6333" --max_files 4
#   --force_ocr --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" --debug
#   - Then run:
#       - uv run test_qdrant_connectivity.py
#       - uv run test_retrieval.py <collection> (or ./test_retrieval.sh <collection>)


# #> ./logs/out_embedder_PDF_ColBERT.txt 2> ./logs/err_embedder_PDF_ColBERT.txt

# uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917/PDF readings (numbered according to lecture #)" --qdrant --qdrant_collection "Psych240_FA2024_701917_PDF_ColBERT" --qdrant_url "http://localhost:6333" --force_ocr --max_files 1 --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0" 


# Full course embedding - Psychology 240 Fall 2024
#uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917" --qdrant --qdrant_collection "Psych240_FA2024_701917" --qdrant_url "http://localhost:6333" --force_ocr --max_files 999999 --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" > ./logs/out_embedder701917.txt 2> ./logs/err_embedder701917.txt


# Full course embedding - Psychology 240 Spring 2025
#uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_755719" --qdrant --qdrant_collection "PSYCH240_SP2025_755719" --qdrant_url "http://localhost:6333" --force_ocr --max_files 999999 --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" > ./logs/out_embedder755719.txt 2> ./logs/err_embedder755719.txt

# ColBERT test with small dataset
#uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917" --qdrant --qdrant_collection "Psych240_FA2024_701917_ColBERT_Test" --qdrant_url "http://localhost:6333" --force_ocr --max_files 3 --recreate_index --embedding_model "BAAI/bge-large-en-v1.5" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0" --debug > ./logs/out_embedder_colbert_test.txt 2> ./logs/err_embedder_colbert_test.txt

#============================================================================
# ADDITIONAL EXAMPLES
#============================================================================

# Test with local PDF directory (adjust path as needed)
#uv run unified_embedder.py --docs_dir "./test_pdfs" --qdrant --qdrant_collection "test_pdf_collection" --qdrant_url "http://localhost:6333" --max_files 5 --force_ocr --ocr_engine "Tesseract" --debug

# Quick single-PDF OCR smoke test (no Qdrant):
#uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_699916/Discussion Section Materials/Logan/Week 4" --max_files 1 --force_ocr --ocr_engine "Tesseract" --debug

# Remote Qdrant API example (replace with your API key)  
#uv run unified_embedder.py --docs_dir "${CANVAS_DATA_DIR}/course_701917" --qdrant --qdrant_collection "remote_test" --force_ocr --max_files 20 --debug --qdrant_api_key "your-api-key-here"

# Ubuntu-specific setup recommendations
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo ""
    echo "🔧 Ubuntu RTX8000 Setup Recommendations:"
    echo "   1. Ensure NVIDIA drivers are installed: nvidia-smi"
    echo "   2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit"
    echo "   3. Install Tesseract for OCR: sudo apt-get install tesseract-ocr"
    echo "   4. Optional: Install pynvml for advanced GPU monitoring: pip install pynvml"
    echo "   5. For dual GPU setup, ensure NVLink is properly configured"
    echo ""
fi

echo "✅ Script configured for $OSTYPE"
echo "💡 Uncomment and modify commands above as needed"
echo "🚀 Performance optimizer will automatically detect and configure GPU optimizations"
