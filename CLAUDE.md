# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. 

## Repository Overview

This is a unified embedding pipeline for Canvas data that embeds text, PDF, DOCX, PPTX files into a Qdrant vector database. The system handles OCR for PDFs with quality checking and re-processing, and supports both dense and sparse embeddings for hybrid search.

See PROJECT_OVERVIEW.md for details.

### 🔗 Sister Project: shared-rag-service
**Location**: `../shared-rag-service/`
**Purpose**: FastAPI service that provides RAG queries to the Qdrant collections created by this project
**Relationship**: Producer → Consumer (this project creates databases, service queries them)

**Integration Points**:
- **Qdrant Collections**: Service queries collections created by this embedder
- **Vector Storage**: Service consumes dense, sparse, and ColBERT embeddings
- **Configuration**: Shared Qdrant connection settings and collection naming
- **Client Applications**: Service powers Enchanted Forest Tutor, av4 chat, and SlideQuest

See `../shared-rag-service/CLAUDE.md` for complete RAG service documentation.

## Core Architecture

### Main Processing Pipeline
- **unified_embedder.py**: Main processing pipeline that handles file discovery, parallel processing, text extraction with OCR, embedding generation, and Qdrant storage
- **enhanced_docling_converter.py**: Enhanced Docling converter with OCR quality assessment and re-processing capabilities
- **ocr_quality_checker.py**: OCR quality analysis and engine comparison utilities
- **qti-to-text-converter.py**: QTI XML to text conversion utility

### ColBERT Multi-Vector Architecture
- **colbert_token_embedder.py**: ColBERT token-level embedding generation using FastEmbed
- **hybrid_qdrant_store.py**: Hybrid storage combining Haystack (dense+sparse) with native Qdrant (ColBERT tokens)
- **multi_vector_retrieval.py**: Advanced retrieval pipeline with query classification and score fusion
- **performance_benchmark.py**: Comprehensive benchmarking suite for <50ms retrieval validation
- **performance_monitor.py**: Real-time performance monitoring and alerting system
- **monitored_retrieval.py**: Production-ready wrapper with integrated monitoring

## Key Dependencies

The project uses uv for dependency management with these main libraries:
- Haystack AI framework for document processing and embeddings
- Docling for document conversion with OCR support
- Qdrant for vector database storage
- FastEmbed for embedding generation
- Sentence Transformers for text embeddings


## Common Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Install required spaCy models (as noted in README.md)
uv run -m spacy download en_core_web_sm en_core_web_lg xx_ent_wiki_sm en_core_web_trf
```

### Running the Embedder

#### Standard Embedding (Dense + Sparse)
```bash
# Basic local run with Qdrant
uv run unified_embedder.py --docs_dir "./canvas_data/course_123" --qdrant --qdrant_collection "collection_name" --qdrant_url "http://localhost:6333" --max_files 5 --force_ocr --debug

# Production run without file limit
uv run unified_embedder.py --docs_dir "/path/to/canvas/data" --qdrant --qdrant_collection "collection_name" --qdrant_url "http://localhost:6333" --force_ocr
```

#### ColBERT Token Integration (NEW)
```bash
# Enable ColBERT token storage for advanced retrieval
uv run unified_embedder.py --docs_dir "./canvas_data/course_123" --qdrant --qdrant_collection "collection_name" --qdrant_url "http://localhost:6333" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0" --max_files 5 --debug

# Production ColBERT deployment
uv run unified_embedder.py --docs_dir "/path/to/canvas/data" --qdrant --qdrant_collection "collection_name" --qdrant_url "http://localhost:6333" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0" --force_ocr --embedding_model "BAAI/bge-large-en-v1.5"
```

### Qdrant Database
```bash
# Start Qdrant Docker container
./start_qdrant_docker.sh

# Or manually:
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

## Configuration

- Environment variables can be set in `.env` file
- OCR engine selection via `--ocr_engine` parameter (default: Tesseract)
- Force OCR processing with `--force_ocr` flag
- Embedding models configurable via `--embedding_model` and `--sparse_embedding_model`
- Chunk size and overlap configurable via `--chunk_size` and `--chunk_overlap`

## File Processing

The system processes:
- **Embeddable files**: .txt, .pdf, .docx, .pptx, .xml
- **Image files**: .png, .jpg, .jpeg (metadata only)
- **Transcript indices**: Detected heuristically and skipped for embedding
- **Metadata files**: `_meta.json` files are automatically loaded and merged

## Parallel Processing

The system uses multiprocessing for file processing with automatic worker count determination based on system resources. Each worker process loads its own CPU-only embedder to avoid GPU/MPS conflicts on Apple Silicon.

## Logging

Comprehensive logging to both console and timestamped log files in `./logs/` directory. Use `--debug` flag for detailed debugging information including OCR processing details.

## Testing and Retrieval

### Retrieval Testing Scripts

- **`test_retrieval.py`**: Test similarity search and document retrieval from Qdrant collections
- **`test_retrieval.sh`**: Automated testing script with multiple queries
- **`diagnose_qdrant.py`**: Diagnose Qdrant collections and identify dashboard issues

### Common Testing Commands

```bash
# Test retrieval from a specific collection
uv run test_retrieval.py --collection "collection_name" --query "search terms"

# Run automated tests
./test_retrieval.sh collection_name

# Diagnose Qdrant issues (fixes dashboard errors)
uv run diagnose_qdrant.py
```

### Known Issues

**Qdrant Dashboard Error**: If you encounter `Service internal error: OutputTooSmall` in the Qdrant dashboard, this is caused by malformed sparse vector data. Use the diagnostic script to identify and fix problematic collections:

```bash
# Identify problematic collections
uv run diagnose_qdrant.py

# Clean a specific collection (WARNING: deletes data)
uv run diagnose_qdrant.py --clean "collection_name"
```

The retrieval scripts work correctly even when the dashboard fails, providing a reliable alternative for querying your embedded documents.

## Task Management with TodoWrite

This project uses the TodoWrite tool extensively for task planning, tracking, and completion. All task activity is logged in `TodoWrite_Log.md` for visibility and historical reference.

### When to Use TodoWrite

**ALWAYS use TodoWrite for:**
- Complex multi-step tasks requiring 3+ distinct steps
- Non-trivial tasks requiring careful planning or multiple operations
- When user provides multiple tasks (numbered or comma-separated)
- Before starting any significant development work
- When breaking down large features into manageable steps

**Task Management Workflow:**
1. **Plan**: Create todos immediately when starting complex tasks
2. **Track**: Mark exactly ONE task as `in_progress` at any time
3. **Complete**: Mark tasks as `completed` immediately upon finishing
4. **Document**: All task activity is automatically logged to `TodoWrite_Log.md`

### Task States

- `pending`: Task not yet started
- `in_progress`: Currently working on (limit to ONE task at a time)  
- `completed`: Task finished successfully

**Critical Requirements:**
- Mark tasks `completed` ONLY when fully accomplished
- If blocked or errors occur, keep as `in_progress` and create new tasks for blockers
- Never mark partially completed tasks as done
- Use both `content` (imperative) and `activeForm` (present continuous) descriptions

### Examples

```
content: "Fix authentication bug"
activeForm: "Fixing authentication bug"

content: "Run tests and fix failures"  
activeForm: "Running tests and fixing failures"
```

### Task Logging

The `TodoWrite_Log.md` file maintains:
- Current active tasks with status
- Historical record of all completed work
- Task planning and breakdown patterns
- Project workflow tracking for future reference

This systematic approach ensures no tasks are forgotten and provides complete visibility into development progress.

## Qdrant Connectivity Validation

The system includes robust early connectivity validation to prevent processing failures:

### Features
- **Early Abort Mechanism**: Checks Qdrant connectivity before document processing begins
- **Multi-Endpoint Support**: Tests both `/health` and root `/` endpoints for compatibility
- **Comprehensive Error Handling**: Provides specific error messages for different failure types
- **Retry Logic**: Built-in retry mechanism with exponential backoff
- **Docker Integration**: Automatic Docker container detection and fallback

### Error Detection
The system detects and provides guidance for:
- **Connection Refused**: Qdrant server not running
- **DNS Resolution**: Invalid hostnames or network issues  
- **Authentication**: Invalid API keys (401/403 errors)
- **Endpoint Compatibility**: Automatically tries multiple endpoints
- **Docker Issues**: Missing or stopped Docker containers

### Troubleshooting Messages
When Qdrant is unreachable, the system provides actionable solutions:
1. Start Qdrant with: `./start_qdrant_docker.sh`
2. Check container status: `docker ps | grep qdrant`
3. Verify URL correctness (default: `http://localhost:6333`)
4. Check firewall/network settings

### Testing Connectivity
Use the included test script to validate connectivity:
```bash
# Test connectivity validation functions
uv run python test_qdrant_connectivity.py
```

This ensures reliable operation and prevents wasted processing time on unreachable databases.

## ColBERT Multi-Vector Integration (Educational Voice Tutor Optimized)

### Overview

The system now includes a production-ready ColBERT token-level embedding integration optimized for educational voice tutor applications with <50ms retrieval requirements.

### Architecture Components

#### 1. ColBERT Token Embedder (`colbert_token_embedder.py`)
- **Technology**: FastEmbed LateInteractionTextEmbedding with colbert-ir/colbertv2.0
- **Performance**: 150+ documents/second embedding generation
- **Token Matrix**: 128-dimensional tokens per document (avg 32 tokens)
- **Memory Efficient**: CPU-only processing with automatic resource optimization

#### 2. Hybrid Qdrant Store (`hybrid_qdrant_store.py`)
- **Primary Collection**: Haystack QdrantDocumentStore for dense (BGE-M3) + sparse (SPLADE) vectors
- **Secondary Collection**: Native Qdrant client for ColBERT multivector storage
- **Synchronization**: Coordinated document IDs and metadata across collections
- **Storage Overhead**: ~27x increase (1.5GB for 10k documents) for superior retrieval quality

#### 3. Multi-Vector Retrieval (`multi_vector_retrieval.py`)
- **Query Classification**: Automatic detection of factual, conceptual, procedural, comparative, and definitional queries
- **Adaptive Weights**: Query-type specific score fusion (e.g., sparse-heavy for facts, dense-heavy for concepts)
- **Parallel Search**: Concurrent execution across all vector types with <50ms target
- **Query Caching**: LRU cache with 1000-query capacity for performance optimization
- **Educational Boosts**: Domain-specific score adjustments for definitions, examples, and formulas

### Performance Features

#### Real-Time Monitoring (`performance_monitor.py`)
```bash
# Start real-time monitoring for production
python performance_monitor.py

# Monitor with custom thresholds
python performance_monitor.py --alert_threshold_ms 30 --history_window_minutes 60
```

#### Comprehensive Benchmarking (`performance_benchmark.py`)
```bash
# Full benchmark suite
uv run performance_benchmark.py --collection "your_collection" --query_count 100 --concurrent_users 20

# Quick performance validation
uv run performance_benchmark.py --collection "your_collection" --quick

# Voice tutor readiness test
uv run performance_benchmark.py --collection "your_collection" --skip_concurrent
```

#### Production Integration (`monitored_retrieval.py`)
```python
from monitored_retrieval import MonitoredRetrieval

# Initialize with monitoring
retriever = MonitoredRetrieval(
    collection_name="psychology_240", 
    enable_colbert=True,
    alert_threshold_ms=50
)

# Voice-optimized search
results = retriever.search("What is classical conditioning?")

# Check voice readiness
if retriever.is_voice_ready():
    print("✅ System ready for voice tutor deployment")
```

### Query Types and Optimization

The system automatically classifies educational queries and optimizes retrieval:

| Query Type | Example | Dense Weight | Sparse Weight | ColBERT Weight | Use Case |
|------------|---------|--------------|---------------|----------------|-----------|
| **Factual** | "What is photosynthesis?" | 0.3 | 0.4 | 0.3 | Quick facts, definitions |
| **Conceptual** | "How does memory work?" | 0.5 | 0.2 | 0.3 | Deep understanding |
| **Procedural** | "Steps to solve equations" | 0.3 | 0.3 | 0.4 | Step-by-step processes |
| **Comparative** | "Compare DNA and RNA" | 0.4 | 0.3 | 0.3 | Analysis, differences |
| **Definitional** | "Define entropy" | 0.2 | 0.5 | 0.3 | Precise terminology |

### Testing ColBERT Integration

#### Multi-Vector Retrieval Testing
```bash
# Test all vector types with adaptive weights
uv run test_retrieval.py --collection "your_collection" --multi_vector --adaptive_weights --show_timing

# Test ColBERT tokens specifically
uv run test_retrieval.py --collection "your_collection" --colbert --colbert_model "colbert-ir/colbertv2.0"

# Performance analysis with detailed metrics
uv run test_retrieval.py --collection "your_collection" --multi_vector --show_timing --limit 5
```

#### Production Readiness Validation
```bash
# Voice tutor readiness assessment
uv run monitored_retrieval.py --collection "your_collection" --queries 50 --export "./logs/readiness_report.json"

# Load testing for classroom deployment (400 students)
uv run performance_benchmark.py --collection "your_collection" --concurrent_users 50 --queries_per_user 10
```

### Storage Analysis

When ColBERT tokens are enabled, expect:

- **Primary Collection**: Dense + sparse vectors (baseline storage)
- **ColBERT Collection**: Token matrices (~27x larger)
- **Total Overhead**: ~1.5GB for 10,000 documents
- **Performance Gain**: 5-10x faster retrieval for complex educational queries
- **Voice Tutor Ready**: >95% of queries under 50ms with proper optimization

### Educational Domain Optimizations

The system includes specialized features for educational content:

#### Content Type Recognition
- **Definitions**: Automatic detection and boosting of definitional content
- **Examples**: Recognition of example patterns ("for instance", "such as")
- **Formulas**: Mathematical notation detection and prioritization
- **Procedures**: Step-by-step content identification

#### Performance Targets
- **Voice Interaction**: <50ms response time for 95% of queries
- **Classroom Scale**: Support for 400 concurrent students
- **Medical-Grade Accuracy**: Optimized for high-stakes educational scenarios
- **Real-Time Monitoring**: Continuous performance tracking with alerts

### Migration from Previous Architecture

If upgrading from the previous FAISS-only ColBERT integration:

1. **Remove Old Integration**: The broken FAISS-only code has been automatically removed
2. **Enable New Tokens**: Add `--enable_colbert_tokens` flag to embedding commands
3. **Update Retrieval**: Use new multi-vector retrieval system for optimal performance
4. **Monitor Performance**: Deploy with integrated monitoring for production readiness

This architecture provides medical-grade accuracy with sub-100ms retrieval for voice-based educational tutoring applications.

## Cross-Project Coordination

### Working with shared-rag-service

When creating collections for use by the shared-rag-service:

1. **Consistent Collection Naming**:
   ```bash
   # Use project-specific collection names that match shared-rag-service config
   uv run unified_embedder.py --qdrant_collection "enchanted_forest_materials"
   uv run unified_embedder.py --qdrant_collection "av4_chat_context" 
   uv run unified_embedder.py --qdrant_collection "slidequest_content"
   ```

2. **Enable ColBERT for Enhanced Retrieval**:
   ```bash
   # Create collections with ColBERT tokens for shared-rag-service advanced features
   uv run unified_embedder.py --qdrant_collection "project_name" --enable_colbert_tokens --colbert_model "colbert-ir/colbertv2.0"
   ```

3. **Validate Collections for Service Use**:
   ```bash
   # Test collections before deploying shared-rag-service
   uv run test_retrieval.py --collection "project_name" --multi_vector --adaptive_weights
   ```

4. **Coordinate Qdrant Settings**:
   - Ensure same QDRANT_URL in both `.env` files
   - Match embedding models between projects for compatibility
   - Coordinate collection lifecycle management

See `../shared-rag-service/CLAUDE.md` for service-side configuration and API usage.