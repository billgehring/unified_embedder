"""
Unified Embedding Pipeline for Canvas Data
==========================================

This script is the result of merging and refactoring two major predecessors:
- qdrant_embedding_mod_docling.py: Provided robust metadata ingestion (Docling/Haystack), multi-database support (Qdrant, Neo4j), and file/metadata conventions.
- local_file_embedder.py: Added robust OCR handling for PDFs, a modern CLI, improved error handling, and modular design.

Key Features:
- Embeds text, PDF, DOCX, PPTX, and other supported files from a Canvas data dump.
- Handles OCR for PDFs (force, threshold, engine selection, comparison).
- Ingests and merges metadata from _meta.json files for all resources.
- Skips or indexes transcript segment/index files (not embedded, but metadata is stored for lookup).
- Modular design for easy extension (e.g., video, image OCR, new file types).
- CLI for all major parameters and options.
- Designed for future support of both Qdrant and Neo4j.

Provenance:
  This script is a synthesis of qdrant_embedding_mod_docling.py and local_file_embedder.py.
  Please refer to those files for historical context and design evolution.

# TODO: embedded archaic .WMF files in pptx files cannot be converted.

# New refacroring 2025-12-13 by GPT5

"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import json
from datetime import datetime
import dotenv
from dotenv import load_dotenv
import re

# Default sparse embedding model
SPARSE_EMBEDDING_MODEL = "prithivida/Splade_PP_en_v1"

# Default max sequence length for text processing - increased for better context retention
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_CHUNK_OVERLAP = 128

try:
    import tqdm
except ImportError:
    print("tqdm not installed. For progress bar support, install with: uv add tqdm")
    tqdm = None

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    print("requests not installed. Required for Qdrant connectivity checks.")
    requests = None

# Load environment variables from .env file
load_dotenv()

# --- Performance Optimization and Device Configuration ---
import os

# Initialize performance optimizer early to configure environment
# Note: Import after logging is configured to avoid recursion
performance_optimizer = None

def initialize_performance_optimizer():
    """Initialize performance optimizer after logging is configured."""
    global performance_optimizer
    if performance_optimizer is not None:
        return  # Already initialized
        
    try:
        from performance_optimizer import create_performance_optimizer
        performance_optimizer = create_performance_optimizer()
        logger.info("🚀 Performance optimizer initialized successfully")
    except ImportError as e:
        # Fallback to original settings if optimizer not available
        logger.warning(f"Performance optimizer not available, using defaults: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        performance_optimizer = None

# --- Logging setup ---
from datetime import datetime
import io
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"unified_embedder_{now_str}.log")
# --- Improved Logging Setup ---
def configure_logging(debug_mode=False):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    handlers = [
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )
    # Set root logger level and handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for h in handlers:
        if h not in root_logger.handlers:
            root_logger.addHandler(h)
    # Ensure all major libraries propagate to root
    for lib_logger_name in [
        "haystack", "haystack_integrations", "qdrant_client", "qdrant_haystack", "requests", "urllib3"
    ]:
        lib_logger = logging.getLogger(lib_logger_name)
        lib_logger.setLevel(log_level)
        lib_logger.propagate = True
    # Confirm logging config
    logging.getLogger("unified_embedder").info(
        f"Logging configured: file='{log_filename}', level={'DEBUG' if debug_mode else 'INFO'}"
    )

# Parse --debug from sys.argv before anything else
DEBUG_MODE = any(arg in ("--debug", "-d") for arg in sys.argv)
configure_logging(debug_mode=DEBUG_MODE)
logger = logging.getLogger("unified_embedder")

# Log the command line at the top of the log file
logger.info("COMMAND LINE: %s", ' '.join(sys.argv))

# --- Tesseract diagnostics (helps debug OSD/OSR issues) ---
try:
    from tesseract_diagnostics import log_tesseract_diagnostics
    log_tesseract_diagnostics(logger, quick=True)
except Exception as _e_diag:
    logger.debug(f"Tesseract diagnostics not available: {_e_diag}")

# --- Redirect all print() and stderr/stdout to logger ---
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''
    def write(self, message):
        message = message.rstrip()
        if message:
            for line in message.splitlines():
                # Filter out tqdm progress bar output and other noise
                if not self._is_progress_bar_output(line):
                    # Use appropriate log level based on content for stderr
                    if self.level == logger.debug and line.strip():
                        # For stderr, categorize based on content
                        if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                            logger.error(line)
                        elif any(keyword in line.lower() for keyword in ['warning', 'warn', 'deprecated']):
                            logger.warning(line)
                        else:
                            logger.debug(line)
                    else:
                        self.level(line)
    def flush(self):
        pass
    
    def _is_progress_bar_output(self, line):
        """Check if the line is tqdm progress bar output or other non-error output that should not be logged."""
        # Common tqdm progress bar patterns and other progress output
        ignore_patterns = [
            'it/s',  # iterations per second
            '%|',    # progress bar percentage
            'Batches:',  # batch progress
            'Processing files:',  # file processing progress
            '##',    # progress bar fill characters
            '[A',    # ANSI escape sequence for cursor movement (common in progress bars)
            '\r',    # carriage return (progress bar updates)
            'Loading checkpoint shards:',  # model loading progress
            'Downloading',  # download progress
            'Fetching',  # fetching progress
            'Map:',  # dataset mapping progress
            'Filter:',  # dataset filtering progress
        ]
        # Also filter out empty or whitespace-only lines
        if not line.strip():
            return True
        return any(pattern in line for pattern in ignore_patterns)

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.debug)  # Changed from error to debug to reduce noise

# Patch built-in print to always log
import builtins
_builtin_print = print
def print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    logger.info(msg)
    _builtin_print(*args, **kwargs)
builtins.print = print

# --- Utility to clean env vars (strip quotes/whitespace) ---
def clean_env_var(val):
    if val is None:
        return None
    val = val.strip()
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        val = val[1:-1]
    return val.strip()


# --- Supported File Extensions ---
EMBEDDABLE_EXTENSIONS = {".txt", ".pdf", ".docx", ".pptx", ".xml", ".vtt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".JPG"}
META_SUFFIX = "_meta.json"

# --- OCR and Embedding Imports ---
try:
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.utils import Secret
    from haystack import component
    from sentence_transformers import SentenceTransformer
    # --- Docling Import ---
    from docling_haystack.converter import DoclingConverter
except ImportError as e:
    logger.error("Required embedding libraries not installed: %s", e)
    sys.exit(1)

# --- Qdrant and FastEmbed Integration ---
try:
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    
    # Import FastEmbed sparse embedders
    from haystack_integrations.components.embedders.fastembed import (
        FastembedDocumentEmbedder,
        FastembedTextEmbedder,
        FastembedSparseDocumentEmbedder,
        FastembedSparseTextEmbedder
    )
except ImportError:
    logger.error("haystack-integrations not installed. Please install haystack-integrations and fastembed for Qdrant support.")
    sys.exit(1)

QDRANT_DEFAULT_COLLECTION = "canvas_unified_embeddings"
QDRANT_DEFAULT_URL = "http://localhost:6333"  # Change as needed


def check_qdrant_connectivity(qdrant_url: str, qdrant_api_key: Optional[str] = None, timeout: int = 10) -> bool:
    """
    Check if Qdrant server is reachable and responsive.
    
    Args:
        qdrant_url: URL of the Qdrant server
        qdrant_api_key: Optional API key for authentication
        timeout: Timeout in seconds for the connectivity check
    
    Returns:
        bool: True if Qdrant is accessible, False otherwise
    """
    if not requests:
        logger.error("requests library not available - cannot perform connectivity check")
        return False
        
    try:
        # Create a session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Updated from method_whitelist
            backoff_factor=0.3
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if qdrant_api_key:
            headers["Api-Key"] = qdrant_api_key
            
        # Test basic connectivity with root endpoint (Qdrant doesn't always have /health)
        # Try health endpoint first, then fallback to root
        endpoints_to_try = [
            f"{qdrant_url.rstrip('/')}/health",
            f"{qdrant_url.rstrip('/')}/"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                logger.info(f"Checking Qdrant connectivity at {endpoint}...")
                response = session.get(endpoint, headers=headers, timeout=timeout)
                
                if response.status_code == 200:
                    try:
                        health_data = response.json()
                        if "version" in health_data or "title" in health_data:
                            logger.info(f"✅ Qdrant server is healthy: {health_data}")
                            return True
                    except json.JSONDecodeError:
                        # Some Qdrant versions return plain text
                        if "ok" in response.text.lower() or "qdrant" in response.text.lower():
                            logger.info(f"✅ Qdrant server is healthy: {response.text}")
                            return True
                        
                elif response.status_code == 401:
                    logger.error(f"❌ Qdrant authentication failed - check your API key")
                    return False
                    
                elif response.status_code == 403:
                    logger.error(f"❌ Qdrant access forbidden - insufficient permissions")
                    return False
                    
                else:
                    logger.warning(f"⚠️  Endpoint {endpoint} returned status {response.status_code}: {response.text}")
                    continue  # Try next endpoint
                    
            except requests.exceptions.RequestException:
                continue  # Try next endpoint
                
        # If we get here, all endpoints failed
        logger.error(f"❌ All Qdrant endpoints failed - server may be unreachable")
        return False
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Cannot connect to Qdrant at {qdrant_url}: {e}")
        logger.error("   Make sure Qdrant is running and accessible at the specified URL")
        return False
        
    except requests.exceptions.Timeout as e:
        logger.error(f"❌ Qdrant connection timeout after {timeout}s: {e}")
        logger.error("   The server might be running but overloaded or slow to respond")
        return False
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Qdrant connectivity check failed: {e}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during Qdrant connectivity check: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False


def check_qdrant_collections_endpoint(qdrant_url: str, qdrant_api_key: Optional[str] = None, timeout: int = 10) -> bool:
    """
    Check if Qdrant collections endpoint is accessible (more thorough test).
    
    Args:
        qdrant_url: URL of the Qdrant server
        qdrant_api_key: Optional API key for authentication  
        timeout: Timeout in seconds
        
    Returns:
        bool: True if collections endpoint is accessible, False otherwise
    """
    if not requests:
        return False
        
    try:
        headers = {"Content-Type": "application/json"}
        if qdrant_api_key:
            headers["Api-Key"] = qdrant_api_key
            
        collections_url = f"{qdrant_url.rstrip('/')}/collections"
        
        response = requests.get(collections_url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            collections_data = response.json()
            collection_count = len(collections_data.get('result', {}).get('collections', []))
            logger.info(f"✅ Qdrant collections endpoint accessible - found {collection_count} collections")
            return True
        else:
            logger.warning(f"⚠️  Collections endpoint returned status {response.status_code}")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️  Collections endpoint check failed: {e}")
        return False
QDRANT_DEFAULT_API_KEY = None  # Set to your API key if using Qdrant Cloud

# --- OCR Engines ---
def perform_ocr(pdf_path: Path, engine: str = "Tesseract") -> str:
    """
    Perform OCR on the given PDF file using the specified engine.
    Returns extracted text as a string.
    """
    # Placeholder: implement engine selection logic as needed.
    logger.info(f"Performing OCR on {pdf_path} using {engine}")
    
    # TODO: Replace with actual OCR logic from enhanced_docling_converter.py
    # For now, we'll just return a placeholder
    return f"[OCR text for {pdf_path.name} processed with {engine}]"

# --- Metadata Handling ---
def load_metadata(meta_path: Path) -> Dict[str, Any]:
    """
    Load metadata from a _meta.json file. Returns an empty dict if not found or invalid.
    """
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load metadata from {meta_path}: {e}")
        return {}

def find_metadata_file(doc_path: Path) -> Optional[Path]:
    """
    Given a document path, find the corresponding _meta.json file if it exists.
    """
    meta_candidate = doc_path.with_name(doc_path.stem + META_SUFFIX)
    if meta_candidate.exists():
        return meta_candidate
    return None

# --- File Type Detection ---
def is_embeddable_file(path: Path) -> bool:
    return path.suffix in EMBEDDABLE_EXTENSIONS

def is_image_file(path: Path) -> bool:
    return path.suffix in IMAGE_EXTENSIONS

# --- Transcript Index Detection ---
def is_transcript_index(meta: Dict[str, Any], doc_path: Path) -> bool:
    """
    Heuristic: If meta contains certain keys (e.g., segment list, timestamps) but little text, treat as index.
    Extend as needed for your data.
    """
    if meta.get("type") == "transcript_index":
        return True
    # Heuristic: XML files with segment/timestamp structure
    if doc_path.suffix == ".xml" and "segments" in meta and "video" in meta:
        return True
    return False

# --- Embedding Logic (deprecated in workers) ---
def embed_text(text: str, embedder: Any) -> Optional[List[float]]:
    """
    Deprecated: embeddings are generated centrally with FastEmbed in the main process.
    Kept for backward compatibility where callers expect a function.
    """
    logger.debug("embed_text called (deprecated); returning None to indicate central embedding.")
    return None

# --- Enhanced Docling Converter and Deduplication Import ---
# Moved here to be accessible within process_file
try:
    from enhanced_docling_converter import EnhancedDoclingConverter
    from deduplication import integrate_deduplication_into_pipeline
except ImportError:
    logger.error("enhanced_docling_converter.py or deduplication.py not found. Please ensure they're in the same directory or Python path.")
    sys.exit(1)

# --- Embedding Dimension Checks and Parallel Processing Imports ---
import multiprocessing
import psutil

# --- Embedding/model compatibility utilities ---
def get_embedding_dimension(model_name):
    """Dynamically determine the embedding dimension for a given model."""
    global performance_optimizer
    
    try:
        # Use performance optimizer to get optimal device
        device = "cpu"  # Default fallback
        if performance_optimizer:
            device = performance_optimizer.get_embedding_device()
            logger.info(f"🎯 Using device '{device}' for embedding dimension test")
        
        model = SentenceTransformer(model_name, device=device)
        sample_text = "This is a sample text to determine embedding dimension."
        embedding = model.encode(sample_text)
        return len(embedding)
    except Exception as e:
        logger.warning(f"Could not determine embedding dimension for %s: %s", model_name, e)
        return 384  # Common default dimension

def determine_optimal_workers(file_count: int = 1000,
                             file_size_estimate: int = 100000,
                             memory_buffer: float = 0.7,
                             max_workers: int = None) -> int:
    """Determine the optimal number of worker processes based on system resources and workload."""
    global performance_optimizer
    
    # Use performance optimizer if available
    if performance_optimizer:
        optimal_workers = performance_optimizer.get_optimal_worker_count(file_count, file_size_estimate)
        logger.info(f"🚀 Performance optimizer determined worker count: %s", optimal_workers)
        return optimal_workers
    
    # Fallback to original logic if optimizer not available
    cpu_count = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available
    if max_workers is None:
        max_workers = cpu_count
    estimated_memory_per_file = file_size_estimate * 3
    total_memory_needed = file_count * estimated_memory_per_file
    memory_based_workers = int((available_memory * memory_buffer) / total_memory_needed * cpu_count)
    cpu_based_workers = max(1, cpu_count - 1)
    optimal_workers = min(memory_based_workers, cpu_based_workers, max_workers)
    optimal_workers = max(1, optimal_workers)
    logger.info(f"System has %s CPU cores and %.2f GB available memory", cpu_count, available_memory / (1024**3))
    logger.info(f"Determined optimal worker count (fallback): %s", optimal_workers)
    return optimal_workers

# --- Batch Document Writer ---
from haystack import component
try:
    from haystack.document_stores.types import DuplicatePolicy
except Exception:
    DuplicatePolicy = None  # Fallback if not available

@component
class BatchDocumentWriter:
    """Component to write documents to a document store in batches for improved performance."""
    def __init__(self, document_store, batch_size: int = 100, recreate_index: bool = False):
        self.document_store = document_store
        self.batch_size = batch_size
        self.recreate_index = recreate_index
        self.stats = {"total_documents": 0, "batches_written": 0, "write_errors": 0}

    def ensure_metadata_preserved(self, documents):
        if not documents:
            return documents
        sample_doc = documents[0]
        metadata_keys = list(getattr(sample_doc, 'meta', {}).keys())
        logger.info(f"Sample document metadata before writing: %s", metadata_keys)
        for i, doc in enumerate(documents[:5]):
            if hasattr(doc, 'meta') and "ocr_engine_used" in doc.meta:
                logger.info(f"Document %s has OCR engine: %s", i, doc.meta['ocr_engine_used'])
            elif hasattr(doc, 'meta'):
                logger.warning(f"Document %s is missing ocr_engine_used field", i)
            if hasattr(doc, 'meta') and "ocr_reprocessed" in doc.meta:
                logger.info(f"Document %s OCR reprocessed: %s", i, doc.meta['ocr_reprocessed'])
        for doc in documents:
            if hasattr(doc, 'meta') and doc.meta.get("ocr_reprocessed", False) and doc.meta.get("ocr_engine_used") is None:
                logger.warning(f"Found document with ocr_reprocessed=True but ocr_engine_used=None. Setting default engine.")
                doc.meta["ocr_engine_used"] = "UnknownEngine"
        return documents

    def write_batch(self, batch):
        if not batch:
            logger.warning("Empty batch, nothing to write")
            return
        self.ensure_collection_exists()
        batch = self.ensure_metadata_preserved(batch)
        try:
            if batch and len(batch) > 0:
                logger.info(f"Sample metadata before writing (first document):")
                doc0 = batch[0]
                for key, value in getattr(doc0, 'meta', {}).items():
                    logger.info(f"  %s: %s", key, value)
            
            # Write documents to the document store with overwrite policy to avoid duplicate lookups
            if DuplicatePolicy is not None:
                try:
                    self.document_store.write_documents(batch, policy=DuplicatePolicy.OVERWRITE)
                except TypeError:
                    # Older interface without policy arg
                    self.document_store.write_documents(batch)
            else:
                self.document_store.write_documents(batch)
            self.stats["batches_written"] += 1
            self.stats["total_documents"] += len(batch)
            
            # Added: Verify metadata after writing for a sample document
            if batch:
                doc_id = getattr(batch[0], 'id', None)
                if doc_id:
                    logger.info(f"Verifying metadata for document %s", doc_id)
                    # For Qdrant, we can try to retrieve the document to check metadata
                    try:
                        if hasattr(self.document_store, 'get_documents_by_id'):
                            retrieved_docs = self.document_store.get_documents_by_id([doc_id])
                            if retrieved_docs:
                                retrieved_doc = retrieved_docs[0]
                                logger.info(f"Retrieved document metadata after writing:")
                                for key, value in getattr(retrieved_doc, 'meta', {}).items():
                                    logger.info(f"  %s: %s", key, value)
                    except Exception as retrieval_err:
                        logger.warning(f"Error retrieving document for metadata verification: %s", retrieval_err)
                
        except Exception as e:
            logger.error(f"Error writing batch to document store: %s", e)
            logger.error(traceback.format_exc())
            self.stats["write_errors"] += 1

    def ensure_collection_exists(self):
        if 'QdrantDocumentStore' in globals() and isinstance(self.document_store, QdrantDocumentStore) and not self.recreate_index:
            collection_name = getattr(self.document_store, 'index', None)
            try:
                client = getattr(self.document_store, '_client', None)
                if client and hasattr(client, 'collection_exists'):
                    collection_exists = client.collection_exists(collection_name)
                    if not collection_exists:
                        print(f"\nERROR: Qdrant collection '{collection_name}' does not exist and --recreate_index is not set.")
                        print("Please either:")
                        print("  1. Use the --recreate_index flag to create a new collection, or")
                        print("  2. Specify an existing collection name with --index_name")
                        sys.exit(1)
            except Exception as e:
                logger.error(f"Error checking if collection exists: %s", e)

    def run(self, documents: List[Any]):
        if not documents:
            logger.warning("No documents to write")
            return {"documents": []}
        logger.info(f"Writing %s documents to document store in batches of %s", len(documents), self.batch_size)
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            batch_size = len(batch)
            try:
                self.write_batch(batch)
                logger.debug(f"Successfully wrote batch %s with %s documents", i//self.batch_size + 1, batch_size)
            except Exception as e:
                logger.error(f"Error processing batch: %s", e)
                logger.error(traceback.format_exc())
        logger.info(f"Finished writing documents. Stats: %s", self.stats)
        return {"documents": documents}

# nb QdrantDocumentStore.write_documents expects a list of documents current Haystack (2.13.1) uses content= to write the text content

# --- Qdrant Writer --- 
def write_to_qdrant(results: list, collection: str, qdrant_url: str, qdrant_api_key: Optional[str]):
    """
    Write embeddings and metadata to Qdrant collection.
    
    results contains the haystack document list
    """
    logger.info(f"Connecting to Qdrant at %s, collection: %s", qdrant_url, collection)
    document_store = QdrantDocumentStore(
        url=qdrant_url,
        api_key=Secret.from_token(qdrant_api_key) if qdrant_api_key else None,
        index=collection,
        recreate_index=False  # Do not recreate by default
    )
    
    logger.debug(results[0:2])
    
    haystack_docs = []
    for item in results:
        # Get the text content if available
        text_content = item.get("text", "")
        if not text_content and "meta" in item and "text" in item["meta"]:
            text_content = item["meta"]["text"]
            
        if item.get("embedding") is not None and not item.get("skipped", False):
            haystack_docs.append(Document(
                content=text_content,  # Store the actual text content
                embedding=item["embedding"],
                meta=item["meta"] | {"path": item["path"]}
            ))
        elif item.get("skipped", False):
            # Store metadata-only document for skipped/index files
            haystack_docs.append(Document(
                content=text_content,  # Store content even for skipped files
                embedding=None,
                meta=item["meta"] | {"path": item["path"], "skipped": True}
            ))
    logger.info(f"Writing %s documents to Qdrant...", len(haystack_docs))
    document_store.write_documents(haystack_docs)
    logger.info("Finished writing to Qdrant.")


def enhanced_process_file(doc_path: Path, embedder: Any, ocr_engine: str, force_ocr: bool) -> Optional[Dict[str, Any]]:
    """
    Enhanced file processor with VTT support.
    This replaces or extends the existing process_file function.
    """
    
    # Check if it's a VTT file
    if doc_path.suffix.lower() == '.vtt':
        logger.info(f"Processing VTT file: {doc_path}")
        
        # Process VTT with timestamp preservation
        texts, metadatas = VTTProcessor.process_vtt_file(
            doc_path,
            chunk_size=1000,  # Adjust based on your needs
            chunk_overlap=100
        )
        
        if not texts:
            logger.warning(f"No content extracted from VTT file: {doc_path}")
            return None
        
        chunks = []
        for text, metadata in zip(texts, metadatas):
            # Load any existing _meta.json
            meta_path = find_metadata_file(doc_path)
            if meta_path:
                base_meta = load_metadata(meta_path)
                metadata.update(base_meta)
            # Append chunk (no embedding here)
            chunks.append({
                "content": text,
                "meta": metadata
            })

        return {"path": str(doc_path), "chunks": chunks, "skipped": False}

# --- Main Processing ---
def process_file(doc_path: Path, embedder: Any, ocr_engine: str, force_ocr: bool) -> Optional[Dict[str, Any]]:
    """
    Process a single file using EnhancedDoclingConverter: extract text, merge metadata, embed if appropriate.
    Returns a dict with embedding and metadata, or None if skipped/failed.
    """
    meta = {}
    meta_path = find_metadata_file(doc_path)
    if meta_path:
        meta = load_metadata(meta_path)

    # Transcript index detection - keep this check
    if is_transcript_index(meta, doc_path):
        logger.info(f"Skipping indexing for transcript index: %s", doc_path)
        return {"path": str(doc_path), "meta": meta, "skipped": True}

    # Skip _meta.json files silently (but record reason)
    if doc_path.name.endswith(META_SUFFIX):
        return {"path": str(doc_path), "meta": meta, "skipped": True, "reason": "meta_json"}

    # Process embeddable files using EnhancedDoclingConverter
    if is_embeddable_file(doc_path):
        try:
            logger.info(f"Processing %s using EnhancedDoclingConverter (OCR Engine: %s, Force OCR: %s)", doc_path, ocr_engine, force_ocr)
            
            # Initialize the enhanced converter within the function
            # Pass relevant arguments like ocr_engine and force_ocr
            # TODO: Consider if converter initialization can be moved outside for efficiency if state is not needed per file
            converter = EnhancedDoclingConverter(
                preferred_ocr_engine=ocr_engine, 
                force_ocr=force_ocr,
                # Add other relevant EnhancedDoclingConverter init params if needed
            )

            # Run the enhanced converter
            # It handles PDFs, DOCX, PPTX, TXT, etc., including OCR logic internally
            result_dict = converter.run(paths=[str(doc_path)])
            
            # Check if conversion was successful and returned documents
            haystack_docs: List[Document] = result_dict.get("documents")
            if not haystack_docs:
                logger.warning(f"EnhancedDoclingConverter returned no documents for %s. Skipping.", doc_path)
                return None

            # Build chunk records directly for central embedding
            chunk_records = []
            for d in haystack_docs:
                if d.content is None or not str(d.content).strip():
                    continue
                chunk_meta = meta.copy()
                if getattr(d, 'meta', None):
                    chunk_meta.update(d.meta)
                chunk_meta["path"] = str(doc_path)
                chunk_meta["filename"] = doc_path.name
                # Remove any legacy 'text' keys if present
                chunk_meta.pop("text", None)
                chunk_records.append({
                    "content": d.content,
                    "meta": chunk_meta
                })

            if not chunk_records:
                logger.warning(f"No content chunks extracted for %s. Skipping.", doc_path)
                return None

            logger.info(f"Extracted {len(chunk_records)} chunks for {doc_path}")
            return {"path": str(doc_path), "chunks": chunk_records, "skipped": False}

        except Exception as e:
            # Catching general Exception here to prevent one file from crashing the whole batch
            # Specific errors during conversion/embedding should be handled internally if possible
            logger.error(f"Error processing %s with EnhancedDoclingConverter: %s", doc_path, e)
            logger.error(traceback.format_exc())
            return None # Indicate failure

    # Image files: store metadata only
    if is_image_file(doc_path):
        logger.info(f"Storing metadata for image file: %s", doc_path)
        return {"path": str(doc_path), "meta": meta, "skipped": True, "reason": "image"}

    # Log unsupported file types and return skipped record
    if not doc_path.name.endswith(META_SUFFIX):
        logger.warning(f"Skipping unsupported file type: %s", doc_path)
    return {"path": str(doc_path), "meta": meta, "skipped": True, "reason": "unsupported"}


# --- Directory Traversal ---
def process_directory(root_dir: Path, embedder: Any, ocr_engine: str, force_ocr: bool) -> List[Dict[str, Any]]:
    """
    Recursively process all files in the directory, embedding or indexing as appropriate.
    """
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            result = process_file(fpath, embedder, ocr_engine, force_ocr)
            if result is not None:
                results.append(result)
    return results

# --- Top-level function for multiprocessing: loads CPU-only embedder in each subprocess. ---
def process_file_wrapper(args_tuple):
    """Top-level worker: extract/ocr + chunk only. No embedding here."""
    doc_path, embedder_args, ocr_engine, force_ocr = args_tuple
    try:
        # Set logging for subprocess
        subprocess_logger = logging.getLogger("unified_embedder.subprocess")
        subprocess_logger.setLevel(logging.INFO)
        
        # Log OCR settings in the subprocess
        subprocess_logger.info(f"Subprocess OCR settings for %s: force_ocr=%s, engine=%s", doc_path, force_ocr, ocr_engine)
        
        # Do not load embedding models in worker processes
        embedder = None
        
        # Check if the file is a PDF (for additional OCR logging)
        is_pdf = str(doc_path).lower().endswith('.pdf')
        if is_pdf:
            subprocess_logger.info(f"PDF file detected: %s, will apply OCR settings: force_ocr=%s", doc_path, force_ocr)
        
        # Process the file
        result = process_file(doc_path, embedder, ocr_engine, force_ocr)
        
        # Log chunk stats if available
        if result is not None and not result.get("skipped", False):
            chunks = result.get("chunks", [])
            subprocess_logger.info(f"Extracted {len(chunks)} chunks for %s", doc_path)
            if chunks:
                sample = chunks[0].get("content", "")
                if isinstance(sample, str) and sample:
                    preview = sample[:100] + ("..." if len(sample) > 100 else "")
                    subprocess_logger.info(f"  - First chunk preview: %s", preview)
        
        return result
    except Exception as e:
        logger.error(f"Error in process_file_wrapper for %s: %s", doc_path, e)
        logger.error(traceback.format_exc())
        return None

# --- Text Chunking Function ---
def chunk_text(text: str, max_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks that fit within the model's maximum sequence length.
    Uses simple paragraph splitting with overlap.
    
    Args:
        text: The text to split into chunks
        max_length: Maximum number of characters per chunk
        overlap: Overlap between consecutive chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    # If text fits in a single chunk, return it
    if len(text) <= max_length:
        return [text]
    
    # First split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the max length
        if len(current_chunk) + len(paragraph) > max_length:
            # If current_chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
            
            # Start a new chunk with this paragraph
            if len(paragraph) <= max_length:
                current_chunk = paragraph
            else:
                # If the paragraph itself is too long, split it by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_length:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If a single sentence is too long, split it by words
                        if len(sentence) > max_length:
                            words = sentence.split()
                            current_chunk = ""
                            
                            for word in words:
                                if len(current_chunk) + len(word) + 1 > max_length:
                                    chunks.append(current_chunk)
                                    current_chunk = word
                                else:
                                    if current_chunk:
                                        current_chunk += " " + word
                                    else:
                                        current_chunk = word
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Handle overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                overlapped_chunks.append(prev_end + chunks[i])
        chunks = overlapped_chunks
        
    # Final check to ensure all chunks are within max_length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # If still too long, truncate
            final_chunks.append(chunk[:max_length])
            
    return final_chunks

# --- CLI ---
def main():
    # Initialize performance optimizer after logging is set up
    initialize_performance_optimizer()
    
    parser = argparse.ArgumentParser(description="Unified Canvas Data Embedder (Qdrant/Neo4j ready)")
    parser.add_argument("--docs_dir", type=str, required=True, help="Directory containing Canvas data files")
    parser.add_argument("--embedding_model", type=str, default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"), help="SentenceTransformer model name for dense embeddings (default: bge-large-en-v1.5 for good performance)")
    parser.add_argument("--sparse_embedding_model", type=str, default=os.getenv("SPARSE_EMBEDDING_MODEL", SPARSE_EMBEDDING_MODEL), help="Model name for sparse embeddings")
    parser.add_argument("--qdrant_collection", type=str, help="Name of the Qdrant collection to use")
    parser.add_argument("--qdrant_url", type=str, help="URL of the Qdrant server")
    parser.add_argument("--qdrant_api_key", type=str, help="API key for Qdrant")
    parser.add_argument("--recreate_index", action="store_true", help="Recreate the Qdrant collection if it exists")
    parser.add_argument("--ocr_engine", type=str, default=os.getenv("OCR_ENGINE", "Tesseract"), help="OCR engine for PDFs")
    parser.add_argument("--force-ocr", "--force_ocr", action="store_true", help="Force OCR for all PDFs (bypass quality check)")
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT", "embeddings.json"), help="Output file for embeddings and metadata")
    parser.add_argument("--qdrant", action="store_true", help="Write results to Qdrant vector DB")
    parser.add_argument("--database", type=str, default=os.getenv("DATABASE", "qdrant"), choices=["qdrant", "neo4j"], help="Database type to use (qdrant or neo4j)")
    parser.add_argument("--use_sparse", action="store_true", default=True, help="Use sparse embeddings for hybrid search (default: True)")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process (default: process all files)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (defaults to auto-detected optimal value)")
    parser.add_argument("--max_seq_length", type=int, default=int(os.getenv("MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH)), 
                        help=f"Maximum sequence length for text chunks (default: {DEFAULT_MAX_SEQ_LENGTH})")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", DEFAULT_MAX_SEQ_LENGTH)), 
                        help=f"Size of document chunks in characters (default: {DEFAULT_MAX_SEQ_LENGTH})")
    parser.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)), 
                        help=f"Overlap between chunks in characters (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--enable_dedup", action="store_true", help="Enable document deduplication")
    parser.add_argument("--dedup_threshold", type=float, default=0.95, help="Similarity threshold for deduplication (default: 0.95)")
    parser.add_argument("--enable_colbert_tokens", action="store_true", help="Generate and store ColBERT token embeddings in Qdrant for ultra-fast retrieval")
    parser.add_argument("--colbert_model", default="sentence-transformers/all-MiniLM-L6-v2", help="Model for ColBERT token generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.error(f"Document directory %s does not exist or is not a directory.", docs_dir)
        sys.exit(1)

    # Check database type and import accordingly
    if args.database == "qdrant":
        try:
            from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
        except ImportError:
            logger.error("haystack-integrations not installed. Please install haystack-integrations for Qdrant support.")
            sys.exit(1)
    elif args.database == "neo4j":
        try:
            from haystack_integrations.document_stores.neo4j import Neo4jDocumentStore
        except ImportError:
            logger.error("haystack-integrations not installed. Please install haystack-integrations for Neo4j support.")
            sys.exit(1)

    # Early Qdrant connectivity check if using Qdrant
    if args.qdrant:
        logger.info("🔍 Performing early Qdrant connectivity check...")
        
        # Resolve Qdrant URL using the same logic as the main processing
        def clean_env_var(var):
            if not var:
                return None
            return var.strip().strip('"').strip("'")

        def mask_api_key(key):
            if not key:
                return None
            if len(key) <= 6:
                return "*" * len(key)
            return key[:2] + "*" * (len(key)-6) + key[-4:]

        # Step 1: CLI arguments first
        qdrant_url = clean_env_var(args.qdrant_url) if args.qdrant_url else None
        qdrant_api_key = clean_env_var(args.qdrant_api_key) if args.qdrant_api_key else None
        qdrant_collection = args.qdrant_collection or os.getenv("QDRANT_COLLECTION") or QDRANT_DEFAULT_COLLECTION

        # Step 2: .env file fallback
        if not qdrant_url:
            qdrant_url = clean_env_var(os.getenv("QDRANT_URL"))
        if not qdrant_api_key:
            qdrant_api_key = clean_env_var(os.getenv("QDRANT_API_KEY"))

        # Step 3: Docker fallback if QDRANT_TYPE=docker
        qdrant_type = os.getenv("QDRANT_TYPE", "docker").lower()
        if not qdrant_url and qdrant_type == "docker":
            # Check if docker is running and qdrant container is up
            import subprocess
            try:
                docker_ps = subprocess.run([
                    "docker", "ps", "--filter", "name=qdrant", "--filter", "status=running", "--format", "{{.Names}}"
                ], capture_output=True, text=True)
                running_containers = docker_ps.stdout.strip().splitlines()
                if any("qdrant" in name for name in running_containers):
                    qdrant_url = QDRANT_DEFAULT_URL
                    logger.info("Using Docker Qdrant instance at %s", qdrant_url)
                else:
                    logger.error("❌ QDRANT_TYPE is set to 'docker' but no running Qdrant Docker container was found.")
                    logger.error("   Please start the Qdrant Docker container with: ./start_qdrant_docker.sh")
                    logger.error("   Or specify QDRANT_URL in your .env file")
                    sys.exit(1)
            except Exception as e:
                logger.error("❌ Error checking Docker containers: %s", e)
                sys.exit(1)

        # Final URL validation
        if not qdrant_url:
            logger.error("❌ No Qdrant URL specified via CLI, .env, or Docker fallback.")
            logger.error("   Please specify --qdrant_url on command line or QDRANT_URL in .env file")
            logger.error("   Or set QDRANT_TYPE=docker and start Qdrant container")
            sys.exit(1)

        # Perform connectivity checks
        logger.info(f"Checking connectivity to: {qdrant_url}")
        logger.info(f"Collection: {qdrant_collection}")
        logger.info(f"API Key: {mask_api_key(qdrant_api_key)}")

        # Basic health check
        if not check_qdrant_connectivity(qdrant_url, qdrant_api_key):
            logger.error("❌ Qdrant connectivity check failed!")
            logger.error("   Common solutions:")
            logger.error("   1. Start Qdrant with: ./start_qdrant_docker.sh")
            logger.error("   2. Check if Qdrant is running: docker ps | grep qdrant")
            logger.error("   3. Verify URL is correct (default: http://localhost:6333)")
            logger.error("   4. Check firewall/network settings")
            sys.exit(1)

        # More thorough collections endpoint check
        if not check_qdrant_collections_endpoint(qdrant_url, qdrant_api_key):
            logger.warning("⚠️  Collections endpoint check failed, but basic connectivity works")
            logger.warning("   This may indicate authentication issues or limited permissions")
            logger.warning("   Proceeding with caution...")

        logger.info("✅ Qdrant connectivity check passed! Proceeding with document processing...")

    logger.info(f"Loading embedding model: %s", args.embedding_model)
    
    # Use performance optimizer for device selection
    embedding_device = "cpu"  # Default fallback
    if performance_optimizer:
        embedding_device = performance_optimizer.get_embedding_device()
        logger.info(f"🎯 Performance optimizer selected device: {embedding_device}")
        
        # Log optimization summary
        summary = performance_optimizer.get_system_summary()
        logger.info(f"📊 System optimization summary: {summary['recommendations']}")
    
    embedder = SentenceTransformer(args.embedding_model, device=embedding_device)
    emb_dim = get_embedding_dimension(args.embedding_model)
    logger.info(f"Embedding dimension for model %s: %s", args.embedding_model, emb_dim)

    logger.info(f"Processing directory: %s", docs_dir)

    # Parallel processing for file embedding
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Get all files to process
    logger.info(f"Scanning directory: %s", docs_dir)
    file_paths = []
    skipped_count = 0
    # Optional skip patterns via env (comma-separated globs or substrings)
    import fnmatch
    skip_patterns_env = os.getenv("SKIP_PATTERNS", "")
    skip_patterns = [p.strip() for p in skip_patterns_env.split(",") if p.strip()]
    if skip_patterns:
        logger.info(f"Applying SKIP_PATTERNS: {skip_patterns}")

    for dirpath, _, filenames in os.walk(docs_dir):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            # Skip obvious non-embeddable files like hidden files and _meta.json files
            if fname.startswith('.') or fname.endswith(META_SUFFIX):
                skipped_count += 1
                continue
            if skip_patterns:
                rel_path = str(fpath)
                # Match against any pattern (supports glob and substring)
                if any(fnmatch.fnmatch(rel_path, pat) or pat in rel_path for pat in skip_patterns):
                    skipped_count += 1
                    logger.debug(f"Skipping due to SKIP_PATTERNS: {rel_path}")
                    continue
            file_paths.append(fpath)
    
    # Apply max_files limit if specified
    if args.max_files is not None and len(file_paths) > args.max_files:
        logger.info(f"Limiting to %s files out of %s found", args.max_files, len(file_paths))
        file_paths = file_paths[:args.max_files]
    
    file_count = len(file_paths)
    logger.info(f"Found %s files to process (skipped %s files).", file_count, skipped_count)
    
    # Determine optimal workers based on system resources
    num_workers = args.num_workers or determine_optimal_workers(file_count)
    logger.info(f"Using %s parallel workers.", num_workers)

    # Process files in parallel
    results = []
    total_processed = 0
    total_errors = 0
    total_skipped = 0
    
    # Get the force_ocr flag value (could be from --force_ocr or --force-ocr)
    force_ocr = getattr(args, 'force_ocr', False) or getattr(args, 'force-ocr', False)
    # Log OCR configuration
    logger.info(f"OCR Configuration: force_ocr=%s, engine=%s", force_ocr, args.ocr_engine)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        # Log command-line args and flags for clarity
        logger.info(f"Command-line arguments: %s", args)
        logger.info(f"Force OCR setting: %s", force_ocr)
        
        futures = {
            executor.submit(
                process_file_wrapper, 
                (fpath, args.embedding_model, args.ocr_engine, force_ocr)
            ): fpath for fpath in file_paths
        }
        
        # Process completed futures with a progress bar if tqdm is available
        if tqdm:
            with tqdm.tqdm(total=len(futures), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    fpath = futures[future]
                    pbar.update(1)
                    
                    try:
                        result = future.result()
                        if result is not None:
                            if result.get("skipped", False):
                                total_skipped += 1
                                logger.debug(f"Skipped file: %s (%s)", fpath, result.get("reason", "unknown"))
                            else:
                                total_processed += 1
                                # Log chunk count if present
                                if 'chunks' in result:
                                    logger.debug(f"Extracted {len(result['chunks'])} chunks from %s", fpath)
                            results.append(result)
                    except Exception as exc:
                        total_errors += 1
                        logger.error(f"File %s generated an exception: %s", fpath, exc)
                        logger.error(traceback.format_exc())
        else:
            # Process without progress bar
            for i, future in enumerate(as_completed(futures)):
                fpath = futures[future]
                if i % 10 == 0:  # Log progress every 10 files
                    logger.info(f"Progress: %s/%s files processed", i, len(futures))
                
                try:
                    result = future.result()
                    if result is not None:
                        if result.get("skipped", False):
                            total_skipped += 1
                            logger.debug(f"Skipped file: %s (%s)", fpath, result.get("reason", "unknown"))
                        else:
                            total_processed += 1
                            if 'chunks' in result:
                                logger.debug(f"Extracted {len(result['chunks'])} chunks from %s", fpath)
                        results.append(result)
                except Exception as exc:
                    total_errors += 1
                    logger.error(f"File %s generated an exception: %s", fpath, exc)
                    logger.error(traceback.format_exc())
        
    logger.info(f"Completed processing: %s files processed, %s files skipped, %s errors.", total_processed, total_skipped, total_errors)

    # Persist a skip report for auditability
    try:
        skip_report = []
        reason_counts = {}
        for item in results:
            if item and item.get("skipped", False):
                entry = {
                    "path": item.get("path"),
                    "reason": item.get("reason", "unknown")
                }
                # Include a small subset of metadata if present
                meta = item.get("meta")
                if isinstance(meta, dict):
                    for k in ("filename", "file_path", "ocr_engine_used", "ocr_reprocessed"):
                        if k in meta:
                            entry[k] = meta[k]
                skip_report.append(entry)
                reason = entry["reason"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if skip_report:
            skip_path = os.path.join("logs", f"skipped_files_{now_str}.json")
            with open(skip_path, "w", encoding="utf-8") as sf:
                json.dump(skip_report, sf, indent=2, ensure_ascii=False)
            logger.info(f"Wrote skip report with {len(skip_report)} entries to {skip_path}")
            logger.info(f"Skip reasons: {reason_counts}")
    except Exception as e:
        logger.warning(f"Could not write skip report: {e}")

    # Prepare flattened chunk list (and optionally deduplicate) for embedding
    prepared_chunks: List[Dict[str, Any]] = []
    for item in results:
        if not item or item.get("skipped", False):
            continue
        for ch in item.get("chunks", []) or []:
            content = ch.get("content", "")
            if isinstance(content, str) and content.strip():
                prepared_chunks.append({"content": content, "meta": ch.get("meta", {})})

    if args.enable_dedup and prepared_chunks:
        logger.info("Starting chunk-level deduplication")
        dedup_config = {
            'similarity_threshold': args.dedup_threshold,
            'use_semantic_similarity': True,
            'use_path_filtering': True,
            'export_report': True
        }
        prepared_chunks = integrate_deduplication_into_pipeline(prepared_chunks, dedup_config)
        logger.info(f"Deduplication completed: {len(prepared_chunks)} unique chunks remain")
    
    # ColBERT token storage is handled during Qdrant indexing phase

    logger.info(f"Writing output to %s", args.output)
    def make_json_serializable(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_serializable(i) for i in obj]
        return obj

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(results), f, indent=2, ensure_ascii=False)

    if args.qdrant:
        logger.info("Setting up Qdrant document store with hybrid search capabilities")
        
        # Check if ColBERT tokens are enabled
        if args.enable_colbert_tokens:
            logger.info("Using Hybrid Qdrant Store (Dense + Sparse + ColBERT tokens)")
            from hybrid_qdrant_store import create_hybrid_store
            
            document_store = create_hybrid_store(
                url=args.qdrant_url,
                collection_name=args.qdrant_collection,
                api_key=args.qdrant_api_key,
                embedding_dim=emb_dim,
                enable_colbert=True,
                colbert_model=args.colbert_model,
                recreate_index=args.recreate_index
            )
        else:
            # Use standard Haystack QdrantDocumentStore for dense + sparse only
            logger.info("Using standard Qdrant Document Store (Dense + Sparse)")
            # Use a client with extended timeout to avoid transient timeouts
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=args.qdrant_url,
                api_key=args.qdrant_api_key if args.qdrant_api_key else None,
                timeout=60
            )
            try:
                document_store = QdrantDocumentStore(
                    client=client,
                    index=args.qdrant_collection,
                    embedding_dim=emb_dim,
                    recreate_index=args.recreate_index,
                    use_sparse_embeddings=args.use_sparse,
                    on_disk_payload=True,
                    sparse_idf=True,
                    hnsw_config={"m": 16, "ef_construct": 128},
                    similarity="cosine"
                )
            except TypeError:
                # Fallback for older integration without `client` kwarg
                document_store = QdrantDocumentStore(
                    url=args.qdrant_url,
                    api_key=Secret.from_token(args.qdrant_api_key) if args.qdrant_api_key else None,
                    index=args.qdrant_collection,
                    embedding_dim=emb_dim,
                    recreate_index=args.recreate_index,
                    use_sparse_embeddings=args.use_sparse,
                    on_disk_payload=True,
                    sparse_idf=True,
                    hnsw_config={"m": 16, "ef_construct": 128},
                    similarity="cosine"
                )
        
        # Initialize the FastEmbed embedders for both dense and sparse
        logger.info(f"Setting up FastEmbed embedders with models:")
        logger.info(f"  - Dense model: %s", args.embedding_model)
        logger.info(f"  - Sparse model: %s", args.sparse_embedding_model)
        
        # Process files with both dense and sparse embeddings in a pipeline
        dense_embedder = FastembedDocumentEmbedder(model=args.embedding_model)
        sparse_embedder = FastembedSparseDocumentEmbedder(model=args.sparse_embedding_model)
        
        # Warm up embedders to load models before use
        logger.info("Warming up embedders...")
        dense_embedder.warm_up()
        if args.use_sparse:
            sparse_embedder.warm_up()
        logger.info("Embedders warmed up and ready")
        
        # Build documents from extracted chunks (Docling preferred)
        haystack_docs = []
        total_chunks = 0
        for ch in prepared_chunks:
            content = ch.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            meta = ch.get("meta", {}).copy()
            haystack_docs.append(Document(content=content, meta=meta))
            total_chunks += 1
        logger.info(f"Prepared {len(haystack_docs)} documents from {total_chunks} chunks for embedding")
        
        if haystack_docs:
            logger.info(f"Embedding %s documents with FastEmbed", len(haystack_docs))
            
            # First generate sparse embeddings
            if args.use_sparse:
                try:
                    logger.info("Generating sparse embeddings...")
                    sparse_result = sparse_embedder.run(haystack_docs)
                    haystack_docs = sparse_result["documents"]
                    logger.info(f"Successfully generated sparse embeddings for %s documents", len(haystack_docs))
                except Exception as e:
                    logger.error(f"Error generating sparse embeddings: %s", e)
                    logger.error(traceback.format_exc())
                    # Continue with dense embeddings even if sparse fails
            
            # Then generate dense embeddings
            try:
                logger.info("Generating dense embeddings...")
                dense_result = dense_embedder.run(haystack_docs)
                haystack_docs = dense_result["documents"]
                logger.info(f"Successfully generated dense embeddings for %s documents", len(haystack_docs))
            except Exception as e:
                logger.error(f"Error generating dense embeddings: %s", e)
                logger.error(traceback.format_exc())
            
            # Verify that documents have the expected embeddings
            if haystack_docs:
                sample_doc = haystack_docs[0]
                has_dense = hasattr(sample_doc, 'embedding') and sample_doc.embedding is not None
                has_sparse = hasattr(sample_doc, 'sparse_embedding') and sample_doc.sparse_embedding is not None
                has_content = hasattr(sample_doc, 'content') and sample_doc.content is not None and str(sample_doc.content).strip() != ""
                logger.info(f"Embedding verification: dense=%s, sparse=%s, content=%s", has_dense, has_sparse, has_content)
                
                # Log content information for debugging
                if has_content:
                    text_str = str(sample_doc.content)
                    logger.info(f"Sample document content length: %s", len(text_str))
                    logger.info(f"Sample document content snippet: %s...", text_str[:100])
                else:
                    logger.warning("Sample document does not have content.")
                
                if has_dense or has_sparse:
                    # Write to Qdrant
                    batch_writer = BatchDocumentWriter(document_store)
                    logger.info(f"Writing %s documents to Qdrant collection '%s'", len(haystack_docs), args.qdrant_collection)
                    try:
                        # Use the improved BatchDocumentWriter which expects Document objects
                        batch_writer.run(haystack_docs)
                        if batch_writer.stats.get("write_errors", 0) == 0:
                            logger.info(f"Successfully wrote %s documents to Qdrant", len(haystack_docs))
                        else:
                            logger.error(f"Qdrant write completed with {batch_writer.stats.get('write_errors')} write errors")
                    except Exception as e:
                        logger.error(f"Error writing to Qdrant: %s", e)
                        logger.error(traceback.format_exc())
                else:
                    logger.error("Documents have neither dense nor sparse embeddings. Check the embedder configuration.")
            else:
                logger.warning("No documents after embedding process. Check the embedder output.")
        else:
            logger.warning("No documents with embeddings to write to Qdrant")

    # Verify documents were stored correctly
    if args.qdrant:
        try:
            logger.info("Verifying document storage in Qdrant...")
            verify_qdrant_document_storage(args.qdrant_url, args.qdrant_api_key, args.qdrant_collection)
            
            # Verify ColBERT collection if tokens were enabled
            if args.enable_colbert_tokens:
                colbert_collection = f"{args.qdrant_collection}_colbert"
                logger.info("Verifying ColBERT token storage...")
                verify_colbert_collection(args.qdrant_url, args.qdrant_api_key, colbert_collection)
                
        except Exception as e:
            logger.error(f"Error during Qdrant verification: %s", e)
            logger.error(traceback.format_exc())

    logger.info("Done!")

def verify_qdrant_document_storage(qdrant_url, qdrant_api_key, collection_name, limit=5):
    """
    Verifies that documents were stored correctly in Qdrant with all expected fields,
    especially the 'content' field containing the document text.
    
    Args:
        qdrant_url: URL of the Qdrant server
        qdrant_api_key: API key for Qdrant (may be None for local instances)
        collection_name: Name of the collection to verify
        limit: Number of documents to retrieve for verification
        
    Returns:
        None, but logs verification results
    """
    logger.info(f"Connecting to Qdrant at %s, collection: %s", qdrant_url, collection_name)
    
    # Import QdrantClient directly
    try:
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError:
        logger.error("qdrant-client not installed. Install with: uv add qdrant-client")
        return
    
    def mask_api_key(key):
        if not key:
            return None
        if len(key) <= 6:
            return "*" * len(key)
        return key[:2] + "*" * (len(key)-6) + key[-4:]

    # Qdrant connection resolution: CLI > .env > docker fallback > exit
    qdrant_url = clean_env_var(qdrant_url) if qdrant_url else None
    qdrant_api_key = clean_env_var(qdrant_api_key) if qdrant_api_key else None
    qdrant_collection = collection_name or os.getenv("QDRANT_COLLECTION") or QDRANT_DEFAULT_COLLECTION

    # Step 2: .env
    if not qdrant_url:
        qdrant_url = clean_env_var(os.getenv("QDRANT_URL"))
    if not qdrant_api_key:
        qdrant_api_key = clean_env_var(os.getenv("QDRANT_API_KEY"))

    # Step 3: Docker fallback if QDRANT_TYPE=docker
    qdrant_type = os.getenv("QDRANT_TYPE", "docker").lower()
    docker_used = False
    if not qdrant_url:
        if qdrant_type == "docker":
            # Check if docker is running and qdrant container is up
            import subprocess
            try:
                docker_ps = subprocess.run([
                    "docker", "ps", "--filter", "name=qdrant", "--filter", "status=running", "--format", "{{.Names}}"
                ], capture_output=True, text=True)
                running_containers = docker_ps.stdout.strip().splitlines()
                if any("qdrant" in name for name in running_containers):
                    qdrant_url = QDRANT_DEFAULT_URL  # fallback to default docker URL
                    docker_used = True
                    logger.info("Falling back to Docker Qdrant instance at %s", qdrant_url)
                else:
                    logger.error("QDRANT_TYPE is set to 'docker' but no running Qdrant Docker container was found.")
                    logger.error("Please start the Qdrant Docker container or specify QDRANT_URL in your .env file.")
                    sys.exit(1)
            except Exception as e:
                logger.error("Error checking Docker containers: %s", e)
                sys.exit(1)
        else:
            logger.error("No Qdrant URL specified via CLI or .env, and QDRANT_TYPE is not 'docker'.")
            logger.error("Please specify QDRANT_URL in your .env file or use --qdrant_url on the command line.")
            sys.exit(1)

    # Final check: if still no URL, exit
    if not qdrant_url:
        logger.error("Could not resolve Qdrant URL from CLI, .env, or Docker fallback.")
        sys.exit(1)

    logger.info(f"Qdrant connection resolved:")
    logger.info(f"  URL: {qdrant_url}")
    logger.info(f"  Collection: {qdrant_collection}")
    logger.info(f"  API Key: {mask_api_key(qdrant_api_key)}")
    if docker_used:
        logger.info("Qdrant Docker fallback was used for this run.")

    # Connect to Qdrant with extended timeout for large collections
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key if qdrant_api_key else None,
        timeout=60  # Increased from default 5s to handle large collections
    )
    
    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection info: %s", collection_info)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Collection {collection_name} does not exist or cannot be accessed: {error_msg}")
        
        # Print raw error for debugging
        if hasattr(e, 'response') and hasattr(e.response, 'content'):
            logger.error(f"Raw response content: {e.response.content}")
        
        # Check for collection doesn't exist error - handle multiple possible error messages
        if "doesn't exist" in error_msg or "Not found: Collection" in error_msg or "404 Not Found" in error_msg:
            logger.warning(f"WARNING: Collection '{collection_name}' does not exist. This may indicate that no documents were successfully written.")
            logger.warning(f"Creating empty collection '{collection_name}' to allow verification to proceed...")
            try:
                # Create the collection with proper configuration
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text-dense": {"size": 1024, "distance": "Cosine"}
                    }
                )
                
                # Add sparse vector support
                try:
                    client.update_collection(
                        collection_name=collection_name,
                        sparse_vectors_config={
                            "text-sparse": {"index": {"on_disk": False}, "modifier": "idf"}
                        }
                    )
                except Exception as sparse_err:
                    logger.warning(f"Could not add sparse vector support: {sparse_err}")
                
                logger.info(f"Successfully created empty collection '{collection_name}'")
                # Continue with verification of the empty collection
            except Exception as create_err:
                logger.error(f"Failed to create collection: {create_err}")
                return
        else:
            # If it's another type of error, return
            return
    
    # Get document count
    count_result = client.count(collection_name=collection_name)
    total_docs = count_result.count
    logger.info(f"Total documents in collection: %s", total_docs)
    
    if total_docs == 0:
        logger.warning("Collection exists but contains no documents!")
        return
    
    # Scroll through some documents to verify content and vector fields
    try:
        logger.info(f"Retrieving %s documents to verify payload contents...", limit)
        scroll_response = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        
        points = scroll_response[0]  # First element is list of points
        
        if not points:
            logger.warning("No points returned from scroll request!")
            return
        
        # Verify each document
        for i, point in enumerate(points):
            point_id = point.id
            payload = point.payload
            # Fix: Use 'vector' not 'vectors'
            vector = point.vector

            logger.info(f"Document %s/%s (ID: %s):", i+1, len(points), point_id)

            # Check for content field
            if "content" in payload:
                content_preview = payload["content"][:100] + "..." if len(payload["content"]) > 100 else payload["content"]
                logger.info(f"  Content field exists with %s characters", len(payload['content']))
                logger.info(f"  Content preview: %s", content_preview)
            else:
                logger.error(f"  Content field missing from payload!")
                logger.info(f"  Available fields: %s", list(payload.keys()))

            # Check for vectors
            if vector:
                if isinstance(vector, dict):
                    has_dense = "text-dense" in vector
                    has_sparse = "text-sparse" in vector
                    logger.info(f"  Vectors: dense=%s, sparse=%s", has_dense, has_sparse)
                    if has_dense:
                        dense_length = len(vector['text-dense'])
                        logger.info(f"  Dense vector dimension: %s", dense_length)
                    if has_sparse:
                        sparse_vec = vector['text-sparse']
                        if hasattr(sparse_vec, "indices") and hasattr(sparse_vec, "values"):
                            indices_len = len(sparse_vec.indices)
                            values_len = len(sparse_vec.values)
                            logger.info(f"  Sparse vector has %s non-zero elements", indices_len)
                        else:
                            logger.warning(f"  Sparse vector format unexpected: %s", type(sparse_vec))
                else:
                    logger.info(f"  Single vector found, length: %s", len(vector))
            else:
                logger.error(f"  No vectors found for document!")

            # Check metadata
            meta_fields = [k for k in payload.keys() if k != "content"]
            if meta_fields:
                logger.info(f"  Metadata fields: %s", meta_fields)
                if "path" in payload:
                    logger.info(f"  Path: %s", payload['path'])
                ocr_fields = [field for field in meta_fields if field.startswith("ocr_")]
                if ocr_fields:
                    logger.info(f"  OCR metadata fields: %s", ocr_fields)
                    if "ocr_quality_score" in payload:
                        logger.info(f"  OCR quality score: %s", payload['ocr_quality_score'])
            else:
                logger.warning(f"  No metadata fields found!")

            logger.info("-" * 50)  # Separator between documents
        
        logger.info(f"Verification complete. Checked %s out of %s documents.", len(points), total_docs)
        
        if len(points) > 0:
            logger.info("SUCCESS: Documents appear to be stored correctly with content field and vectors!")
        else:
            logger.warning("WARNING: No documents were retrieved during verification!")
    
    except Exception as e:
        logger.error(f"Error during verification: %s", e)
        logger.error(traceback.format_exc())

def verify_colbert_collection(qdrant_url, qdrant_api_key, collection_name, limit=3):
    """
    Verifies that ColBERT token collection was created and populated correctly.
    
    Args:
        qdrant_url: URL of the Qdrant server
        qdrant_api_key: API key for Qdrant (may be None for local instances) 
        collection_name: Name of the ColBERT collection to verify
        limit: Number of documents to retrieve for verification
        
    Returns:
        None, but logs verification results
    """
    logger.info(f"🔍 Verifying ColBERT collection: {collection_name}")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
            timeout=60  # Consistent with main verification timeout
        )
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name)
        except Exception as e:
            error_msg = str(e)
            if "doesn't exist" in error_msg or "Not found" in error_msg or "404" in error_msg:
                logger.warning(f"⚠️  ColBERT collection '{collection_name}' does not exist")
                logger.warning("   This may indicate that ColBERT token storage failed during processing")
                return
            else:
                logger.error(f"❌ Error accessing ColBERT collection: {error_msg}")
                return
        
        # Get collection statistics  
        point_count = collection_info.points_count
        vector_config = collection_info.config.params.vectors
        
        logger.info(f"📊 ColBERT Collection Info:")
        logger.info(f"   • Points: {point_count}")
        logger.info(f"   • Vector dimension: {vector_config.size}")
        logger.info(f"   • Distance metric: {vector_config.distance}")
        logger.info(f"   • Multivector mode: {vector_config.multivector_config.comparator if vector_config.multivector_config else 'None'}")
        
        if vector_config.quantization_config:
            logger.info(f"   • Quantization: {vector_config.quantization_config.scalar.type}")
        
        if point_count == 0:
            logger.warning("⚠️  ColBERT collection exists but is empty!")
            logger.warning("   This indicates ColBERT token embedding may have failed")
            return
        
        # Sample some documents to verify token structure
        try:
            logger.info(f"🔍 Sampling {min(limit, point_count)} ColBERT token documents...")
            scroll_response = client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            
            points = scroll_response[0]
            
            if points:
                logger.info(f"✅ Retrieved {len(points)} ColBERT documents for inspection:")
                
                for i, point in enumerate(points):
                    logger.info(f"   📄 Document {i+1} (ID: {point.id}):")
                    
                    # Check payload
                    payload = point.payload
                    if payload:
                        logger.info(f"      • Payload fields: {list(payload.keys())}")
                        if "haystack_id" in payload:
                            logger.info(f"      • Linked to main collection: {payload['haystack_id']}")
                        if "content" in payload:
                            content_length = len(payload["content"])
                            logger.info(f"      • Content length: {content_length} chars")
                    else:
                        logger.warning(f"      ⚠️  No payload found")
                    
                    # Check token matrix
                    vector = point.vector
                    if vector is not None:
                        if isinstance(vector, list) and len(vector) > 0:
                            # Multiple token vectors (token matrix)
                            num_tokens = len(vector)
                            if num_tokens > 0:
                                token_dim = len(vector[0]) if isinstance(vector[0], list) else "unknown"
                                logger.info(f"      • ColBERT tokens: {num_tokens} x {token_dim} dimensions")
                            else:
                                logger.warning(f"      ⚠️  Empty token matrix")
                        else:
                            logger.warning(f"      ⚠️  Unexpected vector format: {type(vector)}")
                    else:
                        logger.warning(f"      ⚠️  No token vectors found")
                
                logger.info(f"✅ ColBERT collection verification completed successfully!")
                logger.info(f"   Collection contains {point_count} documents with token embeddings")
                
            else:
                logger.warning("⚠️  No documents retrieved from ColBERT collection")
                
        except Exception as sample_err:
            logger.error(f"❌ Error sampling ColBERT collection: {sample_err}")
            
    except Exception as e:
        logger.error(f"❌ ColBERT collection verification failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
