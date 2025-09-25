"""
OCR Quality Checker module
This module provides functions to assess OCR quality and determine when re-OCR is needed.
It also supports comparing different OCR engines and selecting the best result.
"""

import re
import logging
import string
import sys
import traceback
import os
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import shutil
import subprocess

import spacy
from spacy.language import Language
from docling.datamodel.document import DoclingDocument, ConversionResult
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import (
    PipelineOptions, PaginatedPipelineOptions, PdfPipelineOptions, 
    TesseractOcrOptions, EasyOcrOptions, OcrOptions, OcrMacOptions, 
    RapidOcrOptions
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize spaCy model for language detection and word recognition
def load_spacy_model():
    """Attempt to load a spaCy model."""
    # Try to load already-installed models without attempting downloads. en_core_web_trf is the most accurate but slow and resource heavy
    models_to_try = ["en_core_web_trf", "en_core_web_sm"]
    
    for model_name in models_to_try: 
        try:
            # Try loading the model
            logger.info(f"Attempting to load spaCy model: {model_name}")
            return spacy.load(model_name)
        except Exception as e:
            logger.warning(f"Could not load spaCy model {model_name}: {e}")
            continue
    
    # If we get here, all models failed
    logger.warning("No spaCy models could be loaded. Will use regex-based quality detection instead.")
    logger.info("To install spaCy models manually, run: uv run -m spacy download en_core_web_sm")
    return None

# Load spaCy model
try:
    nlp = load_spacy_model()
except:
    logger.warning("Error during spaCy model loading, falling back to regex-based quality detection")
    nlp = None

# Regular expressions for quality assessment
NON_WORD_PATTERN = re.compile(r'[^\w\s\.,;:!?\'"\-–—()[\]{}]')
SEQUENTIAL_JUNK_PATTERN = re.compile(r'[\x00-\x1F\x7F-\x9F]{3,}')
CONSECUTIVE_SYMBOLS_PATTERN = re.compile(r'[^\w\s]{4,}')
WORD_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')

def analyze_ocr_quality(text: str, non_word_threshold: float = 0.03, 
                       use_spacy: bool = True) -> Tuple[float, bool]:
    """
    Analyze OCR quality of text to determine if re-OCR is needed.
    
    Args:
        text: The OCR text to analyze
        non_word_threshold: Threshold above which text is considered poor quality (0.0-1.0)
        use_spacy: Whether to use spaCy for more accurate word detection when available
        
    Returns:
        Tuple containing:
            - non_word_ratio: Ratio of non-word characters to total text length
            - needs_reocr: Boolean indicating if re-OCR is recommended
    """
    if not text or len(text) < 10:
        logger.info("Text too short or empty, marking for re-OCR")
        return 1.0, True
    
    # Basic character-level quality checks
    total_chars = len(text)
    text_sample = text[:min(100, len(text))]  # Sample for logging
    
    # Log OCR text sample for better debugging
    logger.debug(f"Text sample (first 100 chars): {repr(text_sample)}")
    
    # Analysis results dictionary for detailed logging
    results = {
        "total_chars": total_chars,
        "sequential_junk": False,
        "consecutive_symbols": False,
        "low_word_ratio": False,
        "spacy_low_word_ratio": False,
        "non_word_count": 0,
        "word_count": 0,
        "recognizable_words": 0,
        "penalties": 0
    }
    
    # Check for sequential junk characters (control chars, etc.)
    if SEQUENTIAL_JUNK_PATTERN.search(text):
        logger.info("Found sequential junk characters in OCR text")
        results["sequential_junk"] = True
        return 1.0, True
        
    # Count non-word characters
    non_word_matches = NON_WORD_PATTERN.findall(text)
    results["non_word_count"] = len(non_word_matches)
    
    # Check for unusual sequences of symbols
    if CONSECUTIVE_SYMBOLS_PATTERN.search(text):
        symbol_matches = CONSECUTIVE_SYMBOLS_PATTERN.findall(text)
        logger.info(f"Found {len(symbol_matches)} consecutive symbol sequences indicating poor OCR")
        results["consecutive_symbols"] = True
        results["penalties"] += 10  # Penalize text with unusual symbol patterns
    
    # Check word-to-text ratio
    word_matches = WORD_PATTERN.findall(text)
    results["word_count"] = len(word_matches)
    text_length = len(text)
    
    word_text_ratio = results["word_count"] / (text_length / 5) if text_length > 0 else 0
    
    if text_length > 0 and word_text_ratio < 0.3:
        # Fewer recognizable words than expected - likely poor OCR
        logger.info(f"Low word-to-text ratio ({word_text_ratio:.4f}) indicating poor OCR")
        results["low_word_ratio"] = True
        penalty = text_length * 0.1
        results["penalties"] += penalty  # Additional penalty
    
    # For more precise analysis, use spaCy when available
    if use_spacy and nlp is not None:
        try:
            # Limit to first 10000 chars for performance
            sample_text = text[:10000]
            with nlp.select_pipes(disable=["parser", "ner"]):  # Disable unnecessary components for better performance
                # Process the text in a way that avoids memory leaks
                spacy_doc = nlp(sample_text)
                
                # Check for recognized words in the text
                recognizable_words = [token.text for token in spacy_doc if token.is_alpha and len(token.text) > 1]
                results["recognizable_words"] = len(recognizable_words)
                word_ratio = results["recognizable_words"] / max(1, len(spacy_doc))
                
                # If word ratio is very low, text is likely poor OCR
                if word_ratio < 0.25 and len(spacy_doc) > 10:
                    logger.info(f"SpaCy analysis: Low recognized word ratio: {word_ratio:.4f}")
                    results["spacy_low_word_ratio"] = True
                    penalty = text_length * 0.1
                    results["penalties"] += penalty  # Additional penalty
            
            # Cleanup to avoid memory leaks
            del spacy_doc
                
        except Exception as e:
            logger.warning(f"Error in spaCy analysis: {e}")
            logger.warning(f"Stack trace: {traceback.format_exc()}")
    
    # Calculate non-word ratio including penalties
    non_word_count = results["non_word_count"] + results["penalties"]
    non_word_ratio = non_word_count / max(1, total_chars)
    
    # Determine if re-OCR is needed
    needs_reocr = non_word_ratio > non_word_threshold
    
    # Log detailed analysis results
    logger.info(f"OCR quality analysis results: {results}")
    
    return non_word_ratio, needs_reocr

def create_ocr_options(lang: List[str] = None, 
                    engine: str = None) -> Tuple[Optional[OcrOptions], Optional[str]]:
    """
    Create OCR options for a specific OCR engine or the best available one.
    Order (if available and no engine specified): OCRmac (macOS only), Tesseract, RapidOCR, EasyOCR.
    
    Args:
        lang: List of language codes for OCR
        engine: Specific engine to use ("OCRmac", "Tesseract", "RapidOCR", "EasyOCR")
              If None, tries all engines in order of preference
        
    Returns:
        Tuple containing:
            - OcrOptions object configured for optimal OCR (or None if error)
            - Name of the OCR engine used (e.g., "OCRmac", "Tesseract", "EasyOCR") or None
    """
    if lang is None:
        lang = ["en"]  # Default to English
    
    # If specific engine is requested, only try that one
    if engine:
        logger.info(f"Attempting to configure {engine} OCR options with language: {lang}")
        
        # OCRmac
        if engine.lower() == "ocrmac":
            if sys.platform != "darwin":
                logger.warning("OCRmac is only available on macOS. Falling back to Tesseract.")
                engine = "Tesseract"
            else:
                try:
                    # OCRmac requires specific language codes like "en-US" instead of just "en"
                    if lang == ["en"]:
                        ocrmac_lang = ["en-US"]
                    elif lang == ["fr"]:
                        ocrmac_lang = ["fr-FR"]
                    elif lang == ["de"]:
                        ocrmac_lang = ["de-DE"]
                    elif lang == ["es"]:
                        ocrmac_lang = ["es-ES"]
                    else:
                        # Default to English if we don't have a mapping
                        ocrmac_lang = ["en-US"]
                        
                    logger.info(f"Mapping language codes for OCRmac: {lang} -> {ocrmac_lang}")
                    options = OcrMacOptions(lang=ocrmac_lang)
                    return options, "OCRmac"
                except Exception as e_mac:
                    logger.warning(f"Could not create OCRmac options: {e_mac}")
                    engine = "Tesseract"  # Fall back to Tesseract
        
        # Tesseract
        if engine.lower() == "tesseract":
            try:
                # Convert language codes to Tesseract format if needed
                if lang == ["en"]:
                    tesseract_lang = ["eng"]
                elif lang == ["fr"]:
                    tesseract_lang = ["fra"]
                elif lang == ["de"]:
                    tesseract_lang = ["deu"]
                elif lang == ["es"]:
                    tesseract_lang = ["spa"]
                else:
                    tesseract_lang = lang
                    
                # Allow explicit tesseract binary via env to avoid Homebrew vs system mismatch
                # TesseractOcrOptions.path expects a tessdata directory, not the binary
                tessdata_prefix = os.getenv("TESSDATA_PREFIX")
                tesseract_cmd = os.getenv("TESSERACT_CMD")
                if tesseract_cmd and not shutil.which(tesseract_cmd):
                    logger.warning(f"TESSERACT_CMD set but not found on PATH: {tesseract_cmd}")
                if tessdata_prefix and not os.path.isdir(tessdata_prefix):
                    logger.warning(f"TESSDATA_PREFIX is set but not a directory: {tessdata_prefix}")
                logger.info(f"Configuring TesseractOCR options with lang={tesseract_lang}, tessdata_path={tessdata_prefix}")
                options = TesseractOcrOptions(
                    lang=tesseract_lang,
                    path=tessdata_prefix if tessdata_prefix else None
                )
                return options, "Tesseract"
            except Exception as e_tess:
                logger.warning(f"Could not create TesseractOCR options: {e_tess}")
                engine = "EasyOCR"  # Fall back to EasyOCR
                
        # RapidOCR
        if engine.lower() == "rapidocr":
            try:
                logger.info(f"Configuring RapidOCR options with lang={lang}")
                # RapidOCR has its own language mapping
                if lang == ["en"]:
                    rapidocr_lang = ["en"]
                else:
                    rapidocr_lang = lang
                    
                options = RapidOcrOptions(
                    lang=rapidocr_lang,
                    det_model="en_PP-OCRv3_det",  # English detector model
                    rec_model="en_PP-OCRv3"  # English recognition model
                )
                return options, "RapidOCR"
            except Exception as e_rapid:
                logger.warning(f"Could not create RapidOCR options: {e_rapid}")
                engine = "EasyOCR"  # Fall back to EasyOCR
        
        # EasyOCR
        if engine.lower() == "easyocr":
            try:
                logger.info(f"Configuring EasyOCR options with lang={lang}")
                options = EasyOcrOptions(
                    lang=lang,
                    confidence_threshold=0.65,  # Slightly higher threshold for better accuracy
                    download_enabled=True
                )
                return options, "EasyOCR"
            except Exception as e_easy:
                logger.warning(f"Could not create EasyOCR options: {e_easy}")
    
    # No specific engine requested, try all in order
    logger.info(f"Attempting to configure OCR options with language: {lang} (Preference: OCRmac > Tesseract > RapidOCR > EasyOCR)")

    # 1. Try OCRmac first (macOS only)
    if sys.platform == "darwin":
        try:
            logger.debug("Trying OCRmac options")
            # OCRmac requires specific language codes like "en-US" instead of just "en"
            if lang == ["en"]:
                ocrmac_lang = ["en-US"]
            elif lang == ["fr"]:
                ocrmac_lang = ["fr-FR"]
            elif lang == ["de"]:
                ocrmac_lang = ["de-DE"]
            elif lang == ["es"]:
                ocrmac_lang = ["es-ES"]
            else:
                # Default to English if we don't have a mapping
                ocrmac_lang = ["en-US"]
                
            logger.info(f"Mapping language codes for OCRmac: {lang} -> {ocrmac_lang}")
            options = OcrMacOptions(lang=ocrmac_lang)
            return options, "OCRmac"
        except Exception as e_mac:
            logger.warning(f"Could not create OCRmac options: {e_mac}")
    else:
        logger.debug("Skipping OCRmac (not on macOS)")

    # 2. Try Tesseract second
    try:
        # Convert language codes to Tesseract format if needed
        if lang == ["en"]:
            tesseract_lang = ["eng"]
        elif lang == ["fr"]:
            tesseract_lang = ["fra"]
        elif lang == ["de"]:
            tesseract_lang = ["deu"]
        elif lang == ["es"]:
            tesseract_lang = ["spa"]
        else:
            tesseract_lang = lang
            
        # Allow explicit tesseract binary via env to avoid Homebrew vs system mismatch
        tessdata_prefix = os.getenv("TESSDATA_PREFIX")
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd and not shutil.which(tesseract_cmd):
            logger.warning(f"TESSERACT_CMD set but not found on PATH: {tesseract_cmd}")
        if tessdata_prefix and not os.path.isdir(tessdata_prefix):
            logger.warning(f"TESSDATA_PREFIX is set but not a directory: {tessdata_prefix}")
        logger.debug(f"Trying TesseractOCR options with lang={tesseract_lang}, tessdata_path={tessdata_prefix}")
        options = TesseractOcrOptions(
            lang=tesseract_lang,
            path=tessdata_prefix if tessdata_prefix else None
        )
        return options, "Tesseract"
    except Exception as e_tess:
        logger.warning(f"Could not create TesseractOCR options: {e_tess}")
        
    # 3. Try RapidOCR third
    try:
        logger.debug("Trying RapidOCR options")
        # RapidOCR has its own language mapping
        if lang == ["en"]:
            rapidocr_lang = ["en"]
        else:
            rapidocr_lang = lang
            
        options = RapidOcrOptions(
            lang=rapidocr_lang,
            det_model="en_PP-OCRv3_det",  # English detector model
            rec_model="en_PP-OCRv3"  # English recognition model
        )
        return options, "RapidOCR"
    except Exception as e_rapid:
        logger.warning(f"Could not create RapidOCR options: {e_rapid}")

    # 4. Fall back to EasyOCR 
    try:
        logger.debug("Trying EasyOCR options as final fallback")
        options = EasyOcrOptions(
            lang=lang,
            confidence_threshold=0.65,  # Slightly higher threshold for better accuracy
            download_enabled=True
        )
        return options, "EasyOCR"
    except Exception as e_easy:
        logger.warning(f"Could not create EasyOCR options: {e_easy}")
            
    # 5. Last resort - return None if no engine works
    logger.error("Could not configure ANY OCR engine.")
    return None, None

def assess_ocr_quality(doc: DoclingDocument) -> Dict[str, Any]:
    """
    Assess the quality of OCR text in a document.
    
    Args:
        doc: DoclingDocument to assess
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Extract text from document
        text = doc.export_to_text()
        
        # Calculate metrics
        total_length = len(text)
        if total_length < 10:
            return {
                "score": 0.0,
                "total_chars": total_length,
                "readable_chars": 0,
                "word_count": 0,
                "avg_word_length": 0.0,
                "non_word_ratio": 1.0,
                "error": "Text too short"
            }
        
        # Get number of alphanumeric characters
        readable_chars = sum(1 for c in text if c.isalnum() or c.isspace())
        
        # Get number of words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        word_count = len(words)
        
        # Calculate average word length (if there are words)
        avg_word_length = sum(len(w) for w in words) / max(1, word_count)
        
        # Calculate non-word ratio
        non_word_ratio = 1.0 - (readable_chars / max(1, total_length))
        
        # Calculate an overall quality score (higher is better)
        # Based on weighted combination of metrics
        # - word count weight: 0.4
        # - avg word length weight: 0.3 
        # - non-word ratio weight: 0.3 (inverted as lower is better)
        
        # Normalize metrics to 0-1 scale (based on typical values)
        norm_word_count = min(1.0, word_count / 1000)  # Normalize up to 1000 words
        norm_avg_length = min(1.0, avg_word_length / 8)  # Normalize up to avg length of 8
        norm_non_word = 1.0 - min(1.0, non_word_ratio * 10)  # Invert and normalize
        
        # Calculate weighted score
        score = (0.4 * norm_word_count) + (0.3 * norm_avg_length) + (0.3 * norm_non_word)
        
        return {
            "score": score,
            "total_chars": total_length,
            "readable_chars": readable_chars,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "non_word_ratio": non_word_ratio
        }
        
    except Exception as e:
        logger.error(f"Error assessing OCR quality: {e}")
        return {
            "score": 0.0,
            "error": str(e)
        }

def perform_ocr_with_engine(file_path: str, engine: str) -> Tuple[Optional[DoclingDocument], Dict[str, Any]]:
    """
    Process a document with a specific OCR engine.
    
    Args:
        file_path: Path to the document file
        engine: OCR engine to use
        
    Returns:
        Tuple containing:
            - DoclingDocument with OCR applied
            - Dictionary with OCR metadata and quality assessment
    """
    logger.info(f"Processing {file_path} with {engine} OCR engine")
    ocr_metadata = {"ocr_engine_used": engine}
    
    try:
        # Import required components
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import FormatOption
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        # Create OCR options for the specified engine
        ocr_options, actual_engine = create_ocr_options(lang=["en"], engine=engine)
        if not ocr_options:
            raise ValueError(f"Could not configure {engine} OCR options")
        
        # Update engine name in case we had to fall back to a different engine
        ocr_metadata["ocr_engine_used"] = actual_engine
        
        # Configure pipeline options with OCR enabled
        def _env_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            val = str(val).strip().lower()
            if val in {"1", "true", "yes", "on"}:
                return True
            if val in {"0", "false", "no", "off"}:
                return False
            return default

        do_table = _env_bool("DOCLING_DO_TABLE_STRUCTURE", True)
        do_formula = _env_bool("DOCLING_DO_FORMULA_ENRICHMENT", True)
        if not do_formula:
            logger.info("Docling: formula enrichment disabled via DOCLING_DO_FORMULA_ENRICHMENT=0")
        if not do_table:
            logger.info("Docling: table structure disabled via DOCLING_DO_TABLE_STRUCTURE=0")

        options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
            # Enable additional enrichments (configurable via env for performance)
            do_table_structure=do_table,
            do_formula_enrichment=do_formula,
        )
        
        # Create format option
        pdf_option = FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=options, 
            backend=DoclingParseV4DocumentBackend
        )
        
        # Create converter
        start_time = time.time()
        converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_option}
        )
        
        # Convert document
        result = converter.convert(source=file_path)
        doc = result.document
        
        # Record processing time
        processing_time = time.time() - start_time
        ocr_metadata["processing_time"] = processing_time
        
        # Assess quality
        quality_metrics = assess_ocr_quality(doc)
        ocr_metadata["quality_metrics"] = quality_metrics
        
        logger.info(f"Completed {engine} OCR for {file_path} in {processing_time:.2f}s with quality score: {quality_metrics.get('score', 0):.4f}")
        
        return doc, ocr_metadata
        
    except Exception as e:
        logger.error(f"Error in {engine} OCR processing for {file_path}: {e}")
        logger.error(traceback.format_exc())
        
        # Create metadata with error info and no document
        ocr_metadata["error"] = str(e)
        ocr_metadata["quality_metrics"] = {"score": 0.0, "error": str(e)}
        
        # Return None for document and metadata
        return None, ocr_metadata

def compare_ocr_engines(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Compare different OCR engines on the same document.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with results for each engine
    """
    logger.info(f"Comparing OCR engines for {file_path}")
    
    # Define engines to test (in priority order)
    engines = []
    
    # Add OCRmac only on macOS
    if sys.platform == "darwin":
        engines.append("OCRmac")
    
    # Add other engines
    engines.extend(["Tesseract", "RapidOCR", "EasyOCR"])
    
    results = {}
    
    # Process with each engine
    for engine in engines:
        try:
            # Run OCR with this engine
            doc, metadata = perform_ocr_with_engine(file_path, engine)
            
            # Extract a sample of text for review
            text_sample = doc.export_to_text()[:2000] if doc else ""
            
            # Store results
            results[engine] = {
                "metadata": metadata,
                "text_sample": text_sample,
                "quality_score": metadata.get("quality_metrics", {}).get("score", 0.0),
                "word_count": metadata.get("quality_metrics", {}).get("word_count", 0),
                "processing_time": metadata.get("processing_time", 0.0)
            }
            
            logger.info(f"{engine} OCR quality score: {results[engine]['quality_score']:.4f}, " 
                       f"word count: {results[engine]['word_count']}, " 
                       f"time: {results[engine]['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error comparing {engine} OCR: {e}")
            results[engine] = {
                "error": str(e),
                "quality_score": 0.0
            }
    
    # Summarize results
    best_engine = max(results.keys(), key=lambda k: results[k].get("quality_score", 0.0))
    logger.info(f"Best OCR engine for {file_path} is {best_engine} with score {results[best_engine]['quality_score']:.4f}")
    
    return results

def select_best_ocr_result(results: Dict[str, Dict[str, Any]]) -> Tuple[Optional[DoclingDocument], Dict[str, Any]]:
    """
    Select the best OCR result from a comparison.
    
    Args:
        results: Dictionary with OCR comparison results
        
    Returns:
        Tuple containing:
            - DoclingDocument with best OCR 
            - Dictionary with OCR metadata
    """
    if not results:
        logger.error("No OCR results to select from")
        return None, {"ocr_engine_used": None, "error": "No OCR results available"}
    
    # Find the best engine based on quality score
    best_engine = max(results.keys(), key=lambda k: results[k].get("quality_score", 0.0))
    best_score = results[best_engine]["quality_score"]
    
    logger.info(f"Selected {best_engine} as best OCR engine with quality score {best_score:.4f}")
    
    # Get the document and metadata for the best engine
    best_doc = results.get(best_engine, {}).get("document")
    if best_doc is None:
        # Need to rerun OCR with the best engine to get the document
        best_doc, _ = perform_ocr_with_engine(results.get(best_engine, {}).get("file_path", ""), best_engine)
    
    # Create metadata
    best_metadata = {
        "ocr_engine_used": best_engine,
        "ocr_quality_score": best_score,
        "ocr_comparison": {k: {"score": v.get("quality_score", 0.0)} for k, v in results.items()}
    }
    
    return best_doc, best_metadata

def perform_reocr(file_path: str, ocr_threshold: float = 0.03, force_ocr: bool = False, 
                compare_engines: bool = False, engine: str = None) -> Tuple[Optional[DoclingDocument], dict]:
    """
    Process a document with fresh OCR if needed or forced.
    Can compare different OCR engines and select the best result.
    
    Args:
        file_path: Path to the document file
        ocr_threshold: Threshold above which text is considered poor quality
        force_ocr: If True, bypasses the quality check and forces re-OCR.
        compare_engines: If True, compares different OCR engines and selects best result
        engine: Specific OCR engine to use (if not comparing engines)
        
    Returns:
        Tuple containing:
            - DoclingDocument with fresh OCR applied if needed/forced
            - Dictionary with OCR metadata (e.g., {"ocr_engine_used": "Tesseract"})
    """
    logger.info(f"Checking OCR quality for: {file_path}")
    ocr_metadata = {"ocr_engine_used": None}  # Initialize metadata
    
    try:
        # Create options with OCR disabled first to check existing text
        initial_ocr_options, _ = create_ocr_options(lang=["en"])
        if not initial_ocr_options:
            raise ValueError("Could not configure initial OCR options for quality check.")
            
        options = PdfPipelineOptions(
            do_ocr=False,
            ocr_options=initial_ocr_options
        )
        
        # We need to create a format option for PDF with our custom pipeline options
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import FormatOption
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        # Create a custom format option for PDF
        pdf_option = FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=options, 
            backend=DoclingParseV4DocumentBackend
        )
        
        # First pass - get document with existing OCR
        converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_option}
        )
        result = converter.convert(source=file_path)
        doc = result.document
        
        # Initialize needs_reocr flag
        needs_reocr = False
        
        # If force_ocr is True, skip quality check and force re-OCR
        if force_ocr:
            logger.info(f"Force OCR enabled for {file_path}. Skipping quality check.")
            needs_reocr = True
        else:
            # Extract all text from the document
            text = doc.export_to_text()
            
            # Analyze OCR quality
            non_word_ratio, needs_reocr = analyze_ocr_quality(text, ocr_threshold)
            logger.info(f"OCR quality analysis: non-word ratio = {non_word_ratio:.4f} ({non_word_ratio:.2%}), threshold = {ocr_threshold:.2%}, needs re-OCR: {needs_reocr}")
        
        # If quality is poor or OCR is forced, re-process with OCR enabled
        if needs_reocr:
            logger.info(f"Re-processing with fresh OCR for: {file_path}")
            
            try:
                # If comparing engines, run comparison and select best result
                if compare_engines:
                    logger.info(f"Comparing OCR engines for {file_path}")
                    comparison_results = compare_ocr_engines(file_path)
                    doc, engine_metadata = select_best_ocr_result(comparison_results)
                    ocr_metadata.update(engine_metadata)
                    logger.info(f"Selected best OCR engine: {ocr_metadata.get('ocr_engine_used')}")
                    
                # Otherwise, use specified engine or default cascade
                else:
                    # Create new options with OCR enabled
                    ocr_options, engine_name = create_ocr_options(lang=["en"], engine=engine)
                    if not ocr_options:
                         raise ValueError("Could not configure OCR options for re-processing.")
                    
                    # Store engine name and make sure it's a string value
                    if engine_name is None:
                        engine_name = "UnknownEngine"
                    ocr_metadata["ocr_engine_used"] = engine_name
                    logger.info(f"Using OCR engine: {engine_name} (stored in metadata)")
    
                    # Enrichment toggles via env for performance on math/slide-heavy PDFs
                    def _env_bool(name: str, default: bool) -> bool:
                        val = os.getenv(name)
                        if val is None:
                            return default
                        val = str(val).strip().lower()
                        if val in {"1", "true", "yes", "on"}:
                            return True
                        if val in {"0", "false", "no", "off"}:
                            return False
                        return default

                    do_table = _env_bool("DOCLING_DO_TABLE_STRUCTURE", True)
                    do_formula = _env_bool("DOCLING_DO_FORMULA_ENRICHMENT", True)
                    if not do_formula:
                        logger.info("Docling: formula enrichment disabled via DOCLING_DO_FORMULA_ENRICHMENT=0")
                    if not do_table:
                        logger.info("Docling: table structure disabled via DOCLING_DO_TABLE_STRUCTURE=0")

                    options = PdfPipelineOptions(
                        do_ocr=True,
                        ocr_options=ocr_options,
                        do_table_structure=do_table,
                        do_formula_enrichment=do_formula,
                    )
                    
                    # Create a custom format option for PDF with our custom pipeline options
                    pdf_option = FormatOption(
                        pipeline_cls=StandardPdfPipeline,
                        pipeline_options=options, 
                        backend=DoclingParseV4DocumentBackend
                    )
                    
                    # Re-convert with OCR enabled
                    converter = DocumentConverter(
                        format_options={InputFormat.PDF: pdf_option}
                    )
                    result = converter.convert(source=file_path)
                    doc = result.document
                
                logger.info(f"Completed fresh OCR for: {file_path}")
            except KeyboardInterrupt:
                logger.warning(f"OCR processing interrupted for {file_path}")
                # Keep the original document but clear the engine name
                ocr_metadata["ocr_engine_used"] = None
                return doc, ocr_metadata
            except Exception as e:
                logger.error(f"Error during re-OCR processing for {file_path}: {e}")
                logger.warning(f"Falling back to original document extraction for {file_path}")
                # Keep the original document but clear the engine name
                ocr_metadata["ocr_engine_used"] = None 
        else:
            logger.info(f"OCR quality acceptable or check skipped, using existing text: {file_path}")
        
        return doc, ocr_metadata # Return doc and metadata
        
    except KeyboardInterrupt:
        logger.warning(f"OCR processing interrupted for {file_path}")
        # Return no document with metadata on interrupt
        return None, {"ocr_engine_used": None}
    except Exception as e:
        logger.error(f"Error during initial document load for OCR check {file_path}: {e}")
        logger.error(traceback.format_exc()) # Add traceback for initial load error
        # Fall back to normal conversion without OCR quality checking
        logger.warning(f"Falling back to standard conversion with default settings for {file_path}")
        
        try:
            # Standard conversion with default settings
            logger.info(f"Using DocumentConverter with default settings for {file_path}")
            converter = DocumentConverter()
            result = converter.convert(source=file_path)
            
            # Log some basic info about the resulting document
            doc = result.document
            text_length = len(doc.export_to_text())
            logger.info(f"Standard conversion completed for {file_path}: got document with {text_length} characters")
            # Return doc and default (None) metadata on fallback
            return doc, ocr_metadata 
        except Exception as e2:
            logger.error(f"Error during fallback conversion: {e2}")
            # Last resort - return no document to avoid validation errors
            logger.error(f"Returning no document for {file_path}")
            return None, ocr_metadata
