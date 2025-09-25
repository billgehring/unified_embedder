"""
Enhanced Docling Converter with OCR quality checking
This module extends the DoclingConverter class to include OCR quality assessment.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple
import logging
import traceback

from haystack import Document, component
from docling.chunking import BaseChunk, BaseChunker, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling_haystack.converter import DoclingConverter, ExportType, BaseMetaExtractor, MetaExtractor

# Import OCR quality checker
from ocr_quality_checker import analyze_ocr_quality, perform_reocr

# Configure logging
logger = logging.getLogger(__name__)

@component
class EnhancedDoclingConverter:
    """Enhanced Docling Converter that checks OCR quality and performs re-OCR when needed."""
    
    def __init__(
        self,
        converter: Optional[DoclingConverter] = None,
        convert_kwargs: Optional[Dict[str, Any]] = None,
        export_type: ExportType = ExportType.DOC_CHUNKS,
        md_export_kwargs: Optional[Dict[str, Any]] = None,
        chunker: Optional[BaseChunker] = None,
        meta_extractor: Optional[BaseMetaExtractor] = None,
        ocr_quality_threshold: float = 0.03,
        enable_ocr_check: bool = True,
        max_ocr_retries: int = 1,
        force_ocr: bool = False,
        compare_engines: bool = False,
        preferred_ocr_engine: Optional[str] = None,
    ):
        """Create an Enhanced Docling Haystack converter with OCR quality checking.
        
        Args:
            converter: The Docling `DoclingConverter` to use
            convert_kwargs: Parameters to pass to Docling conversion
            export_type: The export mode to use
            md_export_kwargs: Parameters to pass to Markdown export
            chunker: The Docling chunker instance to use
            meta_extractor: The extractor for document metadata
            ocr_quality_threshold: Threshold for non-word ratio above which re-OCR is triggered
            enable_ocr_check: Whether to enable OCR quality checking and re-OCR
            max_ocr_retries: Maximum number of OCR retry attempts
            force_ocr: Whether to force re-OCR processing for all PDF documents
            compare_engines: If True, compares different OCR engines and selects the best one
            preferred_ocr_engine: Specific OCR engine to use if not comparing engines 
                                 (options: "OCRmac", "Tesseract", "RapidOCR", "EasyOCR")
        """
        # Initialize DoclingConverter-equivalent attributes
        self._converter = converter or DoclingConverter()
        self._convert_kwargs = convert_kwargs if convert_kwargs is not None else {}
        self._export_type = export_type
        self._md_export_kwargs = (
            md_export_kwargs
            if md_export_kwargs is not None
            else {"image_placeholder": ""}
        )
        self._chunker = chunker
        self._meta_extractor = meta_extractor or MetaExtractor()
        self.ocr_quality_threshold = ocr_quality_threshold
        self.enable_ocr_check = enable_ocr_check
        self.max_ocr_retries = max_ocr_retries
        self.force_ocr = force_ocr
        self.compare_engines = compare_engines
        self.preferred_ocr_engine = preferred_ocr_engine
        
        # Stats to track performance
        self.stats = {
            "total_processed": 0,
            "ocr_checks": 0,
            "ocr_retries": 0,
            "engines_compared": 0,
            "best_engine_counts": {},
            "errors": 0
        }
    
    def _select_chunker_for_filetype(self, filepath: str) -> Tuple[BaseChunker, dict]:
        """Selects the appropriate chunker based on file extension or type.
        
        Returns:
            Tuple containing:
                - The selected chunker instance
                - Dictionary with chunker metadata (type, parameters)
        """
        ext = str(filepath).lower()
        chunker_meta = {}
        
        # Use HybridChunker for PDFs, DOCX, PPTX, XML, JSON with configurable token limit
        if ext.endswith(('.pdf', '.docx', '.pptx', '.xml', '.json')):
            import os as _os
            tok_name = _os.getenv("CHUNKER_TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")
            try:
                max_toks = int(_os.getenv("CHUNKER_MAX_TOKENS", "512"))
            except Exception:
                max_toks = 512
            chunker = HybridChunker(
                tokenizer=tok_name,
                max_tokens=max_toks
            )
            
            # Get the actual tokenizer used
            tokenizer_info = {
                "name": tok_name,
                "type": chunker.tokenizer.__class__.__name__
            }
            
            # Get detailed chunker parameters
            chunker_meta = {
                "chunker_type": "HybridChunker",
                "tokenizer": tokenizer_info,
                "max_tokens": chunker.max_tokens,
                "merge_peers": chunker.merge_peers,
                "delimiter": chunker.delim,
                "strategy": "Combines document structure with token-aware chunking",
                "description": "First chunks by document structure, then ensures chunks don't exceed token limit"
            }
            
            # Add information about the inner chunker (HierarchicalChunker)
            try:
                inner_chunker = chunker._inner_chunker
                chunker_meta["inner_chunker"] = {
                    "type": inner_chunker.__class__.__name__,
                    "description": "Chunks based on document structure (headings, paragraphs, etc.)"
                }
            except:
                pass
        # Use default chunker for text, markdown, asciidoc, html
        else:
            chunker = self._chunker if hasattr(self, '_chunker') and self._chunker is not None else BaseChunker()
            
            # Get detailed chunker parameters
            chunker_meta = {
                "chunker_type": chunker.__class__.__name__,
                "delimiter": chunker.delim,
                "strategy": "Simple text chunking",
                "description": "Basic chunking without token awareness or structural analysis"
            }
            
            # If it's a custom chunker, try to extract additional parameters
            if chunker.__class__.__name__ != "BaseChunker":
                try:
                    # Try to get common chunker parameters
                    if hasattr(chunker, 'chunk_size'):
                        chunker_meta["chunk_size"] = chunker.chunk_size
                    if hasattr(chunker, 'chunk_overlap'):
                        chunker_meta["chunk_overlap"] = chunker.chunk_overlap
                except:
                    pass
            
        return chunker, chunker_meta

    def _process_pdf_with_ocr_check(self, filepath: str) -> Tuple[DoclingDocument, dict]:
        """Process a PDF file with OCR quality checking.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            Tuple containing:
             - DoclingDocument with potentially improved OCR
             - Dictionary containing OCR metadata update
        """
        try:
            self.stats["ocr_checks"] += 1
            
            # Determine if we should compare engines, and which ones to use
            if self.compare_engines:
                logger.info(f"Comparing OCR engines for {filepath}")
                self.stats["engines_compared"] += 1
                # Capture both document and metadata from perform_reocr with engine comparison
                doc, ocr_meta_update = perform_reocr(
                    file_path=filepath,
                    ocr_threshold=self.ocr_quality_threshold,
                    force_ocr=self.force_ocr,
                    compare_engines=True,
                    engine=None  # No specific engine when comparing
                )
                
                # Track which engine was selected as best
                best_engine = ocr_meta_update.get("ocr_engine_used")
                if best_engine:
                    if best_engine not in self.stats["best_engine_counts"]:
                        self.stats["best_engine_counts"][best_engine] = 0
                    self.stats["best_engine_counts"][best_engine] += 1
                
            else:
                # Use specified preferred engine or let the system choose
                doc, ocr_meta_update = perform_reocr(
                    file_path=filepath,
                    ocr_threshold=self.ocr_quality_threshold,
                    force_ocr=self.force_ocr,
                    compare_engines=False,
                    engine=self.preferred_ocr_engine
                )
            
            return doc, ocr_meta_update
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in OCR quality checking process for {filepath}: {e}")
            logger.error(traceback.format_exc())
            
            # Fall back to standard conversion
            logger.warning(f"Falling back to standard conversion for {filepath}")
            try:
                conversion_result = self._converter.convert(
                    source=filepath,
                    **self._convert_kwargs,
                )
                # Return the .document attribute from the ConversionResult and empty meta dict
                return conversion_result.document, {}
            except Exception as e2:
                logger.error(f"Standard conversion also failed for {filepath}: {e2}")
                # Return no document and empty meta dict as last resort
                return None, {}
    
    def _process_document_standard(self, filepath: str) -> DoclingDocument:
        """Process a document using standard conversion without OCR checks.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Converted DoclingDocument
        """
        try:
            # Use the DocumentConverter directly to get the DoclingDocument
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(source=filepath)
            # Return the DoclingDocument directly from the conversion result
            return result.document
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in standard conversion for {filepath}: {e}")
            logger.error(traceback.format_exc())
            # Return None to indicate failure, preventing ValidationError
            return None
    
    @component.output_types(documents=List[Document])
    def run(
        self,
        paths: Iterable[Union[Path, str]],
    ):
        """Run the EnhancedDoclingConverter with OCR quality checking and adaptive chunking."""
        documents: List[Document] = []
        
        for filepath in paths:
            str_filepath = str(filepath)
            self.stats["total_processed"] += 1

            try:
                ext = str_filepath.lower()
                is_pdf = ext.endswith('.pdf')
                is_txt = ext.endswith('.txt')
                is_vtt = ext.endswith('.vtt')
                is_xml = ext.endswith('.xml')
                
                # Check for additional text formats that should be handled directly
                is_md = ext.endswith('.md')
                is_rst = ext.endswith('.rst') 
                is_rtf = ext.endswith('.rtf')
                is_tex = ext.endswith('.tex')
                is_py = ext.endswith('.py')
                is_js = ext.endswith(('.js', '.jsx'))
                is_css = ext.endswith('.css')
                is_yaml = ext.endswith(('.yml', '.yaml'))
                is_ini = ext.endswith('.ini')
                is_log = ext.endswith('.log')
                is_asciidoc = ext.endswith(('.asciidoc', '.adoc'))
                
                # Supported Docling formats (binary/structured documents)
                docling_supported_exts = (
                    '.pdf', '.docx', '.pptx', '.html', '.htm', '.json', '.csv', '.xlsx'
                )
                
                # Handle text-based files directly for better control and encoding handling
                text_based_files = (is_txt or is_vtt or is_xml or is_md or is_rst or 
                                  is_rtf or is_tex or is_py or is_js or is_css or 
                                  is_yaml or is_ini or is_log or is_asciidoc)
                
                if text_based_files:
                    if is_xml:
                        file_type = "xml"
                        processor_name = "direct_xml_reader"
                    elif is_vtt:
                        file_type = "vtt"
                        processor_name = "direct_vtt_reader"
                    elif is_md:
                        file_type = "md"
                        processor_name = "direct_md_reader"
                    elif is_rst:
                        file_type = "rst"
                        processor_name = "direct_rst_reader"
                    elif is_rtf:
                        file_type = "rtf"
                        processor_name = "direct_rtf_reader"
                    elif is_tex:
                        file_type = "tex"
                        processor_name = "direct_tex_reader"
                    elif is_py:
                        file_type = "py"
                        processor_name = "direct_py_reader"
                    elif is_js:
                        file_type = "js"
                        processor_name = "direct_js_reader"
                    elif is_css:
                        file_type = "css"
                        processor_name = "direct_css_reader"
                    elif is_yaml:
                        file_type = "yaml"
                        processor_name = "direct_yaml_reader"
                    elif is_ini:
                        file_type = "ini"
                        processor_name = "direct_ini_reader"
                    elif is_log:
                        file_type = "log"
                        processor_name = "direct_log_reader"
                    elif is_asciidoc:
                        file_type = "asciidoc"
                        processor_name = "direct_asciidoc_reader"
                    else:
                        file_type = "txt"
                        processor_name = "direct_txt_reader"
                    
                    logger.info(f"Processing {file_type.upper()} file directly: {str_filepath}")
                    try:
                        # Try multiple encodings to handle files with different character sets
                        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                        text_content = None
                        encoding_used = None
                        
                        for encoding in encodings_to_try:
                            try:
                                with open(str_filepath, 'r', encoding=encoding) as f:
                                    text_content = f.read()
                                encoding_used = encoding
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if text_content is None:
                            logger.error(f"Could not decode {file_type.upper()} file {str_filepath} with any encoding")
                            continue
                        
                        # Create a Haystack Document directly from the text content
                        doc = Document(
                            content=text_content,
                            meta={
                                "source": str_filepath,
                                "file_path": str_filepath,
                                "filename": Path(str_filepath).name,
                                "file_type": file_type,
                                "processed_by": processor_name,
                                "encoding_used": encoding_used
                            }
                        )
                        documents.append(doc)
                        logger.info(f"Successfully processed {file_type.upper()} file: {str_filepath} ({len(text_content)} chars, encoding: {encoding_used})")
                        continue
                    except Exception as txt_error:
                        logger.error(f"Error reading {file_type.upper()} file {str_filepath}: {txt_error}")
                        continue
                
                elif not ext.endswith(docling_supported_exts):
                    logger.warning(f"Filetype {ext} not supported by Docling. Skipping: {str_filepath}")
                    continue

                # Initialize metadata
                ocr_meta = {
                    "ocr_quality_checked": False,
                    "ocr_reprocessed": False,
                }
                ocr_meta_update = {}

                # PDF logic (OCR and quality checking)
                if self.force_ocr and is_pdf:
                    logger.info(f"Forcing OCR processing for PDF: {str_filepath}")
                    ocr_meta["ocr_quality_checked"] = False
                    ocr_meta["ocr_reprocessed"] = True
                    dl_doc, ocr_meta_update = self._process_pdf_with_ocr_check(str_filepath)
                elif self.enable_ocr_check and is_pdf:
                    logger.info(f"Processing PDF with OCR quality check: {str_filepath}")
                    ocr_meta["ocr_quality_checked"] = True
                    dl_doc, ocr_meta_update = self._process_pdf_with_ocr_check(str_filepath)
                else:
                    # For all other docling-supported filetypes, use standard conversion
                    dl_doc = self._process_document_standard(str_filepath)
                ocr_meta.update(ocr_meta_update)

                # Log docling doc info
                if dl_doc:
                    if hasattr(dl_doc, 'id'):
                        logger.info(f"  dl_doc.id: {dl_doc.id}")
                    try:
                        text_content = dl_doc.export_to_text()
                        logger.info(f"  dl_doc content length: {len(text_content)}")
                    except Exception as e:
                        logger.warning(f"  Could not get content length: {e}")
                else:
                    logger.info("  dl_doc is None or evaluates to False")
                if not dl_doc:
                    logger.warning(f"Null document returned for {str_filepath}, skipping")
                    continue

                # --- Adaptive chunking by filetype ---
                chunker, chunker_meta = self._select_chunker_for_filetype(str_filepath)
                logger.info(f"Using {chunker_meta['chunker_type']} for {str_filepath}")

                if self._export_type == ExportType.DOC_CHUNKS:
                    try:
                        chunk_iter = chunker.chunk(dl_doc=dl_doc)
                        chunk_count = 0
                        for chunk in chunk_iter:
                            chunk_meta = self._meta_extractor.extract_chunk_meta(chunk=chunk)
                            chunk_meta.update(ocr_meta)
                            chunk_meta["file_path"] = str_filepath
                            # Add chunker metadata to each chunk
                            chunk_meta["chunker"] = chunker_meta
                            document = Document(
                                content=chunker.serialize(chunk=chunk),
                                meta=chunk_meta,
                            )
                            logger.debug(f"Document metadata for chunk from {str_filepath}: {document.meta}")
                            if 'ocr_engine_used' in chunk_meta:
                                logger.info(f"OCR engine used and stored in metadata: {chunk_meta['ocr_engine_used']}")
                            documents.append(document)
                            chunk_count += 1
                        if chunk_count == 0:
                            logger.warning(f"No chunks extracted from {str_filepath}")
                    except Exception as e:
                        self.stats["errors"] += 1
                        logger.error(f"Error chunking document {str_filepath}: {e}")
                        logger.error(traceback.format_exc())
                elif self._export_type == ExportType.MARKDOWN:
                    try:
                        doc_meta = self._meta_extractor.extract_dl_doc_meta(dl_doc=dl_doc)
                        doc_meta.update(ocr_meta)
                        doc_meta["file_path"] = str_filepath
                        # Add chunker metadata to the document
                        _, chunker_meta = self._select_chunker_for_filetype(str_filepath)
                        doc_meta["chunker"] = chunker_meta
                        document = Document(
                            content=dl_doc.export_to_markdown(**self._md_export_kwargs),
                            meta=doc_meta,
                        )
                        documents.append(document)
                    except Exception as e:
                        self.stats["errors"] += 1
                        logger.error(f"Error exporting document {str_filepath} to markdown: {e}")
                        logger.error(traceback.format_exc())
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Error processing {str_filepath}: {e}")
                logger.error(traceback.format_exc())
        logger.info(f"EnhancedDoclingConverter stats: {self.stats}")
        return {"documents": documents}
