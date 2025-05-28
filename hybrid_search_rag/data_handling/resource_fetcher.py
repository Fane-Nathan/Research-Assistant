# -*- coding: utf-8 -*-
"""
Fetches and processes data from arXiv and web sources (HTML/PDF) asynchronously.
Refactored for improved modularity and readability.
Uses Playwright to handle dynamic websites and aiohttp for direct PDF downloads.
Processes discovered PDF links in parallel tasks.
Includes safe unpacking for link extraction results.
"""

import asyncio
import io
import logging
import os
import re
import sys
import time # Keep time import
import traceback
from datetime import datetime, timedelta # Ensure timedelta is imported
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse # Added for title extraction

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Error as PlaywrightSyncError # Import PlaywrightSyncError
from playwright.async_api import async_playwright, Error as PlaywrightAsyncError 

# Imports that were missing or causing issues
import aiohttp 
import ssl
import xml.etree.ElementTree as ET

# --- Module-Level Constants ---
CONTENT_TYPE_PDF = "pdf"
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_ARXIV = "arxiv" 
CONTENT_TYPE_FAILED = "failed"
CONTENT_TYPE_OTHER = "other"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 StudyAssistantBot/1.0"
MAX_PAGES_TO_CRAWL_CONFIG = 100 # For crawl_and_fetch_web_articles (even if stubbed)

# --- Module-Level Logger ---
_module_logger = logging.getLogger(__name__)

# --- Check Playwright Availability (for async part, primarily) ---
PLAYWRIGHT_AVAILABLE = False
try:
    # async_playwright and PlaywrightAsyncError are already imported
    PLAYWRIGHT_AVAILABLE = True
    _module_logger.info("Async Playwright API is available.")
except ImportError:
    _module_logger.warning("Playwright async_api not found. Web crawling features requiring it will be limited.")
    # PlaywrightAsyncError is aliased, so it's defined.

# --- Helper function for User Agent (if needed by async part) ---
def get_user_agent() -> str: 
    return DEFAULT_USER_AGENT

# --- Module-Level Helper Functions (Defined ONCE) ---
def _clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        # _module_logger.debug("Helper_CleanText: Input is None.") # Debug, can be noisy
        return None
    try:
        text_no_extra_spaces = re.sub(r'[ \\t]+', ' ', text)
        lines = text_no_extra_spaces.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        if not cleaned_lines:
            return None
        text_joined_lines = "\\n".join(cleaned_lines)
        text_no_extra_newlines = re.sub(r'\\n{3,}', '\\n\\n', text_joined_lines)
        cleaned_text = text_no_extra_newlines.strip()
        return cleaned_text if cleaned_text else None
    except Exception as e:
        _module_logger.error(f"Helper_CleanText: Error cleaning text: {e}", exc_info=False) # exc_info=False for less verbose error
        return text # Return original on error to preserve some data

def _parse_pdf_content(pdf_bytes: bytes, source_url: str) -> Optional[str]:
    _module_logger.debug(f"Helper_ParsePDF: Attempting to parse PDF from {source_url} ({len(pdf_bytes)} bytes)")
    pdf_text_parts: List[str] = []
    try:
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            if doc.is_encrypted and not doc.authenticate(""):
                _module_logger.warning(f"Helper_ParsePDF: PDF is encrypted and cannot be opened: {source_url}")
                return None
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # page.get_text("text") is the correct PyMuPDF usage.
                # Linter errors about this are likely due to incomplete stubs for fitz/PyMuPDF.
                page_text = page.get_text("text") 
                if page_text:
                    pdf_text_parts.append(page_text.strip())
        
        if not pdf_text_parts:
            _module_logger.warning(f"Helper_ParsePDF: No text parts extracted from PDF pages: {source_url}")
            return None

        full_pdf_text = "\\n\\n".join(filter(None, pdf_text_parts))
        cleaned_text = _clean_text(full_pdf_text)
        if not cleaned_text:
            _module_logger.warning(f"Helper_ParsePDF: No text extracted from PDF after cleaning: {source_url}")
        else:
            _module_logger.info(f"Helper_ParsePDF: Successfully parsed PDF {source_url}, extracted text length: {len(cleaned_text)}")
        return cleaned_text
    except Exception as e: # Catching fitz specific errors might be good too if known
        _module_logger.error(f"Helper_ParsePDF: PDF processing failed for {source_url}: {e}", exc_info=True)
        return None

# --- ResourceFetcher Class Definition (Defined ONCE) ---
class ResourceFetcher:
    def __init__(self, timeout=10, cache_dir=None, rate_limiter=None, use_playwright_for_pdfs=True):
        self._configure_logging() 
        self.logger.info(f"ResourceFetcher.__init__ - Playwright for PDFs: {use_playwright_for_pdfs}, Timeout: {timeout}")

        self.timeout = timeout # General timeout for operations if not overridden
        self.cache_dir = cache_dir 
        self.rate_limiter = rate_limiter 

        self.sync_session = None 
        self._init_sync_session()

        self.use_playwright_for_pdfs = use_playwright_for_pdfs
        self.playwright_sync_instance = None
        self.playwright_browser = None
        self.playwright_context = None 
        self._playwright_init_attempted = False

        if self.use_playwright_for_pdfs:
            # Check if we're in an async context (event loop running)
            try:
                import asyncio
                try:
                    asyncio.get_running_loop()
                    # We're in an async context, defer Playwright initialization
                    self.logger.info("ResourceFetcher.__init__ - Detected async context. Deferring Playwright initialization until needed.")
                except RuntimeError:
                    # No event loop running, safe to initialize sync Playwright now
                    self._init_playwright_sync()
            except ImportError:
                # asyncio not available, proceed with sync initialization
                self._init_playwright_sync()
        else:
            self.logger.info("ResourceFetcher.__init__ - Synchronous Playwright for PDF fetching is disabled.")
        
    def _configure_logging(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers and not logging.getLogger(__name__).handlers and not logging.getLogger().handlers:
            handler = logging.StreamHandler(sys.stdout) 
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(os.environ.get("LOG_LEVEL_RF_CLASS", "INFO").upper()) 
            self.logger.propagate = False

    def _init_sync_session(self):
        self.logger.debug("ResourceFetcher._init_sync_session - Placeholder for 'requests' or other sync http client. Not implemented.")
        pass

    def _init_playwright_sync(self):
        """Initialize synchronous Playwright instance, browser, and context."""
        if self._playwright_init_attempted:
            return
        
        self._playwright_init_attempted = True
        self.logger.info("ResourceFetcher._init_playwright_sync - Attempting to initialize synchronous Playwright...")
        try:
            self.playwright_sync_instance = sync_playwright().start()
            self.logger.info("ResourceFetcher._init_playwright_sync - Sync Playwright instance started.")
            self.playwright_browser = self.playwright_sync_instance.chromium.launch(headless=True)
            self.logger.info("ResourceFetcher._init_playwright_sync - Sync Playwright browser (Chromium) launched.")
            self.playwright_context = self.playwright_browser.new_context(
                user_agent=DEFAULT_USER_AGENT, 
                accept_downloads=True 
            )
            self.logger.info("ResourceFetcher._init_playwright_sync - Sync Playwright context created successfully.")
        except PlaywrightSyncError as pse: # Catch specific Playwright error
            self.logger.error(f"ResourceFetcher._init_playwright_sync - Failed to initialize synchronous Playwright (Playwright Error): {pse}", exc_info=True)
            self._cleanup_playwright_sync() 
        except Exception as e:
            self.logger.error(f"ResourceFetcher._init_playwright_sync - Failed to initialize synchronous Playwright (General Error): {e}", exc_info=True)
            self._cleanup_playwright_sync()

    def _cleanup_playwright_sync(self):
        self.logger.debug("ResourceFetcher._cleanup_playwright_sync - Cleaning up synchronous Playwright resources...")
        if self.playwright_context:
            try: 
                self.playwright_context.close()
                self.logger.debug("ResourceFetcher._cleanup_playwright_sync - Playwright context closed.")
            except Exception as ec: self.logger.error(f"Error closing sync Playwright context: {ec}", exc_info=True)
        if self.playwright_browser:
            try: 
                self.playwright_browser.close()
                self.logger.debug("ResourceFetcher._cleanup_playwright_sync - Playwright browser closed.")
            except Exception as eb: self.logger.error(f"Error closing sync Playwright browser: {eb}", exc_info=True)
        if self.playwright_sync_instance:
            try: 
                self.playwright_sync_instance.stop()
                self.logger.debug("ResourceFetcher._cleanup_playwright_sync - Playwright instance stopped.")
            except Exception as es: self.logger.error(f"Error stopping sync Playwright instance: {es}", exc_info=True)
        
        self.playwright_sync_instance = None
        self.playwright_browser = None
        self.playwright_context = None
        self._playwright_init_attempted = False
        self.logger.info("ResourceFetcher._cleanup_playwright_sync - Synchronous Playwright resources have been reset/cleaned up.")

    def _fetch_pdf_content_with_playwright_sync(self, pdf_url: str, verbose: bool = False) -> Optional[str]:
        # Try to initialize Playwright if not already done and we're allowed to use it
        if self.use_playwright_for_pdfs and not self.playwright_context and not self._playwright_init_attempted:
            try:
                import asyncio
                try:
                    asyncio.get_running_loop()
                    # We're in an async context, don't try to initialize sync Playwright
                    self.logger.warning(f"Cannot initialize sync Playwright in async context. Skipping PDF fetch with Playwright: {pdf_url}")
                    return None
                except RuntimeError:
                    # No event loop running, safe to initialize sync Playwright now
                    self._init_playwright_sync()
            except ImportError:
                # asyncio not available, proceed with sync initialization
                self._init_playwright_sync()
        
        if not self.playwright_context:
            self.logger.warning(f"Sync Playwright context not available. Cannot fetch PDF: {pdf_url}")
            return None
        
        log_prefix = "RFetcher._fetch_pdf_sync"
        if verbose: self.logger.info(f"{log_prefix} - Attempting PDF: {pdf_url}")
        
        page = None
        try:
            page = self.playwright_context.new_page()
            if verbose: self.logger.debug(f"{log_prefix} - New page created for {pdf_url}")
            
            response = page.goto(pdf_url, timeout=90000, wait_until="commit") 

            if not response:
                self.logger.error(f"{log_prefix} - Playwright navigation to {pdf_url} returned no response object.")
                return None
            
            final_url = response.url
            self.logger.info(f"{log_prefix} - Response from {final_url} (requested {pdf_url}): Status {response.status}")

            if response.status < 200 or response.status >= 300:
                self.logger.error(f"{log_prefix} - URL {final_url} returned status {response.status}.")
                return None

            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type and "octet-stream" not in content_type:
                 self.logger.warning(f"{log_prefix} - URL {final_url} has Content-Type: '{content_type}'. Attempting to process as PDF.")
            
            if verbose: self.logger.debug(f"{log_prefix} - Navigated successfully. Status: {response.status}. URL: {final_url}")
            
            pdf_bytes = response.body() 
            if not pdf_bytes:
                self.logger.warning(f"{log_prefix} - No PDF content bytes retrieved from {final_url}")
                return None
            
            if verbose: self.logger.info(f"{log_prefix} - Got {len(pdf_bytes)} PDF bytes for {final_url}. Parsing...")
            
            return _parse_pdf_content(pdf_bytes, final_url)

        except PlaywrightSyncError as pwe: 
            self.logger.error(f"{log_prefix} - Playwright error for {pdf_url}: {pwe}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"{log_prefix} - General error for {pdf_url}: {e}", exc_info=True)
            return None
        finally:
            if page:
                try:
                    page.close()
                    if verbose: self.logger.debug(f"{log_prefix} - Closed Playwright page for {pdf_url}")
                except Exception as close_err:
                    self.logger.warning(f"{log_prefix} - Error closing Playwright page for {pdf_url}: {close_err}", exc_info=False)

    async def fetch_document(self, url: str, source: str = "web", force_playwright_html: bool = False, is_arxiv_pdf_link: bool = False) -> Optional[Dict[str, Any]]:
        log_prefix = "RFetcher.fetch_document"
        self.logger.debug(f"{log_prefix} - URL: {url}, Source: {source}, ArXivPDF: {is_arxiv_pdf_link}, ForcePW_HTML: {force_playwright_html}")

        if is_arxiv_pdf_link and self.use_playwright_for_pdfs and self.playwright_context:
            self.logger.info(f"{log_prefix} - Attempting arXiv PDF {url} via sync Playwright in thread.")
            text_content = None
            try:
                text_content = await asyncio.to_thread(self._fetch_pdf_content_with_playwright_sync, url, verbose=True)
            except Exception as e_thread: 
                self.logger.error(f"{log_prefix} - Error in to_thread for sync PDF fetch {url}: {e_thread}", exc_info=True)
            
            if text_content:
                self.logger.info(f"{log_prefix} - Successfully fetched/parsed PDF {url} (sync Playwright).")
                title = os.path.basename(urlparse(url).path) or url.split("/")[-1]
                return {
                    "url": url, "source": source, "content_type": CONTENT_TYPE_PDF,
                    "text": text_content, "title": title,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                self.logger.warning(f"{log_prefix} - Failed to fetch PDF {url} with sync Playwright.")
                return None
        
        self.logger.debug(f"{log_prefix} - Proceeding with placeholder async fetch for {url} (or not an arXiv PDF for sync PW).")
        parsed_info = await self._handle_async_fetching(url, source, force_playwright_html, is_arxiv_pdf_link)

        if parsed_info and parsed_info.get("text"):
            self.logger.info(f"{log_prefix} - Successfully fetched content for {url} via (placeholder) async path.")
            title = parsed_info.get("title", os.path.basename(urlparse(url).path) or url.split("/")[-1])
            return {
                "url": parsed_info.get("final_url", url),
                "source": source, 
                "content_type": parsed_info.get("content_type"),
                "text": parsed_info["text"],
                "title": title,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        self.logger.warning(f"{log_prefix} - No content extracted for {url} by fetch_document.")
        return None

    async def _handle_async_fetching(self, url: str, source: str, force_playwright_html: bool, is_arxiv_pdf_link: bool) -> Optional[Dict[str, Any]]:
        log_prefix = "RFetcher._handle_async_fetching"
        self.logger.debug(f"{log_prefix} - Starting async fetching of {url}.")
        
        # For arXiv PDFs, use aiohttp for direct download
        if is_arxiv_pdf_link or url.endswith('.pdf'):
            return await self._fetch_pdf_with_aiohttp(url, source)
        
        # For other content, this remains a placeholder
        self.logger.warning(f"{log_prefix} - Async fetching for non-PDF {url} is not implemented.")
        return None

    async def _fetch_pdf_with_aiohttp(self, pdf_url: str, source: str = "web") -> Optional[Dict[str, Any]]:
        """Fetch PDF content using aiohttp and parse with PyMuPDF."""
        log_prefix = "RFetcher._fetch_pdf_aiohttp"
        self.logger.info(f"{log_prefix} - Fetching PDF via aiohttp: {pdf_url}")
        
        try:
            # Create SSL context that's less strict for academic sites
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': DEFAULT_USER_AGENT}
            ) as session:
                self.logger.debug(f"{log_prefix} - Making HTTP request to {pdf_url}")
                
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        self.logger.error(f"{log_prefix} - HTTP {response.status} for {pdf_url}")
                        return None
                    
                    content_type = response.headers.get('content-type', '').lower()
                    if 'application/pdf' not in content_type and 'octet-stream' not in content_type:
                        self.logger.warning(f"{log_prefix} - Unexpected content-type '{content_type}' for {pdf_url}")
                    
                    pdf_bytes = await response.read()
                    self.logger.info(f"{log_prefix} - Downloaded {len(pdf_bytes)} bytes from {pdf_url}")
                    
                    if not pdf_bytes:
                        self.logger.warning(f"{log_prefix} - No content received from {pdf_url}")
                        return None
                    
                    # Parse PDF content
                    text_content = _parse_pdf_content(pdf_bytes, pdf_url)
                    
                    if text_content:
                        self.logger.info(f"{log_prefix} - Successfully parsed PDF, extracted {len(text_content)} characters")
                        title = os.path.basename(urlparse(pdf_url).path) or pdf_url.split("/")[-1]
                        return {
                            "url": pdf_url,
                            "source": source,
                            "content_type": CONTENT_TYPE_PDF,
                            "text": text_content,
                            "title": title,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        self.logger.warning(f"{log_prefix} - Failed to extract text from PDF {pdf_url}")
                        return None
                        
        except asyncio.TimeoutError:
            self.logger.error(f"{log_prefix} - Timeout while fetching {pdf_url}")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"{log_prefix} - aiohttp error for {pdf_url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"{log_prefix} - Unexpected error for {pdf_url}: {e}", exc_info=True)
            return None 

    def _parse_html_content(self, html_string: str, source_url: str) -> Tuple[Optional[str], Optional[str]]:
        log_prefix = "RFetcher._parse_html_content"
        self.logger.debug(f"{log_prefix} - Parsing HTML for {source_url} (len: {len(html_string)})")
        text_content: Optional[str] = None
        title_text: Optional[str] = None
        try:
            soup = BeautifulSoup(html_string, 'html.parser')
            
            title_tag = soup.find('title')
            if title_tag:
                # Use get_text() for safer title extraction from BeautifulSoup tag
                title_text = _clean_text(title_tag.get_text(strip=True))
            
            body_tag = soup.find('body')
            if body_tag:
                raw_body_text = body_tag.get_text(separator='\\n', strip=True)
                text_content = _clean_text(raw_body_text)
            else: 
                raw_html_text = soup.get_text(separator='\\n', strip=True)
                text_content = _clean_text(raw_html_text)

            if text_content:
                self.logger.debug(f"{log_prefix} - Extracted text (len: {len(text_content)}) and title ('{title_text}') from {source_url}")
            else:
                self.logger.warning(f"{log_prefix} - Could not extract text from {source_url} using basic parsing.")
            
            return text_content, title_text
        except Exception as e:
            self.logger.error(f"{log_prefix} - HTML parsing failed for {source_url}: {e}", exc_info=True)
            return None, None

    def close(self):
        self.logger.info("ResourceFetcher.close - Attempting to close resources...")
        self._cleanup_playwright_sync() 
        self.logger.info("ResourceFetcher.close - Resource closing process completed.")

# --- arXiv Fetcher (Focus on metadata, PDF fetching delegated) ---
async def fetch_arxiv_papers(
    query: str, max_results: int = 10, days_back: Optional[int] = None,
    sort_by: str = "relevance", 
    aiohttp_session_param: Optional[aiohttp.ClientSession] = None, 
    proxy: Optional[str] = None, 
    verbose: bool = False
) -> List[Dict[str, Any]]:
    base_url = "http://export.arxiv.org/api/query"
    logger_to_use = _module_logger 
    if verbose: logger_to_use.info(f"arXiv Fetch - Query: '{query}', MaxResults: {max_results}")
    
    search_query_parts = [query]
    if days_back is not None:
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            search_query_parts.append(f"submittedDate:[{cutoff_date.strftime('%Y%m%d')}0000 TO {datetime.now().strftime('%Y%m%d')}2359]")
            if verbose: logger_to_use.info(f"arXiv Fetch - Date filter: last {days_back} days.")
        except Exception as date_e: 
            logger_to_use.warning(f"arXiv Fetch - Failed to create date filter: {date_e}")
    
    search_query = " AND ".join(search_query_parts)

    sort_order_val = "descending"
    sort_by_val = "relevance" # Default
    if sort_by == "lastUpdatedDate": sort_by_val = "lastUpdatedDate"
    elif sort_by == "submittedDate": sort_by_val = "submittedDate"

    params = {'search_query': search_query, 'start': 0, 'max_results': max_results,
              'sortBy': sort_by_val, 'sortOrder': sort_order_val}

    session_created_internally = False
    current_session: Optional[aiohttp.ClientSession] = aiohttp_session_param
    if current_session is None:
        ssl_ctx_arxiv = ssl.create_default_context()
        ssl_ctx_arxiv.check_hostname = False
        ssl_ctx_arxiv.verify_mode = ssl.CERT_NONE
        connector_arxiv = aiohttp.TCPConnector(ssl=ssl_ctx_arxiv, enable_cleanup_closed=True) 
        current_session = aiohttp.ClientSession(connector=connector_arxiv, timeout=aiohttp.ClientTimeout(total=180, connect=30, sock_read=120))
        session_created_internally = True
        if verbose: logger_to_use.debug("arXiv Fetch - Created new aiohttp.ClientSession.")

    papers_list = []
    
    # Helper to safely get text from XML element, defined at a scope accessible by the loop
    def get_element_text(element: Optional[ET.Element]) -> str:
        if element is not None and isinstance(element.text, str):
            return element.text.strip()
        return ""

    try:
        if not current_session:
            logger_to_use.error("arXiv Fetch - aiohttp.ClientSession is None. Aborting.")
            return []

        if verbose: logger_to_use.debug(f"arXiv Fetch - API request to {base_url} with params {params}")
        request_kwargs: Dict[str,Any] = {}
        if proxy: request_kwargs['proxy'] = proxy

        async with current_session.get(base_url, params=params, **request_kwargs) as response: 
            response.raise_for_status() 
            xml_content = await response.text()
            if verbose: logger_to_use.info(f"arXiv Fetch - API response OK ({len(xml_content)} chars). Parsing XML...")
            
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'} 
            entries = root.findall('atom:entry', ns)
            logger_to_use.info(f"arXiv Fetch - Found {len(entries)} XML entries for query '{query}'. Max results: {max_results}")

            for i, entry in enumerate(entries):
                if len(papers_list) >= max_results:
                    logger_to_use.info(f"arXiv Fetch - Max results ({max_results}) reached. Stopping entry processing.")
                    break
                
                # The try-except block is now correctly indented for each entry processing
                try:
                    title_elem = entry.find('atom:title', ns)
                    title = get_element_text(title_elem) or "N/A"

                    summary_elem = entry.find('atom:summary', ns)
                    summary = get_element_text(summary_elem)

                    published_elem = entry.find('atom:published', ns)
                    published_date_str = get_element_text(published_elem)

                    updated_elem = entry.find('atom:updated', ns)
                    updated_date_str = get_element_text(updated_elem)

                    id_elem = entry.find('atom:id', ns)
                    entry_id_val = get_element_text(id_elem)

                    pdf_url = ""
                    for link_tag in entry.findall('atom:link', ns):
                        if link_tag.get('title') == 'pdf' and link_tag.get('href'):
                            pdf_url = link_tag.get('href')
                            break 
                    
                    if not pdf_url and entry_id_val:
                        arxiv_id_match = re.search(r'arxiv.org/(?:abs|pdf)/([^vV/?#]+)', entry_id_val)
                        arxiv_id = ""
                        if arxiv_id_match:
                            arxiv_id = arxiv_id_match.group(1)
                            arxiv_id = re.sub(r'v\\d+$', '', arxiv_id) 
                        if arxiv_id:
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        else:
                            parsed_entry_id = urlparse(entry_id_val)
                            if parsed_entry_id.path.startswith("/abs/"):
                                potential_id = parsed_entry_id.path.replace("/abs/", "")
                                potential_id = re.sub(r'v\\d+$', '', potential_id)
                                if potential_id:
                                     pdf_url = f"https://arxiv.org/pdf/{potential_id}.pdf"

                    if not pdf_url:
                         logger_to_use.warning(f"arXiv Fetch - Could not determine PDF URL for entry: {title[:50]} (ID: {entry_id_val})")

                    authors = []
                    for author_tag in entry.findall('atom:author', ns):
                        name_elem = author_tag.find('atom:name', ns)
                        author_name = get_element_text(name_elem)
                        if author_name: authors.append(author_name)
                    
                    categories = [cat.get('term') for cat in entry.findall('atom:category', ns) if cat.get('term')]

                    paper_data = {
                        'title': _clean_text(title),
                        'authors': authors,
                        'published_date': published_date_str.split('T')[0] if 'T' in published_date_str else published_date_str,
                        'updated_date': updated_date_str.split('T')[0] if 'T' in updated_date_str else updated_date_str,
                        'entry_id': entry_id_val, 
                        'pdf_url': pdf_url, 
                        'url': entry_id_val, 
                        'summary': _clean_text(summary.replace('\\n', ' ')),
                        'categories': categories, 
                        'source': CONTENT_TYPE_ARXIV,
                        'text_content': None, 
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    papers_list.append(paper_data)
                    if verbose: logger_to_use.debug(f"arXiv Fetch - Added metadata for '{title[:30]}...'. PDF URL: {pdf_url}")

                except Exception as entry_e: 
                    title_for_log_elem = entry.find('atom:title', ns)
                    # get_element_text is now in scope here
                    title_for_log = get_element_text(title_for_log_elem) or "Unknown Title" 
                    logger_to_use.warning(f"arXiv Fetch - Failed to parse XML entry '{title_for_log}': {entry_e}", exc_info=True)

            if verbose: logger_to_use.info(f"arXiv Fetch - Processed {len(papers_list)} papers from XML.")
    
    except aiohttp.ClientResponseError as http_err:
        logger_to_use.error(f"arXiv Fetch - HTTP error: {http_err.status} {http_err.message}. URL: {http_err.request_info.url}", exc_info=True)
    except ET.ParseError as xml_err:
        logger_to_use.error(f"arXiv Fetch - Error parsing XML: {xml_err}", exc_info=True)
    except Exception as e: 
        logger_to_use.error(f"arXiv Fetch - General error: {e}", exc_info=True)
    finally:
        if session_created_internally and current_session:
            await current_session.close()
            if verbose: logger_to_use.debug("arXiv Fetch - Closed internally created aiohttp.ClientSession.")

    logger_to_use.info(f"arXiv Fetch - Returning {len(papers_list)} items of metadata.")
    return papers_list

# --- Web Crawler (Stubbed out) ---
async def crawl_and_fetch_web_articles(
    start_urls: List[str], 
    process_pdfs_linked: bool = True, 
    max_pages_override: Optional[int] = None, 
    proxies: Optional[List[str]] = None,
    max_concurrent_html: int = 3, 
    max_concurrent_pdf: int = 5
    ) -> List[Dict[str, Any]]:
    
    logger_to_use = _module_logger
    logger_to_use.warning("crawl_and_fetch_web_articles is currently STUBBED and will not perform crawling or fetching.")
    
    if not PLAYWRIGHT_AVAILABLE: 
        logger_to_use.error("crawl_and_fetch_web_articles - Playwright (async_api) is not available. Cannot proceed.")
        return []
    if not start_urls:
        logger_to_use.info("crawl_and_fetch_web_articles - No starting URLs provided.")
        return []
    return [] # Stubbed response

# --- Example Usage (Commented out for use as a module) ---
# async def main_example():
#     # Setup basic logging for the example
#     log_level_str = os.environ.get("LOG_LEVEL_RF_MAIN", "DEBUG").upper()
#     logging.basicConfig(
#         level=log_level_str,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         stream=sys.stdout
#     )
#     _module_logger.info(f"Running resource_fetcher.py standalone example with log level: {log_level_str}")

#     # Initialize ResourceFetcher - enable Playwright for PDFs
#     fetcher = ResourceFetcher(use_playwright_for_pdfs=True)
#     try:
#         # 1. Fetch arXiv metadata
#         arxiv_query = "quantum computing" # "Retrieval-Augmented Generation"
#         _module_logger.info(f"--- Fetching arXiv metadata for query: '{arxiv_query}' ---")
#         async with aiohttp.ClientSession() as http_session:
#             arxiv_metadata_list = await fetch_arxiv_papers(
#                 arxiv_query, 
#                 max_results=2, 
#                 verbose=True, 
#                 aiohttp_session_param=http_session
#             )
        
#         _module_logger.info(f"--- Found {len(arxiv_metadata_list)} arXiv papers. Now fetching PDF content if available ---")

#         # 2. For each paper with a PDF URL, use ResourceFetcher to get its content
#         for paper_meta in arxiv_metadata_list:
#             if paper_meta.get('pdf_url'):
#                 _module_logger.info(f"Fetching PDF for: '{paper_meta['title']}' from {paper_meta['pdf_url']}")
#                 pdf_document_data = await fetcher.fetch_document(
#                     paper_meta['pdf_url'], 
#                     source=CONTENT_TYPE_ARXIV, 
#                     is_arxiv_pdf_link=True
#                 )
#                 if pdf_document_data and pdf_document_data.get("text"):
#                     paper_meta['text_content'] = pdf_document_data['text'] 
#                     _module_logger.info(f"  -> PDF content fetched for '{paper_meta['title']}'. Length: {len(pdf_document_data['text'])}")
#                 else:
#                     _module_logger.warning(f"  -> Failed to fetch PDF content for '{paper_meta['title']}' from {paper_meta['pdf_url']}")
#             else:
#                 _module_logger.warning(f"No PDF URL for paper: '{paper_meta['title']}' (Entry ID: {paper_meta.get('entry_id')})")
#         _module_logger.info("--- Finished processing arXiv papers ---")

#     except Exception as e:
#         _module_logger.error(f"An error occurred in main_example: {e}", exc_info=True)
#     finally:
#         await fetcher.close()

# if __name__ == '__main__':
#     asyncio.run(main_example())