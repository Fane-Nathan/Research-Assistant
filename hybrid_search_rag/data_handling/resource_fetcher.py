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
import random
import re
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Type, Union # Added Union
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
import fitz
import ssl
from bs4 import BeautifulSoup, element as bs4_element
from cachetools import LRUCache
from pybloom_live import BloomFilter
from trafilatura import extract
from trafilatura.settings import use_config
import xml.etree.ElementTree as ET # Added for XML parsing

# --- Playwright Import and Dummy Definitions ---

# Define dummy/fallback exception classes with distinct internal names.
class _InternalDummyPlaywrightError(Exception): pass
class _InternalDummyPlaywrightTimeoutError(Exception): pass

# These are the public names that will be used in `except` blocks.
# They are variables that will hold an exception *class*.
# Initialize them with the dummy classes and set PLAYWRIGHT_AVAILABLE.
PlaywrightError: Type[Exception] = _InternalDummyPlaywrightError
PlaywrightTimeoutError: Type[Exception] = _InternalDummyPlaywrightTimeoutError
PLAYWRIGHT_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    # Import actual Playwright errors
    from playwright.async_api import TimeoutError as _ActualPlaywrightTimeoutError_Imported
    from playwright.async_api import Error as _ActualPlaywrightError_Imported

    # Re-assign the module-level variables to the actual Playwright classes
    PlaywrightTimeoutError = _ActualPlaywrightTimeoutError_Imported
    PlaywrightError = _ActualPlaywrightError_Imported
    PLAYWRIGHT_AVAILABLE = True # Mark Playwright as available
except ImportError:
    logging.getLogger(__name__).critical("Playwright not found. Please install it: pip install playwright && playwright install")
    # PLAYWRIGHT_AVAILABLE remains False.
    # PlaywrightError and PlaywrightTimeoutError retain their initial assignments
    # to _InternalDummyPlaywrightError and _InternalDummyPlaywrightTimeoutError.

    # --- Dummy Playwright classes ---
    # These classes mimic the structure of Playwright objects for when Playwright is not installed.

    class _DummyPage:
        def __init__(self, owning_context: Any):
            self.url: str = "http://dummy.invalid/page"
            self._headers: Dict[str, str] = {}
            self._owning_context = owning_context # Unused for now, but good for structure

        async def close(self, *args: Any, **kwargs: Any) -> None: pass
        
        async def content(self, *args: Any, **kwargs: Any) -> str:
            return f"<html><head><title>Dummy Page</title></head><body>Dummy Content for {self.url}</body></html>"
        
        async def goto(self, url: str, *args: Any, **kwargs: Any) -> Any:
            self.url = url
            class DummyResponse: # Mimic Playwright Response object
                def __init__(self, page_url: str, page_headers: Dict[str,str]):
                    self.status = 200
                    # Ensure content-type is present, as the main code checks it
                    self.headers: Dict[str, str] = {**page_headers, "content-type": "text/html"}
                    self.url: str = page_url
            return DummyResponse(self.url, self._headers)

        async def wait_for_selector(self, selector: str, *args: Any, **kwargs: Any) -> None: pass
        async def wait_for_load_state(self, state: str = 'load', *args: Any, **kwargs: Any) -> None: pass
        async def set_extra_http_headers(self, headers: Dict[str, str]) -> None: self._headers.update(headers)

    class _DummyContext:
        def __init__(self, owning_browser: Any):
            self._owning_browser = owning_browser

        async def new_page(self, *args: Any, **kwargs: Any) -> _DummyPage:
            return _DummyPage(owning_context=self)
        async def close(self, *args: Any, **kwargs: Any) -> None: pass
        # Add set_extra_http_headers if it were called on context, currently called on page
        # async def set_extra_http_headers(self, headers: Dict[str, str]) -> None: pass


    class _DummyBrowser:
        def __init__(self, owning_chromium: Any):
            self._owning_chromium = owning_chromium

        async def new_context(self, *args: Any, **kwargs: Any) -> _DummyContext:
            # kwargs might include user_agent, ignore_https_errors, etc.
            # These are not actively used by the dummy context/page for now,
            # but could be if more sophisticated dummy behavior is needed.
            return _DummyContext(owning_browser=self)
        async def close(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyChromium: # This is a property of _DummyPlaywrightInstance
        async def launch(self, *args: Any, **kwargs: Any) -> _DummyBrowser:
            # kwargs might include headless
            return _DummyBrowser(owning_chromium=self)

    class _DummyPlaywrightInstance: # Returned by ContextManager.start()
        def __init__(self):
            self.chromium = _DummyChromium()
        async def stop(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyPlaywrightContextManager: # Returned by async_playwright()
        def __init__(self):
            self._playwright_instance: Optional[_DummyPlaywrightInstance] = None

        async def start(self, *args: Any, **kwargs: Any) -> _DummyPlaywrightInstance:
            if self._playwright_instance is None: # Create if not existing or after stop
                self._playwright_instance = _DummyPlaywrightInstance()
            return self._playwright_instance

        async def __aenter__(self) -> _DummyPlaywrightInstance:
            return await self.start()

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self._playwright_instance:
                await self._playwright_instance.stop()
                self._playwright_instance = None # Allow re-creation on next start/aenter

    # Singleton instance of the dummy context manager
    _dummy_playwright_cm_singleton: Optional[_DummyPlaywrightContextManager] = None

    def replacement_async_playwright_func(*args: Any, **kwargs: Any) -> _DummyPlaywrightContextManager:
        """Replaces the real async_playwright function when Playwright is not available."""
        global _dummy_playwright_cm_singleton
        if _dummy_playwright_cm_singleton is None:
            _dummy_playwright_cm_singleton = _DummyPlaywrightContextManager()
        return _dummy_playwright_cm_singleton

    # Assign the dummy function to the name 'async_playwright'.
    # This ensures 'async_playwright' is a callable in both try/except branches, resolving the type error.
    async_playwright = replacement_async_playwright_func


logger = logging.getLogger(__name__)

# --- Configuration Loading ---
try:
    from .. import config
    # Playwright uses milliseconds for timeouts
    FETCH_TIMEOUT_CONFIG_MS = config.FETCH_TIMEOUT * 1000
    # aiohttp uses seconds
    AIOHTTP_FETCH_TIMEOUT_CONFIG_S = config.FETCH_TIMEOUT
    AIOHTTP_HEAD_TIMEOUT_CONFIG_S = config.HEAD_TIMEOUT

    MAX_PAGES_TO_CRAWL_CONFIG = config.MAX_PAGES_TO_CRAWL
    CRAWL_DELAY_SECONDS_CONFIG = config.CRAWL_DELAY_SECONDS
    ALLOWED_DOMAINS_CONFIG = config.ALLOWED_DOMAINS
except ImportError:
    logger.warning("config.py not found or variables missing. Using default values for resource_fetcher.")
    FETCH_TIMEOUT_CONFIG_MS = 30000  # ms (for Playwright)
    AIOHTTP_FETCH_TIMEOUT_CONFIG_S = 30  # s (for aiohttp)
    AIOHTTP_HEAD_TIMEOUT_CONFIG_S = 15   # s (for aiohttp head)
    MAX_PAGES_TO_CRAWL_CONFIG = 100
    CRAWL_DELAY_SECONDS_CONFIG = 1.0
    ALLOWED_DOMAINS_CONFIG = []


# --- Constants & Settings ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]
SKIPPED_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    '.zip', '.rar', '.tar.gz', '.7z',
    '.pdf',  # PDFs are handled separately, so skip for general HTML queueing
    '.css', '.js',
    '.xml', '.json', '.txt', # Exclude common data/text files unless specifically targeted
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.mp3', '.wav', '.ogg',
    '.mp4', '.avi', '.mov', '.wmv', '.webm',
    '.exe', '.dmg', '.iso',
]
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_PDF = "pdf"
CONTENT_TYPE_OTHER = "other"
CONTENT_TYPE_FAILED = "failed"
CONTENT_TYPE_ARXIV = "arxiv" # For arXiv specific content type

# --- State Variables ---
LAST_REQUEST_TIME: float = 0.0 # Global for simple rate limiting, manage per domain if needed

# --- Caching ---
CACHE_MAX_SIZE = 50 # Max number of items in LRU cache
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "60") # seconds
# Cache stores: (final_url, content_type_constant, text_content_or_bytes)
resource_cache: LRUCache = LRUCache(maxsize=CACHE_MAX_SIZE)


# --- Helper Functions ---
def get_user_agent() -> str:
    """Selects a random User-Agent string."""
    if not USER_AGENTS:
        return "Mozilla/5.0 (compatible; Python Fetcher)" # Default fallback
    return random.choice(USER_AGENTS)

def get_domain(url: str) -> Optional[str]:
    """Extracts the 'netloc' (e.g., 'google.com') from a URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception: # Catch broad errors from urlparse
        logger.warning(f"Could not parse domain from URL: {url}")
        return None

def _clean_text(text: Optional[str]) -> Optional[str]:
    """Cleans extracted text by normalizing whitespace and reducing excessive newlines."""
    if text is None:
        return None
    try:
        # Normalize spaces and tabs to a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Split into lines, remove leading/trailing whitespace from each, and discard empty lines
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        if not cleaned_lines:
            return None # Return None if all lines were empty after stripping
        # Join lines with a single newline
        text = "\n".join(cleaned_lines)
        # Reduce three or more consecutive newlines to just two
        text = re.sub(r'\n{3,}', '\n\n', text)
        cleaned_text = text.strip() # Final strip of the whole text
        return cleaned_text if cleaned_text else None # Ensure non-empty string or None
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text # Return original text on error


# --- Content Parsing Functions ---
def _parse_pdf_content(pdf_bytes: bytes, source_url: str) -> Optional[str]:
    """Parses text content from PDF bytes using PyMuPDF (fitz)."""
    pdf_text_parts: List[str] = []
    try:
        # Open PDF from bytes
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            if doc.is_encrypted and not doc.authenticate(""): # type: ignore
                logger.warning(f"PDF is encrypted and cannot be opened: {source_url}")
                return None
            for page_num in range(len(doc)):
                page = doc.load_page(page_num) # Correct method to load a page
                pdf_text_parts.append(page.get_text("text", sort=True)) # type: ignore # PyMuPDF page.get_text()
        
        full_pdf_text = "\n\n".join(pdf_text_parts) # Add double newline between page contents
        cleaned_text = _clean_text(full_pdf_text)
        if not cleaned_text:
            logger.warning(f"No text extracted from PDF after cleaning: {source_url}")
        return cleaned_text
    except Exception as e:
        logger.error(f"PDF processing failed for {source_url}: {e}", exc_info=True)
        return None

def _parse_html_content(html_string: str, source_url: str) -> Optional[str]:
    """Parses text content from HTML string using Trafilatura with a basic BS4 fallback."""
    cleaned_text: Optional[str] = None
    try:
        # 1. Attempt extraction with Trafilatura (often good for main content)
        extracted_text_tf: Optional[str] = None
        try:
            extracted_text_tf = extract(
                html_string,
                config=TRAFILATURA_CONFIG,
                favor_recall=True, # Prioritize getting more content
                include_comments=False,
                include_tables=True, # Include table data if relevant
                url=source_url # Provide URL context to Trafilatura
            )
            if extracted_text_tf:
                logger.debug(f"Extracted text using Trafilatura for {source_url}")
            else:
                logger.debug(f"Trafilatura extracted no text for {source_url}, will try BS4 fallback.")
        except Exception as extraction_err:
            logger.warning(f"Trafilatura failed for {source_url}: {extraction_err}. Falling back to BS4.")

        cleaned_text = _clean_text(extracted_text_tf)

        # 2. Fallback to basic BeautifulSoup text extraction if Trafilatura yielded nothing
        if not cleaned_text:
            logger.info(f"Falling back to basic BS4 text extraction for {source_url}")
            soup_fallback = BeautifulSoup(html_string, 'html.parser')
            # Try to find common main content tags, or default to body
            main_content_area = soup_fallback.find('article') or \
                                soup_fallback.find('main') or \
                                soup_fallback.body
            if main_content_area:
                # Get text, separating paragraphs/blocks with newlines, and strip extra whitespace
                raw_text_fallback: str = main_content_area.get_text(separator='\n', strip=True)
                cleaned_text = _clean_text(raw_text_fallback)
                if cleaned_text:
                    logger.debug(f"Extracted text using BS4 fallback for {source_url}")
            if not cleaned_text: # Still no text
                logger.warning(f"Could not extract text using BS4 fallback for {source_url}")
        return cleaned_text
    except Exception as e:
        logger.error(f"HTML parsing failed unexpectedly for {source_url}: {e}", exc_info=True)
        return None


# --- aiohttp Fetching Logic ---
async def _fetch_url_content_aiohttp(
    url: str,
    session: aiohttp.ClientSession,
    proxy: Optional[str] = None
) -> Tuple[str, str, Optional[bytes]]:
    """
    Performs HEAD and GET requests using aiohttp. Primarily for direct file downloads (PDFs).
    Returns: (final_url, content_type_constant, raw_content_bytes)
    """
    final_url = url
    headers = {'User-Agent': get_user_agent()}
    content_type_header = ""
    
    # Create ClientTimeout objects
    head_timeout_obj = aiohttp.ClientTimeout(total=AIOHTTP_HEAD_TIMEOUT_CONFIG_S)
    get_timeout_obj = aiohttp.ClientTimeout(total=AIOHTTP_FETCH_TIMEOUT_CONFIG_S)

    try:
        logger.debug(f"Sending HEAD request (aiohttp) to {url}")
        async with session.head(url, headers=headers, timeout=head_timeout_obj, allow_redirects=True, proxy=proxy) as head_response:
            head_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            content_type_header = head_response.headers.get("Content-Type", "").lower()
            final_url = str(head_response.url) # Update URL in case of redirects
            logger.debug(f"HEAD success (aiohttp) for {url}. Final URL: {final_url}, Content-Type: {content_type_header}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"HEAD request failed (aiohttp) for {url}: {e}. Aborting fetch.")
        return url, CONTENT_TYPE_FAILED, None
    except Exception as e: # Catch any other unexpected error during HEAD
        logger.error(f"Unexpected error during HEAD request (aiohttp) for {url}: {e}", exc_info=True)
        return url, CONTENT_TYPE_FAILED, None

    is_pdf = 'application/pdf' in content_type_header
    is_html = 'text/html' in content_type_header # Less critical here but good to check

    if is_pdf or is_html: # Primarily interested in PDFs for direct download via aiohttp
        try:
            logger.debug(f"Sending GET request (aiohttp) to {final_url}")
            async with session.get(final_url, headers=headers, timeout=get_timeout_obj, allow_redirects=True, proxy=proxy) as response:
                response.raise_for_status()
                final_url = str(response.url) # Update URL again after GET redirects
                raw_content = await response.read()
                logger.debug(f"GET success (aiohttp) for {final_url}. Read {len(raw_content)} bytes.")
                # Determine content type based on header from HEAD, or re-check from GET if necessary
                content_type = CONTENT_TYPE_PDF if is_pdf else CONTENT_TYPE_HTML
                return final_url, content_type, raw_content
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"GET request failed (aiohttp) for {final_url}: {e}")
            return final_url, CONTENT_TYPE_FAILED, None
        except Exception as e: # Catch any other unexpected error during GET
             logger.error(f"Unexpected error during GET request (aiohttp) for {final_url}: {e}", exc_info=True)
             return final_url, CONTENT_TYPE_FAILED, None
    else:
         logger.info(f"Skipping content fetch (aiohttp) for non-PDF/HTML type '{content_type_header}' at {final_url}")
         return final_url, CONTENT_TYPE_OTHER, None


# --- Playwright Fetching Logic ---
async def _fetch_with_playwright(
    url: str,
    playwright_context # Expects a Playwright BrowserContext
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Fetches an HTML URL using Playwright, waits for dynamic content.
    Returns: (final_url, content_type_constant, html_content_string, error_message)
    """
    if not PLAYWRIGHT_AVAILABLE: # Should be checked before calling, but good safeguard
        return url, CONTENT_TYPE_FAILED, None, "Playwright not available"

    page = None
    final_url = url
    error_message: Optional[str] = None
    html_content: Optional[str] = None
    content_type_constant = CONTENT_TYPE_FAILED # Default to failed

    try:
        page = await playwright_context.new_page()
        await page.set_extra_http_headers({'User-Agent': get_user_agent()})

        logger.debug(f"Navigating to {url} with Playwright...")
        response = await page.goto(url, timeout=FETCH_TIMEOUT_CONFIG_MS, wait_until="domcontentloaded")
        final_url = page.url # URL after potential redirects

        if response is None:
            raise PlaywrightError(f"Navigation to {url} returned None response object.")
        
        status = response.status
        if not (200 <= status < 300): # Check for successful HTTP status
            raise PlaywrightError(f"HTTP Error {status} for {final_url}")

        content_type_header = response.headers.get("content-type", "").lower()
        logger.debug(f"Playwright navigation success for {url}. Final URL: {final_url}, Status: {status}, Content-Type: {content_type_header}")

        # Specific wait for OpenReview note lists (example of dynamic content handling)
        if "openreview.net" in final_url:
            wait_timeout_dynamic = 15000 # ms
            try:
                await page.wait_for_selector('.note-list .note a[href*="id="]', timeout=wait_timeout_dynamic)
                logger.info(f"Detected OpenReview note list items with links on {final_url}.")
            except PlaywrightTimeoutError:
                 logger.warning(f"Did not find OpenReview note selector within {wait_timeout_dynamic}ms on {final_url}. Content might be incomplete.")
        
        # General wait for network activity to settle (can help with JS rendering)
        try:
             await page.wait_for_load_state('networkidle', timeout=5000) # Brief network idle wait
             logger.debug(f"Network idle state reached for {final_url}")
        except PlaywrightTimeoutError:
             logger.warning(f"Network did not become idle for {final_url}, proceeding anyway.")

        if 'text/html' in content_type_header:
            html_content = await page.content()
            content_type_constant = CONTENT_TYPE_HTML
            logger.debug(f"Fetched HTML content ({len(html_content) if html_content else 0} chars) for {final_url}")
        else:
            content_type_constant = CONTENT_TYPE_OTHER
            logger.info(f"Skipping content fetch for non-HTML type '{content_type_header}' at {final_url} in Playwright fetch.")

    except PlaywrightTimeoutError as e:
        error_message = f"Playwright TimeoutError for {url}: {str(e)}"
        logger.error(error_message)
    except PlaywrightError as e: # Catch other Playwright-specific errors
        error_message = f"Playwright Error for {url}: {str(e)}"
        logger.error(error_message)
    except Exception as e: # Catch any other unexpected errors
        error_message = f"Unexpected error during Playwright fetch for {url}: {str(e)}"
        logger.error(error_message, exc_info=True)
    finally:
        if page:
             try:
                 await page.close()
             except Exception as close_err: # Catch errors during page close
                 logger.warning(f"Error closing Playwright page for {url}: {close_err}")
    return final_url, content_type_constant, html_content, error_message


# --- Parsing Orchestrator (No changes from previous logic, assuming it's sound) ---
async def _get_parsed_content_orchestrator(
    url: str,
    playwright_context, # Playwright BrowserContext
    session: aiohttp.ClientSession,
    is_pdf_link: bool,
    proxy: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    """Orchestrates fetching and parsing. Uses Playwright for HTML, aiohttp for known PDFs."""
    # This function's internal logic remains largely the same as in the provided snippet,
    # calling the corrected _fetch_url_content_aiohttp or _fetch_with_playwright
    # and then _parse_pdf_content or _parse_html_content.
    final_url = url
    content_type_constant = CONTENT_TYPE_FAILED
    cleaned_text: Optional[str] = None

    if is_pdf_link:
        logger.info(f"Attempting direct download of potential PDF: {url}")
        try:
            final_url, content_type_constant, pdf_bytes = await _fetch_url_content_aiohttp(url, session, proxy)
            if content_type_constant == CONTENT_TYPE_PDF and pdf_bytes:
                cleaned_text = _parse_pdf_content(pdf_bytes, final_url)
            elif content_type_constant != CONTENT_TYPE_FAILED: # Not PDF but not a failure
                 logger.warning(f"Expected PDF but got {content_type_constant} for {url}")
                 content_type_constant = CONTENT_TYPE_OTHER
        except Exception as pdf_err:
            logger.error(f"Error directly fetching/parsing PDF {url}: {pdf_err}", exc_info=True)
            content_type_constant = CONTENT_TYPE_FAILED
    else: # Assume HTML or unknown, use Playwright
        try:
            final_url, content_type_constant, html_content, error_msg = await _fetch_with_playwright(url, playwright_context)
            if error_msg:
                 content_type_constant = CONTENT_TYPE_FAILED
            elif content_type_constant == CONTENT_TYPE_HTML and html_content:
                 cleaned_text = _parse_html_content(html_content, final_url)
        except Exception as html_err:
             logger.error(f"Error fetching/parsing HTML {url} with Playwright: {html_err}", exc_info=True)
             content_type_constant = CONTENT_TYPE_FAILED
    return final_url, content_type_constant, cleaned_text


# --- Link Extraction and Filtering ---
def _extract_links(html_string: str, base_url: str) -> List[Tuple[str, str]]:
    """Extracts navigation and PDF links from HTML. Returns (absolute_url, link_text)."""
    discovered_links: List[Tuple[str, str]] = []
    try:
        soup = BeautifulSoup(html_string, 'html.parser')
        # General navigation links
        for link_tag in soup.find_all('a', href=True):
            if not isinstance(link_tag, bs4_element.Tag): # Ensure it's a Tag
                continue
            href_val = link_tag.get('href') # Use .get() for safety
            href_str: Optional[str] = None

            if isinstance(href_val, list): # Handle if href is unexpectedly a list
                href_str = str(href_val[0]).strip() if href_val else None
            elif isinstance(href_val, str):
                href_str = href_val.strip()
            
            if not href_str: continue

            # Basic filtering for common non-navigational links
            if href_str.startswith(('#', 'javascript:', 'mailto:')):
                continue

            link_text = link_tag.get_text(strip=True) or "N/A" # Get link text or default

            try:
                absolute_url = urljoin(base_url, href_str)
                parsed_abs_url = urlparse(absolute_url)
                # Remove fragment and ensure scheme/netloc are present
                if parsed_abs_url.scheme and parsed_abs_url.netloc:
                    absolute_url_no_frag = urlunparse(parsed_abs_url._replace(fragment=""))
                    discovered_links.append((absolute_url_no_frag, link_text))
            except ValueError:
                logger.debug(f"Skipping invalid relative URL '{href_str}' found on {base_url}")
        
        # Specific selector for OpenReview PDF links (example)
        pdf_link_tags = soup.find_all('a', href=re.compile(r'^/(pdf|attachment)\?id='))
        for link_tag in pdf_link_tags:
            if not isinstance(link_tag, bs4_element.Tag):
                logger.debug(f"Skipping non-Tag element in PDF link extraction: {type(link_tag)}")
                continue
            href_attr = link_tag.get('href')
            if not isinstance(href_attr, str): continue # Ensure href is a string
            
            href = href_attr.strip()
            link_text = link_tag.get_text(strip=True)
            parent_note = link_tag.find_parent('div', class_='note')
            if parent_note and isinstance(parent_note, bs4_element.Tag): # Check if parent_note is a Tag
                title_tag_candidate = parent_note.find('h4')
                # Ensure title_tag_candidate is a Tag before calling get_text
                if isinstance(title_tag_candidate, bs4_element.Tag):
                    title_text = title_tag_candidate.get_text(strip=True)
                    if title_text: link_text = title_text
                elif title_tag_candidate is not None: # It's something else
                    logger.warning(f"Expected a Tag for OpenReview title_tag, got {type(title_tag_candidate)}")

            try:
                absolute_url = urljoin(base_url, href)
                parsed_abs_url = urlparse(absolute_url)
                absolute_url_no_frag = urlunparse(parsed_abs_url._replace(fragment=""))
                # Add if not already discovered by general link finding, or to update link_text
                # This simple check might add duplicates if general also found it; consider a set for URLs if problematic
                discovered_links.append((absolute_url_no_frag, link_text or "PDF Link"))
            except ValueError:
                logger.debug(f"Skipping invalid OpenReview PDF relative URL '{href}' on {base_url}")

    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {e}", exc_info=True)
    return discovered_links


def _is_link_valid_for_queueing(
    link_url: str,
    visited: BloomFilter, # Using BloomFilter for visited check
    start_domains: Set[Optional[str]],
    allowed_domains_override: Optional[List[str]] = None # Allow overriding global config
    ) -> bool:
    """Applies filtering rules to a discovered link to decide if it should be CRAWLED (queued)."""
    if link_url in visited: return False
    if not link_url.startswith(('http://', 'https://')): return False

    try:
        parsed_link = urlparse(link_url)
        link_path_lower = parsed_link.path.lower()
        # Check for explicit PDF patterns that should be handled by _process_pdf_link_task
        if link_path_lower.endswith(".pdf") or link_path_lower.startswith(('/pdf', '/attachment')):
            return False # Don't queue these for general HTML crawling
        # General skipped extensions for queueing
        if any(link_path_lower.endswith(ext) for ext in SKIPPED_EXTENSIONS):
            return False
    except Exception: # Catch errors during path parsing
        logger.debug(f"Could not parse path for link {link_url}, skipping queue check.")
        return False

    link_domain = get_domain(link_url)
    if not link_domain: return False
    
    current_allowed_domains = allowed_domains_override if allowed_domains_override is not None else ALLOWED_DOMAINS_CONFIG

    domain_allowed = False
    if current_allowed_domains: # Explicit allow list
        if any(link_domain == allowed or link_domain.endswith('.' + allowed) for allowed in current_allowed_domains):
            domain_allowed = True
    elif start_domains: # Fallback to same-domain policy
        if link_domain in start_domains:
            domain_allowed = True
    # If neither current_allowed_domains nor start_domains allow it, it's disallowed.

    return domain_allowed


# --- Dedicated PDF Processing Task ---
async def _process_pdf_link_task(
    pdf_url: str,
    link_text: str,
    found_on_url: str,
    session: aiohttp.ClientSession,
    results_list: List[Dict[str, Any]],
    visited: BloomFilter,
    stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore,
    proxy: Optional[str] = None
):
    """Fetches, parses, and adds a single PDF link result."""
    global LAST_REQUEST_TIME, MAX_PAGES_TO_CRAWL_CONFIG, CRAWL_DELAY_SECONDS_CONFIG
    if stop_event.is_set() or pdf_url in visited:
        return

    async with semaphore: # Acquire semaphore for this PDF task
        if stop_event.is_set() or pdf_url in visited:
            return
        visited.add(pdf_url)

        loop = asyncio.get_event_loop()
        now = loop.time()
        time_since_last = now - LAST_REQUEST_TIME
        if time_since_last < CRAWL_DELAY_SECONDS_CONFIG:
            await asyncio.sleep(CRAWL_DELAY_SECONDS_CONFIG - time_since_last)
        LAST_REQUEST_TIME = loop.time()

        logger.debug(f"Starting PDF processing task for: {pdf_url}")
        try:
            final_pdf_url, pdf_content_type, pdf_bytes = await _fetch_url_content_aiohttp(pdf_url, session, proxy)

            if pdf_content_type == CONTENT_TYPE_PDF and pdf_bytes:
                pdf_text_content = _parse_pdf_content(pdf_bytes, final_pdf_url)
                if pdf_text_content:
                    if len(results_list) < MAX_PAGES_TO_CRAWL_CONFIG:
                        pdf_title = link_text if link_text and link_text != "N/A" else os.path.basename(urlparse(final_pdf_url).path)
                        if not pdf_title: pdf_title = final_pdf_url # Fallback title

                        pdf_result_data = {
                            "entry_id": final_pdf_url, # Add entry_id
                            "source": CONTENT_TYPE_PDF, "url": final_pdf_url, "title": pdf_title,
                            "content": f"{pdf_title}\\n\\n{pdf_text_content}", # Prepend title to content
                            "authors": [], "published": None, "found_on": found_on_url
                        }
                        results_list.append(pdf_result_data)
                        logger.info(f"Processed PDF: {final_pdf_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL_CONFIG}) Found on: {found_on_url}")
                        if len(results_list) >= MAX_PAGES_TO_CRAWL_CONFIG and not stop_event.is_set():
                            logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL_CONFIG}) reached after PDF. Signalling stop.")
                            stop_event.set()
                    else: # Limit reached
                         if not stop_event.is_set(): stop_event.set()
                else:
                    logger.warning(f"Failed to parse PDF content from: {final_pdf_url}")
            elif pdf_content_type == CONTENT_TYPE_FAILED:
                 logger.warning(f"Failed to fetch PDF link: {pdf_url}")
            else: # Not a PDF, but fetch didn't fail (e.g., got HTML page for a .pdf link)
                 logger.warning(f"Expected PDF but got {pdf_content_type} for link: {pdf_url}")
        except Exception as pdf_err:
            logger.error(f"Error processing PDF link task for {pdf_url}: {pdf_err}", exc_info=True)


# --- Web Crawler Task (Playwright for HTML, spawns PDF tasks) ---
async def _process_crawl_url_pw(
    current_url: str,
    playwright_context, # Playwright BrowserContext
    session: aiohttp.ClientSession, # For spawning PDF tasks
    results_list: List[Dict[str, Any]],
    visited: BloomFilter,
    queue: deque[str],
    active_tasks: Set[asyncio.Task],
    start_domains: Set[Optional[str]],
    visited_set_limit: int, # Max capacity of the Bloom filter
    stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore, # Main semaphore for HTML page tasks
    pdf_semaphore: asyncio.Semaphore, # Separate semaphore for PDF tasks
    process_pdfs: bool,
    proxy: Optional[str] = None
):
    """Task function using Playwright for HTML fetch, spawns PDF tasks."""
    global LAST_REQUEST_TIME, MAX_PAGES_TO_CRAWL_CONFIG, CRAWL_DELAY_SECONDS_CONFIG
    if stop_event.is_set() or current_url in visited: return

    loop = asyncio.get_event_loop()
    now = loop.time()
    time_since_last = now - LAST_REQUEST_TIME
    if time_since_last < CRAWL_DELAY_SECONDS_CONFIG:
        await asyncio.sleep(CRAWL_DELAY_SECONDS_CONFIG - time_since_last)
    LAST_REQUEST_TIME = loop.time()
    
    visited.add(current_url) # Mark as visited *before* processing

    html_content_for_links: Optional[str] = None
    final_url = current_url

    try:
        final_url, content_type, html_content_for_links, error_msg = await _fetch_with_playwright(current_url, playwright_context,)

        if final_url != current_url: # Handle redirects
             if final_url in visited: return
             visited.add(final_url)

        if error_msg:
            logger.warning(f"Playwright fetch failed for {current_url}: {error_msg}")
            return # Stop processing this URL if fetch failed

        if content_type == CONTENT_TYPE_HTML and html_content_for_links:
            text_content = _parse_html_content(html_content_for_links, final_url)
            if text_content:
                 if len(results_list) < MAX_PAGES_TO_CRAWL_CONFIG:
                     page_title = final_url # Default title
                     try: # Try to extract a better title from HTML
                         soup_title = BeautifulSoup(html_content_for_links, 'html.parser')
                         title_tag = soup_title.find('title')
                         if title_tag:
                             title_text = title_tag.get_text(strip=True)
                             if title_text: page_title = title_text
                     except Exception: pass # Ignore errors in title extraction

                     if not page_title or page_title == final_url: # Fallback title from content
                          first_line = text_content.split('\n', 1)[0].strip()
                          if first_line and len(first_line) < 150: page_title = first_line # Use first line if reasonable

                     result_data = {"entry_id": final_url, # Add entry_id
                                    "source": content_type, "url": final_url, "title": page_title,
                                    "content": text_content, "authors": [], "published": None}
                     results_list.append(result_data)
                     logger.info(f"Processed HTML: {final_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL_CONFIG})")
                     if len(results_list) >= MAX_PAGES_TO_CRAWL_CONFIG and not stop_event.is_set():
                         logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL_CONFIG}) reached. Signalling stop.")
                         stop_event.set()
                 else: # Limit reached
                      if not stop_event.is_set(): stop_event.set()
            else:
                 logger.warning(f"Failed to parse text content from HTML: {final_url}")

            # Link Extraction and Task Spawning (if HTML content was fetched)
            if not stop_event.is_set():
                discovered_links = _extract_links(html_content_for_links, final_url)
                for link_url_abs, link_text_hint in discovered_links:
                    if stop_event.is_set(): break
                    if link_url_abs in visited: continue
                    if len(visited) >= visited_set_limit and not stop_event.is_set(): # Check Bloom filter capacity
                        logger.warning("Bloom filter capacity reached. Stopping new link processing.")
                        stop_event.set()
                        break
                    
                    parsed_link_path = urlparse(link_url_abs).path.lower()
                    is_potential_pdf = parsed_link_path.endswith(".pdf") or \
                                       parsed_link_path.startswith(('/pdf', '/attachment'))

                    if process_pdfs and is_potential_pdf:
                        pdf_task = asyncio.create_task(
                            _process_pdf_link_task(
                                pdf_url=link_url_abs, link_text=link_text_hint, found_on_url=final_url,
                                session=session, results_list=results_list, visited=visited,
                                stop_event=stop_event, semaphore=pdf_semaphore, proxy=proxy # Use PDF semaphore
                            )
                        )
                        active_tasks.add(pdf_task)
                        pdf_task.add_done_callback(active_tasks.discard)
                    elif not is_potential_pdf: # It's a potential HTML link
                        if _is_link_valid_for_queueing(link_url_abs, visited, start_domains):
                            queue.append(link_url_abs)
                            # logger.debug(f"Queued HTML link: {link_url_abs}")
    except Exception as e:
        logger.error(f"Processing {current_url} failed unexpectedly in task: {e}", exc_info=True)


# --- Main Crawler Orchestration ---
async def _crawl_manager_pw(
    start_urls: List[str],
    playwright_context, # Playwright BrowserContext
    session: aiohttp.ClientSession, # For PDF tasks
    html_semaphore: asyncio.Semaphore, # For controlling HTML page fetches
    pdf_semaphore: asyncio.Semaphore,  # For controlling PDF fetches
    process_pdfs: bool,
    proxies: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Manages the crawling process using Playwright for HTML and aiohttp for PDFs."""
    global MAX_PAGES_TO_CRAWL_CONFIG
    results_list: List[Dict[str, Any]] = []
    # Estimate Bloom filter capacity: more if processing PDFs as they add to visited items
    estimated_capacity = MAX_PAGES_TO_CRAWL_CONFIG * (50 if process_pdfs else 20)
    visited: BloomFilter = BloomFilter(capacity=max(estimated_capacity, 5000), error_rate=0.001)
    
    queue: deque[str] = deque()
    for url in start_urls: # Add initial URLs if not already visited (e.g. from a previous run)
        if url not in visited:
            queue.append(url)

    start_domains: Set[Optional[str]] = {get_domain(url) for url in start_urls if get_domain(url)}
    visited_set_limit = visited.capacity # Max items before Bloom filter saturation is high
    stop_event = asyncio.Event()
    active_tasks: Set[asyncio.Task] = set()

    logger.info(f"Crawl Manager started. Initial Queue: {len(queue)}. Visited Capacity: {visited.capacity}. Max Pages/PDFs: {MAX_PAGES_TO_CRAWL_CONFIG}")

    while (queue or active_tasks) and not stop_event.is_set():
        # Launch new HTML processing tasks from queue
        while queue and len(active_tasks) < html_semaphore._value + pdf_semaphore._value and not stop_event.is_set(): # Consider total tasks
            if html_semaphore.locked(): # Check if we can launch more HTML tasks
                 await asyncio.sleep(0.1) # Wait if HTML semaphore is full
                 continue

            current_url = queue.popleft()
            if current_url in visited: continue

            await html_semaphore.acquire() # Acquire semaphore for this HTML page task
            proxy_to_use = random.choice(proxies) if proxies else None
            
            task = asyncio.create_task(
                _process_crawl_url_pw(
                    current_url=current_url, playwright_context=playwright_context, session=session,
                    results_list=results_list, visited=visited, queue=queue, active_tasks=active_tasks,
                    start_domains=start_domains, visited_set_limit=visited_set_limit,
                    stop_event=stop_event, semaphore=html_semaphore, # Pass HTML semaphore
                    pdf_semaphore=pdf_semaphore, # Pass PDF semaphore for it to use
                    process_pdfs=process_pdfs, proxy=proxy_to_use
                )
            )
            active_tasks.add(task)
            # Callback to release HTML semaphore and remove task from active_tasks
            def html_task_done_callback(fut, url=current_url, sem=html_semaphore):
                try: fut.result() # Check for exceptions from the task
                except asyncio.CancelledError: logger.debug(f"HTML task for {url} was cancelled.")
                except Exception as task_exc: logger.error(f"HTML task for {url} raised: {task_exc}", exc_info=False)
                finally:
                    sem.release()
                    active_tasks.discard(fut)
                    logger.debug(f"HTML task for {url} finished. Released HTML semaphore. Active tasks: {len(active_tasks)}")
            task.add_done_callback(html_task_done_callback)

        if not active_tasks and not queue and not stop_event.is_set(): # All done
            break
        if not active_tasks and queue and not stop_event.is_set(): # Tasks finished but queue has items
            await asyncio.sleep(0.01) # Brief pause to allow loop to pick up new tasks
            continue

        # Wait for some tasks to complete if queue is empty or semaphores are full
        logger.debug(f"Waiting for tasks. Active: {len(active_tasks)}, Queue: {len(queue)}, Stop: {stop_event.is_set()}")
        if active_tasks:
            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=5.0)
            if not done and not stop_event.is_set() and not queue: # Timeout with no tasks done, but still expecting work
                 logger.debug("Task wait timed out, but work might still be pending or queue might refill.")
        else: # No active tasks, but loop hasn't exited (e.g. stop_event was just set)
            await asyncio.sleep(0.1)


    if active_tasks: # Cleanup remaining tasks if loop exited due to stop_event or other reasons
        logger.info(f"Crawling loop ended. Cancelling {len(active_tasks)} remaining tasks...")
        for task_to_cancel in list(active_tasks): task_to_cancel.cancel()
        try:
            await asyncio.gather(*active_tasks, return_exceptions=True) # Wait for cancellations
            logger.info("All remaining tasks cancelled/completed.")
        except Exception as e: logger.error(f"Error during final task cleanup: {e}")
    
    logger.info(f"Crawling finished. Processed {len(results_list)} total items. Queue size: {len(queue)}.")
    return results_list


# --- Public API Function ---
async def crawl_and_fetch_web_articles(
    start_urls: List[str],
    process_pdfs_linked: bool = True,
    max_pages_override: Optional[int] = None,
    proxies: Optional[List[str]] = None,
    max_concurrent_html: int = 5, # Max concurrent Playwright page loads
    max_concurrent_pdf: int = 10  # Max concurrent PDF downloads
    ) -> List[Dict[str, Any]]:
    """Public entry point for web crawling using Playwright and aiohttp."""
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Playwright is not installed or available. Cannot perform dynamic web crawling.")
        return []
    if not start_urls:
        logger.info("No starting URLs provided for crawling.")
        return []

    global MAX_PAGES_TO_CRAWL_CONFIG # Allow override
    original_max_pages = MAX_PAGES_TO_CRAWL_CONFIG
    if max_pages_override is not None and max_pages_override > 0:
        logger.info(f"Overriding config MAX_PAGES_TO_CRAWL ({original_max_pages}) with {max_pages_override}")
        MAX_PAGES_TO_CRAWL_CONFIG = max_pages_override

    logger.info(f"Starting web crawl. HTML Concurrency: {max_concurrent_html}, PDF Concurrency: {max_concurrent_pdf}. Max Items: {MAX_PAGES_TO_CRAWL_CONFIG}. Process PDFs: {process_pdfs_linked}.")
    
    html_semaphore = asyncio.Semaphore(max_concurrent_html)
    pdf_semaphore = asyncio.Semaphore(max_concurrent_pdf)
    results: List[Dict[str, Any]] = []

    playwright_instance = None
    browser = None
    context = None
    aiohttp_session = None

    try:
        playwright_instance = await async_playwright().start()
        browser = await playwright_instance.chromium.launch(headless=True) # Consider headless=False for debugging
        context = await browser.new_context(
            user_agent=get_user_agent(),
            ignore_https_errors=True,
            # Consider viewport, geolocation, etc. if needed
        )
        # Setup aiohttp session
        # Adjust connector limits based on total expected concurrency
        connector = aiohttp.TCPConnector(limit_per_host=max(5, max_concurrent_pdf // 2), limit=max(10, max_concurrent_pdf * 2), ssl=False) # ssl=False for simplicity, use system certs in prod
        aiohttp_session = aiohttp.ClientSession(connector=connector)

        results = await _crawl_manager_pw(
            start_urls=start_urls,
            playwright_context=context,
            session=aiohttp_session,
            html_semaphore=html_semaphore,
            pdf_semaphore=pdf_semaphore,
            process_pdfs=process_pdfs_linked,
            proxies=proxies
        )
        logger.info(f"Web crawl finished. Returning {len(results)} processed items.")

    except Exception as e:
         logger.error(f"Web crawl failed with a critical error: {e}", exc_info=True)
    finally:
        if aiohttp_session: await aiohttp_session.close()
        if context: await context.close()
        if browser: await browser.close()
        if playwright_instance: await playwright_instance.stop()
        
        if max_pages_override is not None: # Restore original config value
            MAX_PAGES_TO_CRAWL_CONFIG = original_max_pages
            logger.info(f"Restored MAX_PAGES_TO_CRAWL to {original_max_pages}")
        logger.info("Web crawling resources cleaned up.")
    return results


# --- arXiv Fetcher (Unchanged from provided snippet) ---
async def fetch_arxiv_papers(
    query: str,
    max_results: int = 10,
    days_back: Optional[int] = None,
    sort_by: str = "relevance", # "relevance" or "lastUpdatedDate"
    session: Optional[aiohttp.ClientSession] = None,
    proxy: Optional[str] = None,
    fetch_pdfs: bool = False,
    playwright_context = None,  # Kept for backward compatibility but not used
    verbose: bool = False  # Add verbose flag to control detailed logging
) -> List[Dict[str, Any]]:
    """
    Fetches paper metadata from the arXiv API based on a search query.
    Includes pagination to retrieve more than the typical API limit per call.
    Returns a list of dictionaries, where each dictionary contains metadata for a paper,
    or an error dictionary if something went wrong.
    
    Parameters:
        query: Search query string
        max_results: Maximum number of results to fetch
        days_back: Optional number of days back to limit search
        sort_by: Sort order ("relevance" or "lastUpdatedDate")
        session: Optional aiohttp session to use
        proxy: Optional proxy URL
        fetch_pdfs: Whether to fetch PDF content using direct HTTP requests
        playwright_context: Kept for backward compatibility but no longer used
        verbose: Enable detailed progress logging
    """
    # Ensure necessary imports are available at the module level or add them here if scoped.
    # For this edit, assuming:
    import xml.etree.ElementTree as ET
    from datetime import datetime, timedelta # Ensure timedelta is imported
    import ssl
    from typing import Union, List, Dict, Any, Optional # Ensure Optional is imported
    import fitz # PyMuPDF
    # logger is assumed to be defined at module level e.g., logger = logging.getLogger(__name__)
    # from urllib.parse import urlencode # Used earlier for debug URL

    from .. import config

    # Helper function to fetch and extract text from PDF using aiohttp directly
    async def _fetch_pdf_content_direct(pdf_url: str, session: aiohttp.ClientSession) -> Optional[str]:
        """
        Fetches a PDF from a URL using aiohttp directly (much more reliable for arXiv PDFs),
        extracts text content using PyMuPDF.
        """
        if verbose:
            logger.info(f"🔄 PDF Fetch: Starting direct download for {pdf_url}")
            
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)  # 2 minute timeout
            
            async with session.get(pdf_url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    pdf_bytes = await response.read()
                    
                    if verbose:
                        logger.info(f"✅ PDF Fetch: Downloaded {len(pdf_bytes)} bytes from {pdf_url}")
                    
                    if len(pdf_bytes) == 0:
                        logger.warning(f"Downloaded PDF is empty from {pdf_url}")
                        return None
                    
                    # Use PyMuPDF to extract text from PDF bytes
                    try:
                        if verbose:
                            logger.info(f"🔄 PDF Fetch: Extracting text from PDF: {pdf_url}")
                            
                        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        text = ""
                        page_count = pdf_doc.page_count
                        
                        if verbose:
                            logger.info(f"📄 PDF Fetch: Processing {page_count} pages from {pdf_url}")
                        
                        for page_num in range(page_count):
                            if verbose and page_num % 10 == 0 and page_num > 0:
                                logger.info(f"📄 PDF Fetch: Processed {page_num}/{page_count} pages from {pdf_url}")
                                
                            pdf_page = pdf_doc.load_page(page_num)
                            page_text = pdf_page.get_text("text", sort=True) # type: ignore # PyMuPDF page.get_text()
                            text += page_text + "\n"

                        pdf_doc.close()
                        
                        if verbose:
                            text_length = len(text)
                            logger.info(f"✅ PDF Fetch: Successfully extracted {text_length} characters from {page_count} pages for {pdf_url}")
                            
                        return text.strip()
                        
                    except Exception as e:
                        logger.error(f"Error extracting text from PDF bytes for {pdf_url}: {e}", exc_info=True)
                        if verbose:
                            logger.error(f"❌ PDF Fetch: Failed to extract text from {pdf_url}. Error: {str(e)}")
                        return None
                        
                else:
                    logger.error(f"Failed to download PDF from {pdf_url}: HTTP {response.status}")
                    if verbose:
                        logger.error(f"❌ PDF Fetch: HTTP {response.status} error for {pdf_url}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout while downloading PDF from {pdf_url}")
            if verbose:
                logger.error(f"❌ PDF Fetch: Timeout error for {pdf_url}")
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF {pdf_url}: {e}", exc_info=True)
            if verbose:
                logger.error(f"❌ PDF Fetch: Failed to download {pdf_url}. Error: {str(e)}")
            return None
    
    # Main arXiv API fetching logic
    base_url = "http://export.arxiv.org/api/query"
    if verbose:
        logger.info(f"🔄 ArXiv Fetch: Starting fetch for query='{query}', max_results={max_results}")
    
    # Build query parameters
    search_query = query
    if days_back is not None:
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            date_filter = cutoff_date.strftime('%Y%m%d%H%M%S')
            search_query += f" AND submittedDate:[{date_filter}* TO *]"
            if verbose:
                logger.info(f"📅 ArXiv Fetch: Added date filter for last {days_back} days")
        except Exception as date_e:
            logger.warning(f"Failed to add date filter: {date_e}")
    
    # Set sort order
    sort_order = "relevance" if sort_by == "relevance" else "lastUpdatedDate"
    
    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': max_results,
        'sortBy': sort_order,
        'sortOrder': 'descending'
    }
    
    # Create session if not provided
    session_created = False
    if session is None:
        connector = None
        if proxy:
            connector = aiohttp.TCPConnector()
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=180, connect=30, sock_read=60)  # Increased timeout: 3 minutes total, 30s connect, 60s read
        )
        session_created = True
    
    papers_list = []
    
    try:
        if verbose:
            logger.info(f"🌐 ArXiv Fetch: Making API request to {base_url}")
        
        kwargs = {}
        if proxy:
            kwargs['proxy'] = proxy
        
        async with session.get(base_url, params=params, **kwargs) as response:
            if response.status == 200:
                xml_content = await response.text()
                if verbose:
                    logger.info(f"✅ ArXiv Fetch: Got API response ({len(xml_content)} chars)")
                
                # Parse XML response
                try:
                    root = ET.fromstring(xml_content)
                    
                    # Find all entry elements
                    entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                    if verbose:
                        logger.info(f"📄 ArXiv Fetch: Found {len(entries)} entries in XML")
                    
                    for entry in entries:
                        try:
                            # Extract basic metadata
                            title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                            title = title_elem.text.strip() if (title_elem is not None and title_elem.text is not None) else "N/A"
                            
                            summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                            summary = summary_elem.text.strip() if (summary_elem is not None and summary_elem.text is not None) else ""
                            
                            # Extract published date
                            published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                            published = published_elem.text.strip() if (published_elem is not None and published_elem.text is not None) else ""
                            
                            # Extract updated date
                            updated_elem = entry.find('.//{http://www.w3.org/2005/Atom}updated')
                            updated = updated_elem.text.strip() if (updated_elem is not None and updated_elem.text is not None) else ""
                            
                            # Extract arXiv ID from the entry ID
                            id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
                            entry_id = ""
                            pdf_url = ""
                            if id_elem is not None and id_elem.text is not None:
                                entry_id = id_elem.text.strip()
                                # Extract arXiv ID from URL like http://arxiv.org/abs/1234.5678v1
                                arxiv_id = entry_id.split('/')[-1] if '/' in entry_id else entry_id
                                pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
                            
                            # Extract authors
                            authors = []
                            author_elems = entry.findall('.//{http://www.w3.org/2005/Atom}author')
                            for author_elem in author_elems:
                                name_elem = author_elem.find('.//{http://www.w3.org/2005/Atom}name')
                                if name_elem is not None and name_elem.text is not None:
                                    authors.append(name_elem.text.strip())
                            
                            # Extract categories
                            categories = []
                            category_elems = entry.findall('.//{http://www.w3.org/2005/Atom}category')
                            for cat_elem in category_elems:
                                term = cat_elem.get('term')
                                if term:
                                    categories.append(term)
                            
                            # Build paper metadata
                            paper_data = {
                                'title': title,
                                'authors': authors,
                                'published': published.split('T')[0] if 'T' in published else published,  # Keep date only
                                'updated': updated.split('T')[0] if 'T' in updated else updated,
                                'entry_id': entry_id,
                                'pdf_url': pdf_url,
                                'url': entry_id,  # For compatibility
                                'summary': summary.replace('\n', ' ').strip(),
                                'categories': categories,
                                'source': 'arxiv'
                            }
                            
                            # Fetch PDF content if requested
                            content = ""
                            if fetch_pdfs and pdf_url:
                                if verbose:
                                    logger.info(f"📄 ArXiv Fetch: Attempting PDF fetch for {title[:50]}...")
                                
                                try:
                                    # Use direct aiohttp download instead of Playwright
                                    pdf_content = await _fetch_pdf_content_direct(pdf_url, session)
                                    if pdf_content:
                                        content = pdf_content
                                        if verbose:
                                            logger.info(f"✅ ArXiv Fetch: PDF content extracted ({len(content)} chars)")
                                    else:
                                        if verbose:
                                            logger.warning(f"⚠️ ArXiv Fetch: PDF content extraction failed for {title[:50]}")
                                except Exception as pdf_e:
                                    logger.warning(f"PDF fetch failed for {title}: {pdf_e}")
                                    if verbose:
                                        logger.warning(f"⚠️ ArXiv Fetch: PDF error for {title[:50]}: {str(pdf_e)}")
                            
                            # Use abstract as fallback if no PDF content
                            if not content:
                                content = summary
                            
                            paper_data['content'] = content
                            papers_list.append(paper_data)
                            
                        except Exception as entry_e:
                            logger.warning(f"Failed to parse arXiv entry: {entry_e}")
                            continue
                    
                    if verbose:
                        logger.info(f"✅ ArXiv Fetch: Successfully processed {len(papers_list)} papers")
                
                except ET.ParseError as parse_e:
                    logger.error(f"Failed to parse arXiv XML response: {parse_e}")
                    if verbose:
                        logger.error(f"❌ ArXiv Fetch: XML parsing failed: {str(parse_e)}")
                
            else:
                logger.error(f"arXiv API request failed with status {response.status}")
                if verbose:
                    logger.error(f"❌ ArXiv Fetch: API request failed with status {response.status}")
    
    except Exception as e:
        logger.error(f"Error fetching arXiv papers: {e}", exc_info=True)
        if verbose:
            logger.error(f"❌ ArXiv Fetch: General error: {str(e)}")
    
    finally:
        if session_created:
            await session.close()
    
    if verbose:
        logger.info(f"🏁 ArXiv Fetch: Completed. Returning {len(papers_list)} papers")
    
    # Always return a list, never None
    return papers_list

