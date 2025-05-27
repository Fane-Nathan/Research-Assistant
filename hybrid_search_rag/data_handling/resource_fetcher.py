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
from typing import Any, Dict, List, Optional, Tuple, Set, Type, Union 
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp # type: ignore
import fitz # type: ignore
import ssl
from bs4 import BeautifulSoup, element as bs4_element # type: ignore
from cachetools import LRUCache # type: ignore
from pybloom_live import BloomFilter # type: ignore
from trafilatura import extract # type: ignore
from trafilatura.settings import use_config # type: ignore
import xml.etree.ElementTree as ET 

# --- Playwright Import and Dummy Definitions ---
class _InternalDummyPlaywrightError(Exception): pass
class _InternalDummyPlaywrightTimeoutError(Exception): pass

PlaywrightError: Type[Exception] = _InternalDummyPlaywrightError
PlaywrightTimeoutError: Type[Exception] = _InternalDummyPlaywrightTimeoutError
PLAYWRIGHT_AVAILABLE = False

try:
    from playwright.async_api import async_playwright # type: ignore
    from playwright.async_api import TimeoutError as _ActualPlaywrightTimeoutError_Imported # type: ignore
    from playwright.async_api import Error as _ActualPlaywrightError_Imported # type: ignore

    PlaywrightTimeoutError = _ActualPlaywrightTimeoutError_Imported
    PlaywrightError = _ActualPlaywrightError_Imported
    PLAYWRIGHT_AVAILABLE = True 
except ImportError:
    logging.getLogger(__name__).critical("Playwright not found. Please install it: pip install playwright && playwright install")
    
    class _DummyPage:
        def __init__(self, owning_context: Any):
            self.url: str = "http://dummy.invalid/page"
            self._headers: Dict[str, str] = {}
            self._owning_context = owning_context 

        async def close(self, *args: Any, **kwargs: Any) -> None: pass
        async def content(self, *args: Any, **kwargs: Any) -> str:
            return f"<html><head><title>Dummy Page</title></head><body>Dummy Content for {self.url}</body></html>"
        async def goto(self, url: str, *args: Any, **kwargs: Any) -> Any:
            self.url = url
            class DummyResponse: 
                def __init__(self, page_url: str, page_headers: Dict[str,str]):
                    self.status = 200
                    self.headers: Dict[str, str] = {**page_headers, "content-type": "text/html"}
                    self.url: str = page_url
                async def body(self) -> bytes: # Add async body method
                    return f"Dummy body for {self.url}".encode()
            return DummyResponse(self.url, self._headers)
        async def wait_for_selector(self, selector: str, *args: Any, **kwargs: Any) -> None: pass
        async def wait_for_load_state(self, state: str = 'load', *args: Any, **kwargs: Any) -> None: pass
        async def set_extra_http_headers(self, headers: Dict[str, str]) -> None: self._headers.update(headers)

    class _DummyContext:
        def __init__(self, owning_browser: Any):
            self._owning_browser = owning_browser
        async def new_page(self, *args: Any, **kwargs: Any) -> _DummyPage: return _DummyPage(owning_context=self)
        async def close(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyBrowser:
        def __init__(self, owning_chromium: Any): self._owning_chromium = owning_chromium
        async def new_context(self, *args: Any, **kwargs: Any) -> _DummyContext: return _DummyContext(owning_browser=self)
        async def close(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyChromium: 
        async def launch(self, *args: Any, **kwargs: Any) -> _DummyBrowser: return _DummyBrowser(owning_chromium=self)

    class _DummyPlaywrightInstance: 
        def __init__(self): self.chromium = _DummyChromium()
        async def stop(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyPlaywrightContextManager: 
        def __init__(self): self._playwright_instance: Optional[_DummyPlaywrightInstance] = None
        async def start(self, *args: Any, **kwargs: Any) -> _DummyPlaywrightInstance:
            if self._playwright_instance is None: self._playwright_instance = _DummyPlaywrightInstance()
            return self._playwright_instance
        async def __aenter__(self) -> _DummyPlaywrightInstance: return await self.start()
        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self._playwright_instance:
                await self._playwright_instance.stop()
                self._playwright_instance = None 
    
    _dummy_playwright_cm_singleton: Optional[_DummyPlaywrightContextManager] = None
    def replacement_async_playwright_func(*args: Any, **kwargs: Any) -> _DummyPlaywrightContextManager:
        global _dummy_playwright_cm_singleton
        if _dummy_playwright_cm_singleton is None: _dummy_playwright_cm_singleton = _DummyPlaywrightContextManager()
        return _dummy_playwright_cm_singleton
    async_playwright = replacement_async_playwright_func


logger = logging.getLogger(__name__)

# --- Configuration Loading ---
try:
    from .. import config
    FETCH_TIMEOUT_CONFIG_MS = config.FETCH_TIMEOUT * 1000
    AIOHTTP_FETCH_TIMEOUT_CONFIG_S = config.FETCH_TIMEOUT
    AIOHTTP_HEAD_TIMEOUT_CONFIG_S = config.HEAD_TIMEOUT
    MAX_PAGES_TO_CRAWL_CONFIG = config.MAX_PAGES_TO_CRAWL
    CRAWL_DELAY_SECONDS_CONFIG = config.CRAWL_DELAY_SECONDS
    ALLOWED_DOMAINS_CONFIG = config.ALLOWED_DOMAINS
except ImportError:
    logger.warning("config.py not found or variables missing. Using default values for resource_fetcher.")
    FETCH_TIMEOUT_CONFIG_MS = 30000 
    AIOHTTP_FETCH_TIMEOUT_CONFIG_S = 30  
    AIOHTTP_HEAD_TIMEOUT_CONFIG_S = 15   
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
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.zip', '.rar', '.tar.gz', '.7z',
    '.pdf', '.css', '.js', '.xml', '.json', '.txt', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov', '.wmv', '.webm', '.exe', '.dmg', '.iso',
]
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_PDF = "pdf"
CONTENT_TYPE_OTHER = "other"
CONTENT_TYPE_FAILED = "failed"
CONTENT_TYPE_ARXIV = "arxiv" 

LAST_REQUEST_TIME: float = 0.0 
CACHE_MAX_SIZE = 50 
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "60") 
resource_cache: LRUCache = LRUCache(maxsize=CACHE_MAX_SIZE)


# --- Helper Functions ---
def get_user_agent() -> str:
    if not USER_AGENTS: return "Mozilla/5.0 (compatible; Python Fetcher)" 
    return random.choice(USER_AGENTS)

def get_domain(url: str) -> Optional[str]:
    try: return urlparse(url).netloc.lower()
    except Exception: logger.warning(f"Could not parse domain from URL: {url}"); return None

def _clean_text(text: Optional[str]) -> Optional[str]:
    if text is None: return None
    try:
        text = re.sub(r'[ \t]+', ' ', text)
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        if not cleaned_lines: return None 
        text = "\n".join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        cleaned_text = text.strip() 
        return cleaned_text if cleaned_text else None 
    except Exception as e: logger.error(f"Text cleaning failed: {e}"); return text 


# --- Content Parsing Functions ---
def _parse_pdf_content(pdf_bytes: bytes, source_url: str) -> Optional[str]:
    pdf_text_parts: List[str] = []
    try:
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc: # type: ignore
            if doc.is_encrypted and not doc.authenticate(""): 
                logger.warning(f"PDF is encrypted and cannot be opened: {source_url}")
                return None
            for page_num in range(len(doc)):
                page = doc.load_page(page_num) 
                pdf_text_parts.append(page.get_text("text", sort=True)) 
        
        full_pdf_text = "\n\n".join(pdf_text_parts) 
        cleaned_text = _clean_text(full_pdf_text)
        if not cleaned_text:
            logger.warning(f"No text extracted from PDF after cleaning: {source_url}")
        return cleaned_text
    except Exception as e:
        logger.error(f"PDF processing failed for {source_url}: {e}", exc_info=True)
        return None

def _parse_html_content(html_string: str, source_url: str) -> Optional[str]:
    cleaned_text: Optional[str] = None
    try:
        extracted_text_tf: Optional[str] = None
        try:
            extracted_text_tf = extract(
                html_string, config=TRAFILATURA_CONFIG, favor_recall=True, 
                include_comments=False, include_tables=True, url=source_url
            )
            if extracted_text_tf: logger.debug(f"Extracted text using Trafilatura for {source_url}")
            else: logger.debug(f"Trafilatura extracted no text for {source_url}, will try BS4 fallback.")
        except Exception as extraction_err:
            logger.warning(f"Trafilatura failed for {source_url}: {extraction_err}. Falling back to BS4.")

        cleaned_text = _clean_text(extracted_text_tf)

        if not cleaned_text:
            logger.info(f"Falling back to basic BS4 text extraction for {source_url}")
            soup_fallback = BeautifulSoup(html_string, 'html.parser')
            main_content_area = soup_fallback.find('article') or \
                                soup_fallback.find('main') or \
                                soup_fallback.body
            if main_content_area:
                raw_text_fallback: str = main_content_area.get_text(separator='\n', strip=True)
                cleaned_text = _clean_text(raw_text_fallback)
                if cleaned_text: logger.debug(f"Extracted text using BS4 fallback for {source_url}")
            if not cleaned_text: logger.warning(f"Could not extract text using BS4 fallback for {source_url}")
        return cleaned_text
    except Exception as e:
        logger.error(f"HTML parsing failed unexpectedly for {source_url}: {e}", exc_info=True)
        return None


# --- aiohttp Fetching Logic ---
async def _fetch_url_content_aiohttp(
    url: str, session: aiohttp.ClientSession, proxy: Optional[str] = None
) -> Tuple[str, str, Optional[bytes]]:
    final_url = url
    headers = {'User-Agent': get_user_agent()}
    content_type_header = ""
    head_timeout_obj = aiohttp.ClientTimeout(total=AIOHTTP_HEAD_TIMEOUT_CONFIG_S)
    get_timeout_obj = aiohttp.ClientTimeout(total=AIOHTTP_FETCH_TIMEOUT_CONFIG_S)

    try:
        logger.debug(f"Sending HEAD request (aiohttp) to {url}")
        async with session.head(url, headers=headers, timeout=head_timeout_obj, allow_redirects=True, proxy=proxy) as head_response:
            head_response.raise_for_status() 
            content_type_header = head_response.headers.get("Content-Type", "").lower()
            final_url = str(head_response.url) 
            logger.debug(f"HEAD success (aiohttp) for {url}. Final URL: {final_url}, Content-Type: {content_type_header}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"HEAD request failed (aiohttp) for {url}: {e}. Aborting fetch.")
        return url, CONTENT_TYPE_FAILED, None
    except Exception as e: 
        logger.error(f"Unexpected error during HEAD request (aiohttp) for {url}: {e}", exc_info=True)
        return url, CONTENT_TYPE_FAILED, None

    is_pdf = 'application/pdf' in content_type_header
    is_html = 'text/html' in content_type_header 

    if is_pdf or is_html: 
        try:
            logger.debug(f"Sending GET request (aiohttp) to {final_url}")
            async with session.get(final_url, headers=headers, timeout=get_timeout_obj, allow_redirects=True, proxy=proxy) as response:
                response.raise_for_status()
                final_url = str(response.url) 
                raw_content = await response.read()
                logger.debug(f"GET success (aiohttp) for {final_url}. Read {len(raw_content)} bytes.")
                content_type = CONTENT_TYPE_PDF if is_pdf else CONTENT_TYPE_HTML
                return final_url, content_type, raw_content
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"GET request failed (aiohttp) for {final_url}: {e}")
            return final_url, CONTENT_TYPE_FAILED, None
        except Exception as e: 
             logger.error(f"Unexpected error during GET request (aiohttp) for {final_url}: {e}", exc_info=True)
             return final_url, CONTENT_TYPE_FAILED, None
    else:
         logger.info(f"Skipping content fetch (aiohttp) for non-PDF/HTML type '{content_type_header}' at {final_url}")
         return final_url, CONTENT_TYPE_OTHER, None


# --- Playwright Fetching Logic ---
async def _fetch_with_playwright(
    url: str, playwright_context 
) -> Tuple[str, str, Optional[str], Optional[str]]:
    if not PLAYWRIGHT_AVAILABLE: 
        return url, CONTENT_TYPE_FAILED, None, "Playwright not available"

    page = None; final_url = url; error_message: Optional[str] = None
    html_content: Optional[str] = None; content_type_constant = CONTENT_TYPE_FAILED 

    try:
        page = await playwright_context.new_page()
        await page.set_extra_http_headers({'User-Agent': get_user_agent()})
        logger.debug(f"Navigating to {url} with Playwright...")
        response = await page.goto(url, timeout=FETCH_TIMEOUT_CONFIG_MS, wait_until="domcontentloaded")
        final_url = page.url 
        if response is None: raise PlaywrightError(f"Navigation to {url} returned None response object.")
        status = response.status
        if not (200 <= status < 300): raise PlaywrightError(f"HTTP Error {status} for {final_url}")
        content_type_header = response.headers.get("content-type", "").lower()
        logger.debug(f"Playwright navigation success for {url}. Final URL: {final_url}, Status: {status}, Content-Type: {content_type_header}")
        if "openreview.net" in final_url:
            wait_timeout_dynamic = 15000 
            try:
                await page.wait_for_selector('.note-list .note a[href*="id="]', timeout=wait_timeout_dynamic)
                logger.info(f"Detected OpenReview note list items with links on {final_url}.")
            except PlaywrightTimeoutError: logger.warning(f"Did not find OpenReview note selector within {wait_timeout_dynamic}ms on {final_url}. Content might be incomplete.")
        try:
             await page.wait_for_load_state('networkidle', timeout=5000) 
             logger.debug(f"Network idle state reached for {final_url}")
        except PlaywrightTimeoutError: logger.warning(f"Network did not become idle for {final_url}, proceeding anyway.")
        if 'text/html' in content_type_header:
            html_content = await page.content()
            content_type_constant = CONTENT_TYPE_HTML
            logger.debug(f"Fetched HTML content ({len(html_content) if html_content else 0} chars) for {final_url}")
        else:
            content_type_constant = CONTENT_TYPE_OTHER
            logger.info(f"Skipping content fetch for non-HTML type '{content_type_header}' at {final_url} in Playwright fetch.")
    except PlaywrightTimeoutError as e: error_message = f"Playwright TimeoutError for {url}: {str(e)}"; logger.error(error_message)
    except PlaywrightError as e: error_message = f"Playwright Error for {url}: {str(e)}"; logger.error(error_message)
    except Exception as e: error_message = f"Unexpected error during Playwright fetch for {url}: {str(e)}"; logger.error(error_message, exc_info=True)
    finally:
        if page:
             try: await page.close()
             except Exception as close_err: logger.warning(f"Error closing Playwright page for {url}: {close_err}")
    return final_url, content_type_constant, html_content, error_message


async def _get_parsed_content_orchestrator(
    url: str, playwright_context, session: aiohttp.ClientSession,
    is_pdf_link: bool, proxy: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    final_url = url; content_type_constant = CONTENT_TYPE_FAILED; cleaned_text: Optional[str] = None
    if is_pdf_link:
        logger.info(f"Attempting direct download of potential PDF: {url}")
        try:
            final_url, content_type_constant, pdf_bytes = await _fetch_url_content_aiohttp(url, session, proxy)
            if content_type_constant == CONTENT_TYPE_PDF and pdf_bytes: cleaned_text = _parse_pdf_content(pdf_bytes, final_url)
            elif content_type_constant != CONTENT_TYPE_FAILED: logger.warning(f"Expected PDF but got {content_type_constant} for {url}"); content_type_constant = CONTENT_TYPE_OTHER
        except Exception as pdf_err: logger.error(f"Error directly fetching/parsing PDF {url}: {pdf_err}", exc_info=True); content_type_constant = CONTENT_TYPE_FAILED
    else: 
        try:
            final_url, content_type_constant, html_content, error_msg = await _fetch_with_playwright(url, playwright_context,)
            if error_msg: content_type_constant = CONTENT_TYPE_FAILED
            elif content_type_constant == CONTENT_TYPE_HTML and html_content: cleaned_text = _parse_html_content(html_content, final_url)
        except Exception as html_err: logger.error(f"Error fetching/parsing HTML {url} with Playwright: {html_err}", exc_info=True); content_type_constant = CONTENT_TYPE_FAILED
    return final_url, content_type_constant, cleaned_text


# --- Link Extraction and Filtering ---
def _extract_links(html_string: str, base_url: str) -> List[Tuple[str, str]]:
    discovered_links: List[Tuple[str, str]] = []
    try:
        soup = BeautifulSoup(html_string, 'html.parser')
        for link_tag in soup.find_all('a', href=True):
            if not isinstance(link_tag, bs4_element.Tag): continue
            href_val = link_tag.get('href'); href_str: Optional[str] = None
            if isinstance(href_val, list): href_str = str(href_val[0]).strip() if href_val else None
            elif isinstance(href_val, str): href_str = href_val.strip()
            if not href_str or href_str.startswith(('#', 'javascript:', 'mailto:')): continue
            link_text = link_tag.get_text(strip=True) or "N/A" 
            try:
                absolute_url = urljoin(base_url, href_str); parsed_abs_url = urlparse(absolute_url)
                if parsed_abs_url.scheme and parsed_abs_url.netloc:
                    discovered_links.append((urlunparse(parsed_abs_url._replace(fragment="")), link_text))
            except ValueError: logger.debug(f"Skipping invalid relative URL '{href_str}' found on {base_url}")
        pdf_link_tags = soup.find_all('a', href=re.compile(r'^/(pdf|attachment)\?id='))
        for link_tag in pdf_link_tags:
            if not isinstance(link_tag, bs4_element.Tag): logger.debug(f"Skipping non-Tag element in PDF link extraction: {type(link_tag)}"); continue
            href_attr = link_tag.get('href')
            if not isinstance(href_attr, str): continue 
            href = href_attr.strip(); link_text = link_tag.get_text(strip=True)
            parent_note = link_tag.find_parent('div', class_='note')
            if parent_note and isinstance(parent_note, bs4_element.Tag): 
                title_tag_candidate = parent_note.find('h4')
                if isinstance(title_tag_candidate, bs4_element.Tag):
                    title_text = title_tag_candidate.get_text(strip=True)
                    if title_text: link_text = title_text
                elif title_tag_candidate is not None: logger.warning(f"Expected a Tag for OpenReview title_tag, got {type(title_tag_candidate)}")
            try:
                absolute_url = urljoin(base_url, href); parsed_abs_url = urlparse(absolute_url)
                discovered_links.append((urlunparse(parsed_abs_url._replace(fragment="")), link_text or "PDF Link"))
            except ValueError: logger.debug(f"Skipping invalid OpenReview PDF relative URL '{href}' on {base_url}")
    except Exception as e: logger.error(f"Error extracting links from {base_url}: {e}", exc_info=True)
    return discovered_links

def _is_link_valid_for_queueing(
    link_url: str, visited: BloomFilter, start_domains: Set[Optional[str]], 
    allowed_domains_override: Optional[List[str]] = None 
    ) -> bool:
    if link_url in visited or not link_url.startswith(('http://', 'https://')): return False
    try:
        parsed_link = urlparse(link_url); link_path_lower = parsed_link.path.lower()
        if link_path_lower.endswith(".pdf") or link_path_lower.startswith(('/pdf', '/attachment')): return False 
        if any(link_path_lower.endswith(ext) for ext in SKIPPED_EXTENSIONS): return False
    except Exception: logger.debug(f"Could not parse path for link {link_url}, skipping queue check."); return False
    link_domain = get_domain(link_url)
    if not link_domain: return False
    current_allowed_domains = allowed_domains_override if allowed_domains_override is not None else ALLOWED_DOMAINS_CONFIG
    domain_allowed = False
    if current_allowed_domains: 
        if any(link_domain == allowed or link_domain.endswith('.' + allowed) for allowed in current_allowed_domains): domain_allowed = True
    elif start_domains: 
        if link_domain in start_domains: domain_allowed = True
    return domain_allowed


# --- Dedicated PDF Processing Task ---
async def _process_pdf_link_task(
    pdf_url: str, link_text: str, found_on_url: str, session: aiohttp.ClientSession,
    results_list: List[Dict[str, Any]], visited: BloomFilter, stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore, proxy: Optional[str] = None
):
    global LAST_REQUEST_TIME, MAX_PAGES_TO_CRAWL_CONFIG, CRAWL_DELAY_SECONDS_CONFIG
    if stop_event.is_set() or pdf_url in visited: return
    async with semaphore: 
        if stop_event.is_set() or pdf_url in visited: return
        visited.add(pdf_url)
        loop = asyncio.get_event_loop(); now = loop.time(); time_since_last = now - LAST_REQUEST_TIME
        if time_since_last < CRAWL_DELAY_SECONDS_CONFIG: await asyncio.sleep(CRAWL_DELAY_SECONDS_CONFIG - time_since_last)
        LAST_REQUEST_TIME = loop.time()
        logger.debug(f"Starting PDF processing task for: {pdf_url}")
        try:
            final_pdf_url, pdf_content_type, pdf_bytes = await _fetch_url_content_aiohttp(pdf_url, session, proxy)
            if pdf_content_type == CONTENT_TYPE_PDF and pdf_bytes:
                pdf_text_content = _parse_pdf_content(pdf_bytes, final_pdf_url)
                if pdf_text_content:
                    if len(results_list) < MAX_PAGES_TO_CRAWL_CONFIG:
                        pdf_title = link_text if link_text and link_text != "N/A" else os.path.basename(urlparse(final_pdf_url).path)
                        if not pdf_title: pdf_title = final_pdf_url 
                        results_list.append({
                            "entry_id": final_pdf_url, "source": CONTENT_TYPE_PDF, "url": final_pdf_url, 
                            "title": pdf_title, "content": f"{pdf_title}\\n\\n{pdf_text_content}", 
                            "authors": [], "published": None, "found_on": found_on_url
                        })
                        logger.info(f"Processed PDF: {final_pdf_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL_CONFIG}) Found on: {found_on_url}")
                        if len(results_list) >= MAX_PAGES_TO_CRAWL_CONFIG and not stop_event.is_set():
                            logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL_CONFIG}) reached after PDF. Signalling stop.")
                            stop_event.set()
                    else: 
                         if not stop_event.is_set(): stop_event.set()
                else: logger.warning(f"Failed to parse PDF content from: {final_pdf_url}")
            elif pdf_content_type == CONTENT_TYPE_FAILED: logger.warning(f"Failed to fetch PDF link: {pdf_url}")
            else: logger.warning(f"Expected PDF but got {pdf_content_type} for link: {pdf_url}")
        except Exception as pdf_err: logger.error(f"Error processing PDF link task for {pdf_url}: {pdf_err}", exc_info=True)


# --- Web Crawler Task (Playwright for HTML, spawns PDF tasks) ---
async def _process_crawl_url_pw(
    current_url: str, playwright_context, session: aiohttp.ClientSession, 
    results_list: List[Dict[str, Any]], visited: BloomFilter, queue: deque[str], active_tasks: Set[asyncio.Task],
    start_domains: Set[Optional[str]], visited_set_limit: int, stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore, pdf_semaphore: asyncio.Semaphore, process_pdfs: bool, proxy: Optional[str] = None
):
    global LAST_REQUEST_TIME, MAX_PAGES_TO_CRAWL_CONFIG, CRAWL_DELAY_SECONDS_CONFIG
    if stop_event.is_set() or current_url in visited: return
    loop = asyncio.get_event_loop(); now = loop.time(); time_since_last = now - LAST_REQUEST_TIME
    if time_since_last < CRAWL_DELAY_SECONDS_CONFIG: await asyncio.sleep(CRAWL_DELAY_SECONDS_CONFIG - time_since_last)
    LAST_REQUEST_TIME = loop.time()
    visited.add(current_url) 
    html_content_for_links: Optional[str] = None; final_url = current_url
    try:
        final_url, content_type, html_content_for_links, error_msg = await _fetch_with_playwright(current_url, playwright_context,)
        if final_url != current_url: 
             if final_url in visited: return
             visited.add(final_url)
        if error_msg: logger.warning(f"Playwright fetch failed for {current_url}: {error_msg}"); return 
        if content_type == CONTENT_TYPE_HTML and html_content_for_links:
            text_content = _parse_html_content(html_content_for_links, final_url)
            if text_content:
                 if len(results_list) < MAX_PAGES_TO_CRAWL_CONFIG:
                     page_title = final_url 
                     try: 
                         soup_title = BeautifulSoup(html_content_for_links, 'html.parser')
                         title_tag = soup_title.find('title')
                         if title_tag:
                             title_text = title_tag.get_text(strip=True)
                             if title_text: page_title = title_text
                     except Exception: pass 
                     if not page_title or page_title == final_url: 
                          first_line = text_content.split('\n', 1)[0].strip()
                          if first_line and len(first_line) < 150: page_title = first_line 
                     results_list.append({"entry_id": final_url, 
                                    "source": content_type, "url": final_url, "title": page_title,
                                    "content": text_content, "authors": [], "published": None})
                     logger.info(f"Processed HTML: {final_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL_CONFIG})")
                     if len(results_list) >= MAX_PAGES_TO_CRAWL_CONFIG and not stop_event.is_set():
                         logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL_CONFIG}) reached. Signalling stop.")
                         stop_event.set()
                 else: 
                      if not stop_event.is_set(): stop_event.set()
            else: logger.warning(f"Failed to parse text content from HTML: {final_url}")
            if not stop_event.is_set():
                discovered_links = _extract_links(html_content_for_links, final_url)
                for link_url_abs, link_text_hint in discovered_links:
                    if stop_event.is_set(): break
                    if link_url_abs in visited: continue
                    if len(visited) >= visited_set_limit and not stop_event.is_set(): 
                        logger.warning("Bloom filter capacity reached. Stopping new link processing.")
                        stop_event.set(); break
                    parsed_link_path = urlparse(link_url_abs).path.lower()
                    is_potential_pdf = parsed_link_path.endswith(".pdf") or parsed_link_path.startswith(('/pdf', '/attachment'))
                    if process_pdfs and is_potential_pdf:
                        pdf_task = asyncio.create_task(_process_pdf_link_task(
                                pdf_url=link_url_abs, link_text=link_text_hint, found_on_url=final_url,
                                session=session, results_list=results_list, visited=visited,
                                stop_event=stop_event, semaphore=pdf_semaphore, proxy=proxy 
                            ))
                        active_tasks.add(pdf_task); pdf_task.add_done_callback(active_tasks.discard)
                    elif not is_potential_pdf: 
                        if _is_link_valid_for_queueing(link_url_abs, visited, start_domains):
                            queue.append(link_url_abs)
    except Exception as e: logger.error(f"Processing {current_url} failed unexpectedly in task: {e}", exc_info=True)


# --- Main Crawler Orchestration ---
async def _crawl_manager_pw(
    start_urls: List[str], playwright_context, session: aiohttp.ClientSession, 
    html_semaphore: asyncio.Semaphore, pdf_semaphore: asyncio.Semaphore,  
    process_pdfs: bool, proxies: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    global MAX_PAGES_TO_CRAWL_CONFIG
    results_list: List[Dict[str, Any]] = []; estimated_capacity = MAX_PAGES_TO_CRAWL_CONFIG * (50 if process_pdfs else 20)
    visited: BloomFilter = BloomFilter(capacity=max(estimated_capacity, 5000), error_rate=0.001)
    queue: deque[str] = deque()
    for url in start_urls: 
        if url not in visited: queue.append(url)
    start_domains: Set[Optional[str]] = {get_domain(url) for url in start_urls if get_domain(url)}
    visited_set_limit = visited.capacity; stop_event = asyncio.Event(); active_tasks: Set[asyncio.Task] = set()
    logger.info(f"Crawl Manager started. Initial Queue: {len(queue)}. Visited Capacity: {visited.capacity}. Max Items: {MAX_PAGES_TO_CRAWL_CONFIG}")
    while (queue or active_tasks) and not stop_event.is_set():
        while queue and len(active_tasks) < html_semaphore._value + pdf_semaphore._value and not stop_event.is_set(): 
            if html_semaphore.locked(): await asyncio.sleep(0.1); continue
            current_url = queue.popleft()
            if current_url in visited: continue
            await html_semaphore.acquire(); proxy_to_use = random.choice(proxies) if proxies else None
            task = asyncio.create_task(_process_crawl_url_pw(
                    current_url=current_url, playwright_context=playwright_context, session=session,
                    results_list=results_list, visited=visited, queue=queue, active_tasks=active_tasks,
                    start_domains=start_domains, visited_set_limit=visited_set_limit,
                    stop_event=stop_event, semaphore=html_semaphore, 
                    pdf_semaphore=pdf_semaphore, process_pdfs=process_pdfs, proxy=proxy_to_use
                ))
            active_tasks.add(task)
            def html_task_done_callback(fut, url=current_url, sem=html_semaphore):
                try: fut.result() 
                except asyncio.CancelledError: logger.debug(f"HTML task for {url} was cancelled.")
                except Exception as task_exc: logger.error(f"HTML task for {url} raised: {task_exc}", exc_info=False)
                finally: sem.release(); active_tasks.discard(fut)
                logger.debug(f"HTML task for {url} finished. Released HTML semaphore. Active tasks: {len(active_tasks)}")
            task.add_done_callback(html_task_done_callback)
        if not active_tasks and not queue and not stop_event.is_set(): break
        if not active_tasks and queue and not stop_event.is_set(): await asyncio.sleep(0.01); continue
        logger.debug(f"Waiting for tasks. Active: {len(active_tasks)}, Queue: {len(queue)}, Stop: {stop_event.is_set()}")
        if active_tasks:
            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=5.0)
            if not done and not stop_event.is_set() and not queue: logger.debug("Task wait timed out, but work might still be pending or queue might refill.")
        else: await asyncio.sleep(0.1)
    if active_tasks: 
        logger.info(f"Crawling loop ended. Cancelling {len(active_tasks)} remaining tasks...")
        for task_to_cancel in list(active_tasks): task_to_cancel.cancel()
        try: await asyncio.gather(*active_tasks, return_exceptions=True); logger.info("All remaining tasks cancelled/completed.")
        except Exception as e: logger.error(f"Error during final task cleanup: {e}")
    logger.info(f"Crawling finished. Processed {len(results_list)} total items. Queue size: {len(queue)}.")
    return results_list


# --- Public API Function ---
async def crawl_and_fetch_web_articles(
    start_urls: List[str], process_pdfs_linked: bool = True, 
    max_pages_override: Optional[int] = None, proxies: Optional[List[str]] = None,
    max_concurrent_html: int = 5, max_concurrent_pdf: int = 10  
    ) -> List[Dict[str, Any]]:
    if not PLAYWRIGHT_AVAILABLE: logger.error("Playwright is not available. Cannot web crawl."); return []
    if not start_urls: logger.info("No starting URLs for crawling."); return []
    global MAX_PAGES_TO_CRAWL_CONFIG 
    original_max_pages = MAX_PAGES_TO_CRAWL_CONFIG
    if max_pages_override is not None and max_pages_override > 0:
        logger.info(f"Overriding MAX_PAGES_TO_CRAWL ({original_max_pages}) with {max_pages_override}")
        MAX_PAGES_TO_CRAWL_CONFIG = max_pages_override
    logger.info(f"Starting web crawl. HTML Concurrency: {max_concurrent_html}, PDF Concurrency: {max_concurrent_pdf}. Max Items: {MAX_PAGES_TO_CRAWL_CONFIG}. Process PDFs: {process_pdfs_linked}.")
    html_semaphore = asyncio.Semaphore(max_concurrent_html); pdf_semaphore = asyncio.Semaphore(max_concurrent_pdf)
    results: List[Dict[str, Any]] = []; playwright_instance = None; browser = None; context = None; aiohttp_session = None
    try:
        playwright_instance = await async_playwright().start()
        browser = await playwright_instance.chromium.launch(headless=True) 
        context = await browser.new_context(user_agent=get_user_agent(), ignore_https_errors=True)
        connector = aiohttp.TCPConnector(limit_per_host=max(5, max_concurrent_pdf // 2), limit=max(10, max_concurrent_pdf * 2), ssl=False) 
        aiohttp_session = aiohttp.ClientSession(connector=connector)
        results = await _crawl_manager_pw(
            start_urls=start_urls, playwright_context=context, session=aiohttp_session,
            html_semaphore=html_semaphore, pdf_semaphore=pdf_semaphore,
            process_pdfs=process_pdfs_linked, proxies=proxies
        )
        logger.info(f"Web crawl finished. Returning {len(results)} processed items.")
    except Exception as e: logger.error(f"Web crawl failed critically: {e}", exc_info=True)
    finally:
        if aiohttp_session: await aiohttp_session.close()
        if context: await context.close()
        if browser: await browser.close()
        if playwright_instance: await playwright_instance.stop()
        if max_pages_override is not None: MAX_PAGES_TO_CRAWL_CONFIG = original_max_pages
        logger.info("Web crawling resources cleaned up.")
    return results


# --- arXiv Fetcher ---
async def _fetch_pdf_content_with_playwright(pdf_url: str, playwright_context, verbose: bool = False) -> Optional[str]:
    if not playwright_context:
        logger.warning(f"Playwright context not available, cannot fetch PDF from {pdf_url}")
        return None

    if verbose:
        logger.info(f"🔄 PDF Fetch: Attempting for {pdf_url}")
            
    page = None
    try:
        page = await playwright_context.new_page()
        if verbose:
            logger.info(f"🔄 PDF Fetch: Created new page for {pdf_url}")
        
        response = await page.goto(pdf_url, timeout=60000, wait_until="load") 
        
        if not response:
            logger.error(f"Playwright navigation to {pdf_url} returned no response.")
            if page: await page.close()
            return None

        if response.status != 200:
            logger.error(f"PDF URL {pdf_url} returned status {response.status} (Final URL: {response.url})")
            if page: await page.close()
            return None
        
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" not in content_type:
            logger.warning(f"Expected PDF content-type for {pdf_url} (Final URL: {response.url}), but got '{content_type}'.")
            # Optionally, still try to get body if it might be a PDF served with wrong CT
            # For now, let's be strict for direct PDF links.
            # If it's an HTML page that *links* to a PDF, this function isn't for that.
            if page: await page.close()
            return None 

        if verbose:
            logger.info(f"🔄 PDF Fetch: Successfully navigated to {response.url}. Status: {response.status}. Content-Type: {content_type}")

        pdf_bytes = await response.body()
        
        if not pdf_bytes:
            logger.warning(f"No PDF content bytes retrieved from {response.url}")
            if page: await page.close()
            return None
            
        if verbose:
            logger.info(f"✅ PDF Fetch: Got {len(pdf_bytes)} PDF bytes for {response.url}. Parsing with PyMuPDF...")
                
        pdf_doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") # type: ignore
        text_parts: List[str] = [] # Initialize list for text parts
        page_count = pdf_doc.page_count
        
        if verbose:
            logger.info(f"📄 PDF Fetch: Processing {page_count} pages from {response.url}")
        
        for page_num in range(page_count):
            if verbose and page_num % 10 == 0 and page_num > 0:
                logger.info(f"📄 PDF Fetch: Processed {page_num}/{page_count} pages from {response.url}")
            current_page = pdf_doc.load_page(page_num) # Use current_page variable
            page_text = current_page.get_text() # Get text from current_page
            text_parts.append(page_text) # Append to list

        pdf_doc.close()
        
        full_text = "\n\n".join(text_parts) # Join parts with double newline
        cleaned_text = _clean_text(full_text) 

        if verbose:
            text_length = len(cleaned_text) if cleaned_text else 0
            logger.info(f"✅ PDF Fetch: Successfully extracted {text_length} characters from {page_count} pages for {response.url}")
            
        if page: await page.close() 
        return cleaned_text 
            
    except Exception as e:
        # Import PlaywrightTimeoutError locally for specific handling if not already global
        if PLAYWRIGHT_AVAILABLE:
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError_Local # type: ignore
            if isinstance(e, PlaywrightTimeoutError_Local):
                logger.error(f"Playwright TimeoutError accessing PDF {pdf_url}: {e}", exc_info=False) 
            else:
                logger.error(f"Error accessing/processing PDF {pdf_url} with Playwright: {e}", exc_info=True)
        else: # Fallback if Playwright types aren't available for isinstance
            logger.error(f"Error accessing/processing PDF {pdf_url} with Playwright: {e}", exc_info=True)

        if verbose:
            logger.error(f"❌ PDF Fetch: Failed for {pdf_url}. Error: {str(e)}")
        if page:
            try:
                await page.close()
            except Exception as close_err:
                logger.warning(f"Error closing page during exception handling for {pdf_url}: {close_err}")
        return None

async def fetch_arxiv_papers(
    query: str, max_results: int = 10, days_back: Optional[int] = None,
    sort_by: str = "relevance", session: Optional[aiohttp.ClientSession] = None,
    proxy: Optional[str] = None, fetch_pdfs: bool = False,
    playwright_context = None, verbose: bool = False 
) -> List[Dict[str, Any]]:
    base_url = "http://export.arxiv.org/api/query"
    if verbose: logger.info(f"🔄 ArXiv Fetch: query='{query}', max_results={max_results}, fetch_pdfs={fetch_pdfs}")
    search_query = query
    if days_back is not None:
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            search_query += f" AND submittedDate:[{cutoff_date.strftime('%Y%m%d%H%M%S')} TO *]" # Removed * before TO
            if verbose: logger.info(f"📅 ArXiv Fetch: Date filter for last {days_back} days added: submittedDate:[{cutoff_date.strftime('%Y%m%d%H%M%S')} TO *]")
        except Exception as date_e: logger.warning(f"Failed to add date filter: {date_e}")
    
    sort_order_val = "descending" # Common default, can be 'ascending'
    sort_by_val = "relevance"
    if sort_by == "lastUpdatedDate":
        sort_by_val = "lastUpdatedDate"
    elif sort_by == "submittedDate":
        sort_by_val = "submittedDate"

    params = {'search_query': search_query, 'start': 0, 'max_results': max_results, 
              'sortBy': sort_by_val, 'sortOrder': sort_order_val}
    
    session_created = False
    if session is None:
        connector = None # type: ignore
        if proxy: connector = aiohttp.TCPConnector(ssl=False if proxy.startswith("http://") else None) # type: ignore # Basic proxy SSL handling
        session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=180, connect=30, sock_read=60))
        session_created = True
    
    papers_list = []
    try:
        if verbose: logger.info(f"🌐 ArXiv Fetch: API request to {base_url} with params {params}")
        request_kwargs: Dict[str,Any] = {} # type: ignore
        if proxy: request_kwargs['proxy'] = proxy
        
        async with session.get(base_url, params=params, **request_kwargs) as response: # type: ignore
            if response.status == 200:
                xml_content = await response.text()
                if verbose: logger.info(f"✅ ArXiv Fetch: API response OK ({len(xml_content)} chars)")
                root = ET.fromstring(xml_content)
                entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                logger.info(f"[fetch_arxiv_papers] Found {len(entries)} XML entries from arXiv API for query '{query}'. Intended max_results: {max_results}")
                
                for i, entry in enumerate(entries):
                    logger.info(f"[fetch_arxiv_papers] Processing XML entry {i+1}/{len(entries)}")
                    try:
                        title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                        title = title_elem.text.strip() if (title_elem is not None and title_elem.text is not None) else "N/A"
                        
                        summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                        summary = summary_elem.text.strip() if (summary_elem is not None and summary_elem.text is not None) else ""
                        
                        published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                        published = published_elem.text.strip() if (published_elem is not None and published_elem.text is not None) else ""
                        
                        updated_elem = entry.find('.//{http://www.w3.org/2005/Atom}updated')
                        updated = updated_elem.text.strip() if (updated_elem is not None and updated_elem.text is not None) else ""
                        
                        id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
                        entry_id_val = ""; pdf_url = "" # Renamed entry_id to entry_id_val
                        if id_elem is not None and id_elem.text is not None:
                            entry_id_val = id_elem.text.strip()
                            arxiv_id_match = re.search(r'arxiv.org/(?:abs|pdf)/([^vV]+)', entry_id_val) # More robust regex for ID
                            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else entry_id_val.split('/')[-1]
                            pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
                        
                        authors_elems = entry.findall('.//{http://www.w3.org/2005/Atom}author')
                        authors = []
                        for auth_elem in authors_elems:
                            name_elem = auth_elem.find('.//{http://www.w3.org/2005/Atom}name')
                            if name_elem is not None and name_elem.text is not None:
                                authors.append(name_elem.text.strip())
                        
                        category_elems = entry.findall('.//{http://www.w3.org/2005/Atom}category')
                        categories = [cat.get('term') for cat in category_elems if cat.get('term')]
                        
                        paper_data = {
                            'title': title, 'authors': authors, 
                            'published': published.split('T')[0] if 'T' in published else published, 
                            'updated': updated.split('T')[0] if 'T' in updated else updated,
                            'entry_id': entry_id_val, 'pdf_url': pdf_url, 'url': entry_id_val, 
                            'summary': summary.replace('\n', ' ').strip(), 
                            'categories': categories, 'source': 'arxiv'
                        }
                        content = ""
                        if fetch_pdfs and pdf_url and playwright_context:
                            if verbose: logger.info(f"📄 ArXiv Fetch: PDF fetch for {title[:50]}... (Entry {i+1})")
                            try:
                                pdf_content = await _fetch_pdf_content_with_playwright(pdf_url, playwright_context, verbose=verbose)
                                if pdf_content: content = pdf_content
                                if verbose: 
                                    logger.info(f"✅ ArXiv Fetch: PDF content for {title[:50]}... (Entry {i+1}, Len: {len(content)})") if content else logger.warning(f"⚠️ ArXiv Fetch: PDF empty for {title[:50]}... (Entry {i+1})")
                            except Exception as pdf_e: logger.warning(f"PDF fetch failed for {title} (Entry {i+1}): {pdf_e}")
                        
                        paper_data['content'] = content
                        papers_list.append(paper_data)
                        logger.info(f"[fetch_arxiv_papers] Appended paper_data for entry {i+1}. List size: {len(papers_list)}")
                            
                    except Exception as entry_e: logger.warning(f"Failed to parse arXiv XML entry {i+1}: {entry_e}", exc_info=True) # Added exc_info
                
                if verbose: logger.info(f"✅ ArXiv Fetch: Processed {len(papers_list)} papers from XML.")
            else: logger.error(f"arXiv API request failed with status {response.status}")
    
    except Exception as e: logger.error(f"Error fetching arXiv papers: {e}", exc_info=True)
    finally:
        if session_created and session: await session.close()
    
    logger.info(f"[fetch_arxiv_papers] Returning {len(papers_list)} papers from fetch_arxiv_papers.")
    return papers_list