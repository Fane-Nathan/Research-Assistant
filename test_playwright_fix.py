#!/usr/bin/env python3
"""
Test script to verify that the Playwright sync API warning has been fixed.
This script simulates the same usage pattern as the CLI script and Streamlit app.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_search_rag.data_handling.resource_fetcher import ResourceFetcher, fetch_arxiv_papers

# Configure logging to see all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def test_playwright_fix():
    """Test ResourceFetcher initialization and usage in async context."""
    logger = logging.getLogger(__name__)
    logger.info("Testing ResourceFetcher in async context...")
    
    # This should not produce the Playwright sync API warning
    logger.info("Creating ResourceFetcher instance with use_playwright_for_pdfs=True...")
    resource_fetcher = ResourceFetcher(use_playwright_for_pdfs=True)
    
    try:
        # Test fetching arXiv metadata
        logger.info("Fetching arXiv metadata...")
        papers = await fetch_arxiv_papers("quantum computing", max_results=2)
        
        if papers and papers[0].get('pdf_url'):
            pdf_url = papers[0]['pdf_url']
            logger.info(f"Testing PDF content fetching for: {pdf_url}")
            
            # This should use aiohttp, not Playwright, and should work without warnings
            result = await resource_fetcher.fetch_document(
                pdf_url, 
                source="arxiv",
                is_arxiv_pdf_link=True
            )
            
            if result and result.get('text'):
                logger.info(f"Successfully fetched PDF content: {len(result['text'])} characters")
            else:
                logger.warning("Failed to fetch PDF content")
        else:
            logger.warning("No PDF URL found in fetched papers")
      except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    
    finally:
        # Clean up
        resource_fetcher.close()
        logger.info("Test completed.")

if __name__ == '__main__':
    print("Running Playwright fix test...")
    asyncio.run(test_playwright_fix())
    print("Test finished. Check the logs above for any Playwright sync API warnings.")
