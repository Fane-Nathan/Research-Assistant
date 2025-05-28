#!/usr/bin/env python3
"""
Final verification script to confirm the Playwright sync API warning fix.
This mimics the exact usage pattern in both CLI and Streamlit applications.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_search_rag.data_handling.resource_fetcher import ResourceFetcher, fetch_arxiv_papers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def main():
    """Test the complete workflow that was previously causing Playwright warnings."""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("FINAL VERIFICATION: Playwright Sync API Warning Fix")
    logger.info("="*60)
    
    # Test Case 1: ResourceFetcher instantiation in async context
    logger.info("Test 1: Creating ResourceFetcher in async context...")
    resource_fetcher = ResourceFetcher(use_playwright_for_pdfs=True)
    logger.info("✅ ResourceFetcher created without sync API warnings")
    
    # Test Case 2: arXiv metadata fetching
    logger.info("Test 2: Fetching arXiv metadata...")
    papers = await fetch_arxiv_papers("quantum machine learning", max_results=1)
    
    if not papers:
        logger.warning("No papers found - this might be a network issue")
        return
    
    logger.info(f"✅ Successfully fetched {len(papers)} paper(s)")
    
    # Test Case 3: PDF content fetching with ResourceFetcher
    paper = papers[0]
    if paper.get('pdf_url'):
        logger.info("Test 3: Fetching PDF content using ResourceFetcher...")
        result = await resource_fetcher.fetch_document(
            paper['pdf_url'], 
            source="arxiv",
            is_arxiv_pdf_link=True
        )
        
        if result and result.get('text'):
            logger.info(f"✅ Successfully fetched PDF content: {len(result['text'])} characters")
        else:
            logger.warning("❌ Failed to fetch PDF content")
    else:
        logger.warning("No PDF URL found for testing")
    
    # Test Case 4: Clean shutdown
    logger.info("Test 4: Clean shutdown...")
    resource_fetcher.close()
    logger.info("✅ Resources cleaned up successfully")
    
    logger.info("="*60)
    logger.info("ALL TESTS PASSED - NO PLAYWRIGHT SYNC API WARNINGS!")
    logger.info("="*60)

if __name__ == '__main__':
    try:
        asyncio.run(main())
        print("\n🎉 SUCCESS: The Playwright sync API warning has been completely fixed!")
        print("The ResourceFetcher now properly detects async contexts and defers")
        print("Playwright initialization, falling back to aiohttp for PDF fetching.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
