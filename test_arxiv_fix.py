#!/usr/bin/env python3
"""
Test script to verify that the arXiv PDF fetching fix works correctly.
"""

import asyncio
import logging
from hybrid_search_rag.data_handling.resource_fetcher import fetch_arxiv_papers

# Set up logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_arxiv_fetch():
    """Test the fixed arXiv PDF fetching functionality."""
    try:
        print('🧪 Testing arXiv PDF fetching with direct HTTP...')
        print('=' * 60)
        
        papers = await fetch_arxiv_papers(
            query='quantum machine learning',
            max_results=1,
            fetch_pdfs=True,
            verbose=True
        )
        
        print('=' * 60)
        
        if papers:
            paper = papers[0]
            print(f'✅ Successfully fetched: {paper.get("title", "N/A")}')
            print(f'📄 Content length: {len(paper.get("content", ""))} characters')
            print(f'🔗 PDF URL: {paper.get("pdf_url", "N/A")}')
            
            # Check if PDF content was actually fetched
            content = paper.get("content", "")
            if content and len(content) > 100:
                print(f'📝 Content preview: {content[:200]}...')
                print('✅ PDF content successfully extracted!')
            else:
                print('⚠️  No PDF content found or content too short')
                
        else:
            print('❌ No papers returned')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_arxiv_fetch())
