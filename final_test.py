#!/usr/bin/env python3
"""
Final test script to verify the arXiv PDF fetching fix is complete and working.
"""

import asyncio
import logging
from hybrid_search_rag.data_handling.resource_fetcher import fetch_arxiv_papers

# Set up concise logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

async def final_test():
    """Final test to confirm the fix works correctly."""
    try:
        print('🔬 Final Test: arXiv PDF fetching with direct HTTP')
        print('=' * 50)
        
        papers = await fetch_arxiv_papers(
            query='machine learning',
            max_results=1,
            fetch_pdfs=True,
            verbose=False  # Reduced verbosity for cleaner output
        )
        
        if papers and len(papers) > 0:
            paper = papers[0]
            title = paper.get("title", "N/A")
            content_length = len(paper.get("content", ""))
            pdf_url = paper.get("pdf_url", "N/A")
            
            print(f'✅ SUCCESS: Paper fetched and processed')
            print(f'📄 Title: {title[:60]}...' if len(title) > 60 else f'📄 Title: {title}')
            print(f'📊 Content extracted: {content_length:,} characters')
            print(f'🔗 PDF URL: {pdf_url}')
            
            if content_length > 1000:
                print('🎉 COMPLETE: PDF content successfully extracted!')
                print('✅ The net::ERR_ABORTED error has been FIXED!')
                return True
            else:
                print('⚠️  Warning: PDF content seems too short')
                return False
                
        else:
            print('❌ FAILED: No papers returned')
            return False
            
    except Exception as e:
        print(f'❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(final_test())
    if success:
        print('\n🚀 Fix verification COMPLETED successfully!')
    else:
        print('\n💥 Fix verification FAILED!')
