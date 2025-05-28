#!/usr/bin/env python3
"""
Test script to verify that the arXiv PDF fetching fix works correctly.
"""

import asyncio
import logging
from hybrid_search_rag.data_handling.resource_fetcher import fetch_arxiv_papers, ResourceFetcher

# Set up logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_arxiv_fetch():
    """Test the fixed arXiv PDF fetching functionality."""
    try:
        print(' Testing arXiv PDF fetching with direct HTTP...')
        print('=' * 60)
        
        # Step 1: Fetch metadata only
        papers_metadata = await fetch_arxiv_papers(
            query='quantum machine learning',
            max_results=1,
            verbose=True
        )
        
        print('=' * 60)
        
        if papers_metadata:
            paper_meta = papers_metadata[0]
            print(f' Successfully fetched metadata for: {paper_meta.get("title", "N/A")}')
            print(f' PDF URL: {paper_meta.get("pdf_url", "N/A")}')

            pdf_url = paper_meta.get("pdf_url")
            if pdf_url:
                # Step 2: Use ResourceFetcher to get the document content
                resource_fetcher = ResourceFetcher()
                try:
                    print(f' Attempting to fetch content from: {pdf_url}')
                    # fetch_document is an async method, specify it's an arXiv PDF
                    document_content = await resource_fetcher.fetch_document(
                        url=pdf_url, 
                        source='arxiv', 
                        is_arxiv_pdf_link=True
                    )
                    
                    if document_content:
                        # document_content is a dict with 'text' key
                        text_content = document_content.get('text', '')
                        paper_meta["content"] = text_content
                        print(f' Content length: {len(text_content)} characters')
                        
                        # Check if PDF content was actually fetched
                        if text_content and len(text_content) > 100:
                            print(f' Content preview: {text_content[:200]}...')
                            print(' PDF content successfully extracted!')
                        else:
                            print('  No PDF content found or content too short after fetching.')
                    else:
                        print(f' Failed to fetch document content from {pdf_url}')
                        
                except Exception as e_fetch:
                    print(f' Error during document fetching: {e_fetch}')
                    import traceback
                    traceback.print_exc()
                finally:
                    resource_fetcher.close() # close is a synchronous method
                    print("ResourceFetcher closed.")
            else:
                print(' No PDF URL found for this paper.')
                
        else:
            print(' No papers returned')
            
    except Exception as e:
        print(f' Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_arxiv_fetch())
