\
import hashlib
import json
from typing import List, Union, Any

def calculate_document_hash(
    content: str,
    title: str,
    authors: Union[List[Any], str], 
    abstract: str = ""
) -> str:
    """
    Calculates a SHA256 hash for a document based on its title, authors, content, and abstract.
    Normalizes inputs to ensure consistent hashing.
    """
    authors_list: List[str] = []
    if isinstance(authors, str):
        try:
            # Attempt to parse if it's a JSON-like string list
            parsed_authors = json.loads(authors.replace("'", "\""))
            if isinstance(parsed_authors, list):
                authors_list = sorted([str(author).strip().lower() for author in parsed_authors if author is not None])
            else:
                authors_list = [str(parsed_authors).strip().lower()] if parsed_authors is not None else []
        except json.JSONDecodeError:
            # If not a JSON list, treat as a single author name or comma-separated string
            authors_list = sorted([a.strip().lower() for a in authors.split(',') if a.strip()])
    elif isinstance(authors, list):
        authors_list = sorted([str(author).strip().lower() for author in authors if author is not None])

    normalized_title = str(title).strip().lower() if title is not None else ""
    normalized_content = str(content).strip().lower() if content is not None else ""
    normalized_abstract = str(abstract).strip().lower() if abstract is not None else ""

    hash_parts = [
        f"title:{normalized_title}",
        f"authors:{','.join(authors_list)}", # Join sorted authors
        f"content:{normalized_content}"
    ]
    if normalized_abstract: # Only include abstract if it's non-empty
        hash_parts.append(f"abstract:{normalized_abstract}")

    hash_input_str = "|".join(hash_parts)

    return hashlib.sha256(hash_input_str.encode('utf-8')).hexdigest()
