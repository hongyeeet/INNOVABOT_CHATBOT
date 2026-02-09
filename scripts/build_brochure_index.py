"""
build_brochure_index.py

Builds and saves FAISS (semantic) and BM25 (lexical) indexes from Company_Brochure.pdf.

This script should be run:
- Once initially to create indexes
- Whenever Company_Brochure.pdf is updated
- If chunking parameters (chunk_size, overlap) are changed

Outputs:
- data/brochure_chunks.pkl      - Text chunks from PDF
- data/brochure_faiss.index     - FAISS semantic vector index
- data/brochure_bm25.pkl        - BM25 keyword index
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
from rank_bm25 import BM25Okapi

# Configuration
BROCHURE_PDF_PATH = "data/Company_Brochure.pdf"
BROCHURE_FAISS_INDEX_PATH = "data/brochure_faiss.index"
BROCHURE_CHUNKS_PATH = "data/brochure_chunks.pkl"
BROCHURE_BM25_PATH = "data/brochure_bm25.pkl"

CHUNK_SIZE = 1500
OVERLAP = 300
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_pdf(pdf_path: str) -> str:
    """
    Load and extract text from PDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Concatenated text from all pages
    """
    print(f"📄 Loading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages)
    print(f"   ✓ Extracted {len(full_text)} characters from {len(pages)} pages")
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between consecutive chunks
    
    Returns:
        List of text chunks
    """
    print(f"✂️  Chunking text (size={chunk_size}, overlap={overlap})")
    
    if overlap >= chunk_size:
        raise ValueError(f"Overlap ({overlap}) must be < chunk_size ({chunk_size})")
    
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        
        if not chunk.strip():
            break
        
        chunks.append(chunk)
        
        # Move forward
        next_start = start + (chunk_size - overlap)
        if next_start <= start:
            break
        start = next_start
    
    print(f"   ✓ Created {len(chunks)} chunks")
    return chunks


def build_faiss_index(chunks: list, model_name: str = EMBEDDING_MODEL):
    """
    Build FAISS semantic index from text chunks.
    
    Args:
        chunks: List of text chunks
        model_name: SentenceTransformer model name
    
    Returns:
        Tuple of (faiss_index, embeddings)
    """
    print(f"🧠 Building FAISS index with model: {model_name}")
    
    # Load embedding model
    embedder = SentenceTransformer(model_name)
    
    # Generate embeddings
    print(f"   → Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    # Normalize embeddings (required for cosine similarity with IndexFlatIP)
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"   ✓ FAISS index built ({index.ntotal} vectors, dim={dimension})")
    return index, embeddings


def build_bm25_index(chunks: list):
    """
    Build BM25 keyword index from text chunks.
    Args:
        chunks: List of text chunks
    
    Returns:
        BM25Okapi index
    """
    print(f"🔤 Building BM25 index")
    # Tokenize chunks (simple whitespace tokenization)
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_chunks)
    print(f"   ✓ BM25 index built ({len(chunks)} documents)")
    return bm25


def save_indexes(chunks: list, faiss_index, bm25_index):
    """
    Save all indexes to disk.
    
    Args:
        chunks: Text chunks
        faiss_index: FAISS index
        bm25_index: BM25 index
    """
    print(f"\n💾 Saving indexes to disk...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save chunks
    with open(BROCHURE_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"   ✓ Saved chunks to {BROCHURE_CHUNKS_PATH}")
    
    # Save FAISS index
    faiss.write_index(faiss_index, BROCHURE_FAISS_INDEX_PATH)
    print(f"   ✓ Saved FAISS index to {BROCHURE_FAISS_INDEX_PATH}")
    
    # Save BM25 index
    with open(BROCHURE_BM25_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"   ✓ Saved BM25 index to {BROCHURE_BM25_PATH}")


def main():
    """
    Main function to build all indexes.
    """
    print("=" * 80)
    print("BUILDING BROCHURE INDEXES (FAISS + BM25)")
    print("=" * 80)
    print()
    
    # Check if PDF exists
    if not os.path.exists(BROCHURE_PDF_PATH):
        print(f"❌ ERROR: PDF not found at {BROCHURE_PDF_PATH}")
        return
    
    # Step 1: Load PDF
    text = load_pdf(BROCHURE_PDF_PATH)
    print()
    
    # Step 2: Chunk text
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print()
    
    # Step 3: Build FAISS index
    faiss_index, embeddings = build_faiss_index(chunks, model_name=EMBEDDING_MODEL)
    print()
    
    # Step 4: Build BM25 index
    bm25_index = build_bm25_index(chunks)
    print()
    
    # Step 5: Save everything
    save_indexes(chunks, faiss_index, bm25_index)
    
    print()
    print("=" * 80)
    print("✅ INDEX BUILDING COMPLETE")
    print("=" * 80)
    print()
    print("Created files:")
    print(f"  • {BROCHURE_CHUNKS_PATH}")
    print(f"  • {BROCHURE_FAISS_INDEX_PATH}")
    print(f"  • {BROCHURE_BM25_PATH}")
    print()
    print("You can now use these indexes for retrieval!")


if __name__ == "__main__":
    main()
