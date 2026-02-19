"""
RAG from Scratch
================
A research paper Q&A system built piece by piece.

Modules:
  1. pdf_parser  — PDF ingestion, section detection, text cleaning
  2. chunkers    — Three chunking strategies (fixed, semantic, hierarchical)
  3. compare     — Run & compare all strategies on a paper

Usage:
  uv run rag-parse paper.pdf        # parse & show structure
  uv run rag-chunk paper.pdf        # chunk with all strategies & compare
"""