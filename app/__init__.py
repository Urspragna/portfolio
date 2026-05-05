"""Pragna's AI Portfolio — FastAPI backend.

A production-flavoured Python backend that powers the portfolio's AI features:
  - /api/chat           : RAG-grounded chatbot over Pragna's CV
  - /api/demo/sentiment : sentiment analysis (HF transformers)
  - /api/demo/embed     : embedding similarity tool
  - /api/demo/summarise : abstractive summarisation
  - /api/health         : healthcheck

Entry point: app.main:app
"""
__version__ = "1.0.0"
