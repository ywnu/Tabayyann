# ØªØ¨ÙŠÙ‘Ù†Ù‘ | Tabayyann

**Tabayyann** is an Arabic grounded Islamic Q&A system built for reliable, source-based answers.

The system uses Retrieval-Augmented Generation (RAG) to retrieve evidence from an approved local knowledge base before generating an answer. It is designed to avoid unsupported answers when sufficient evidence is not available.

## Live Demo

https://tabayyann-production.up.railway.app

## Main Features

- Arabic Islamic question answering
- Retrieval-Augmented Generation (RAG)
- Hybrid semantic and lexical retrieval
- Evidence-grounded answers
- Inline citations and source references
- Answer Policy for hallucination prevention
- Refusal when evidence is insufficient
- Query guard and scope checking
- Arabic voice input
- Speech-to-Text using ElevenLabs Scribe
- Arabic Text-to-Speech using ElevenLabs
- Responsive Arabic web interface
- FastAPI backend
- Railway deployment

## Architecture

User Question  
â†“  
Query Guard  
â†“  
Hybrid Retriever  
â†“  
Grounded Evidence Preparation  
â†“  
Answer Policy  
â†“  
Claude  
â†“  
Validated Answer + Citations

## Main Project Files

- `main.py` â€” Railway application entry point
- `tabayyann_backend_api_final.py` â€” FastAPI backend
- `tabayyann_query_guard_v9.py` â€” query guard
- `unified_02_retriever_v7.py` â€” hybrid retriever
- `unified_04_prepare_grounded_answer (4).py` â€” grounded evidence preparation
- `tabayyan_answer_policy_v1.json` â€” answer policy
- `index.html` â€” Arabic user interface
- `unified_rag/` â€” retrieval dataset and embeddings
- `requirements.txt` â€” Python dependencies

## Safety and Grounding

Tabayyann is designed to answer from retrieved evidence rather than relying on unsupported model memory.

The answer policy enforces rules including:

- citations are required for supported answers
- citations must correspond to retrieved evidence
- sources must not be fabricated
- Quranic verses and hadith text must not be fabricated
- the model must not fall back to memory when evidence is insufficient

If sufficient evidence is not available, the system abstains instead of inventing an answer.

## Voice Features

### Speech-to-Text

Voice questions are recorded from the browser and converted to Arabic text using ElevenLabs Scribe.

### Text-to-Speech

Generated answers can be read aloud using an Arabic ElevenLabs voice.

Citation markers such as `[1]` and `[2]` are removed before speech generation.

## Local Setup

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Required environment variables:

```text
ANTHROPIC_API_KEY
ELEVENLABS_API_KEY
```

## Health Check

`/health`

## Security

API keys are not stored in this repository. Configure them through environment variables locally or on the deployment platform.
