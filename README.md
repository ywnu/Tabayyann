# تبيّنّ | Tabayyann

**Tabayyann** is an Arabic grounded Islamic Q&A system built for reliable, source-based answers.

The system uses Retrieval-Augmented Generation (RAG) to retrieve evidence from an approved local knowledge base before generating an answer. It is designed to avoid unsupported answers when sufficient evidence is not available.

## Live Demo

https://tabayyann-production.up.railway.app

## Team

- [Deem Almanea](https://github.com/ywnu) — `@ywnu`
- [Alanoud Alhamad](https://github.com/Alanoudb1) — `@Alanoudb1`
- [Wahaj Almarwi](https://github.com/wahaj2005x-blip) — `@wahaj2005x-blip`
- [Jumanah Alotaibi](https://github.com/Jumanah-1) — `@Jumanah-1`
- [Hanan Almutairi](https://github.com/lixr-7) — `@lixr-7`

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

```text
User Question
    |
    v
Query Guard
    |
    v
Hybrid Retriever
    |
    v
Grounded Evidence Preparation
    |
    v
Answer Policy
    |
    v
Claude
    |
    v
Validated Answer + Citations
```

## Main Project Files

- `main.py` - Railway application entry point
- `tabayyann_backend_api_final.py` - FastAPI backend
- `tabayyann_query_guard_v9.py` - query guard
- `unified_02_retriever_v7.py` - hybrid retriever
- `unified_04_prepare_grounded_answer (4).py` - grounded evidence preparation
- `tabayyan_answer_policy_v1.json` - answer policy
- `index.html` - Arabic user interface
- `unified_rag/` - retrieval dataset and embeddings
- `requirements.txt` - Python dependencies

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

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the required environment variables:

```text
ANTHROPIC_API_KEY
ELEVENLABS_API_KEY
```

Run the application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## Health Check

```text
/health
```

Example:

```text
https://tabayyann-production.up.railway.app/health
```

## Security

API keys are not stored in this repository. Configure them through environment variables locally or on the deployment platform.

## Deployment

The production version is deployed using Railway.

## Project

Built as a hackathon project focused on trustworthy Arabic Islamic question answering with retrieval, grounding, citations, and voice interaction.
