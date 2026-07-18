# Conversational RAG Agent

A retrieval-augmented conversational agent built with **Python, LangChain, FAISS, Hugging Face, and the OpenAI API** — with URL ingestion, two-stage retrieval, conversation memory, and an evaluation and monitoring layer.

## Pipeline

```
URLs → ingestion (trafilatura) → clean .txt → chunk → embed → FAISS
                                                                 │
question → condense (memory) → FAISS top-10 → HF reranker → top-3 → grounded LLM answer
                                                                        │
                                              monitoring log (latency, retrieval scores, quality)
```

## How it works

**1. Ingestion** — `Ingest_articles.py` pulls articles from URLs using trafilatura, which extracts the main body text and strips navigation, ads, and boilerplate. Clean text matters: junk content would get chunked and embedded too, polluting retrieval.

**2. Chunking** — LangChain's `RecursiveCharacterTextSplitter`, 500 characters with 100 overlap. Splits on natural boundaries (paragraphs → sentences → words) before hard cuts, so chunk edges land on meaningful breaks.

**3. Embedding + indexing** — each chunk embedded with OpenAI's `text-embedding-3-small` and stored in a FAISS index.

**4. Two-stage retrieval** — FAISS pulls a wide net of candidate chunks fast via embedding similarity, then a Hugging Face cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks them. The cross-encoder reads the question and each chunk *together*, scoring actual relevance that embedding distance alone misses. Top chunks after reranking go to generation.

**5. Grounded generation** — the prompt restricts `gpt-4o-mini` to the retrieved context only. If the answer isn't there, it says "I don't know" rather than hallucinating.

**6. Conversation memory** — LangChain memory keeps chat history, and each new question is rewritten into a standalone question using that history — so follow-ups like "how is it different?" resolve correctly before retrieval.

## Evaluation

A labeled test set of document-specific questions, run through both a **baseline** (LLM alone, no retrieval) and the full **RAG pipeline**, graded automatically by a temperature-0 LLM grader.

Results on a corpus of recent articles (including Twilio's Signal 2026 announcements — content past the model's training data):

- Baseline accuracy: ~37%
- RAG accuracy: ~50-60% depending on retrieval settings

**Lessons from building the eval:**
- General-knowledge eval questions show zero RAG improvement — the baseline already knows the answers. Questions must target facts only present in the documents.
- More retrieved chunks isn't always better — extra context can dilute answers. Retrieval precision beat volume.
- Metadata headers written into the text files by ingestion were matching queries without containing content, hurting retrieval — stripping them before embedding improved results.

## Monitoring

Every query logs to an inspectable dataframe:
- **Latency** — wall-clock time for retrieve + rerank + generate
- **Retrieval signal** — top FAISS distance and top rerank score for the chunks used
- **Response quality** — graded against an expected answer when available

## Setup

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu sentence-transformers trafilatura python-dotenv pandas
```

1. Create a `.env` file with `OPENAI_API_KEY=your-key`
2. Add article URLs to `Ingest_articles.py` and run it to populate `docs/`
3. Run the notebook top to bottom

## Stack

| Tool | Role |
|---|---|
| LangChain | Document loading, chunking, memory, prompts |
| OpenAI API | Embeddings + generation |
| FAISS | First-stage vector similarity search |
| Hugging Face | Cross-encoder reranking (second-stage retrieval) |
| trafilatura | Clean article extraction from URLs |
| Python | Evaluation harness, monitoring, glue |