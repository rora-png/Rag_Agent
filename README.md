# RAG Agent

A small retrieval-augmented generation (RAG) pipeline built from scratch to understand how retrieval, chunking, and grounded generation fit together — no framework, just OpenAI's API and FAISS.

## How it works

1. **Load** — plain text documents are read in from a local `docs/` folder.
2. **Chunk** — each document is split into overlapping chunks (500 characters, 100 character overlap) using a simple sliding window.
3. **Embed** — each chunk is embedded with OpenAI's `text-embedding-3-small` model.
4. **Index** — embeddings are stored in a FAISS `IndexFlatL2` index for similarity search.
5. **Query** — at question time, the query is embedded, FAISS returns the top-k most similar chunks (default k=3), and those chunks are passed as context into `gpt-4o-mini` to generate a grounded answer.

The system prompt explicitly restricts the model to the provided context — if the answer isn't in the retrieved chunks, it says it doesn't know rather than guessing.

## What I tested

- **Groundedness** — asked a question with no answer in the source docs ("Who won the 2022 World Cup?") to confirm the model says "I don't know" instead of hallucinating.
- **Retrieval depth** — compared answer quality at k=1 vs k=8 for the same question, to see how much retrieved context changes the response.

## Stack

- Python
- OpenAI API (embeddings + chat completions)
- FAISS (local vector similarity search)
- python-dotenv for API key management

## Scale

Small demo corpus: 3 source documents chunked into 134 pieces, covering topics like Docker containers/images and BERT.

## Notes

This was built to understand the mechanics of RAG end to end — chunking strategy, embedding choice, and how retrieval quality affects generation — rather than to be a production system. No orchestration framework (LangChain, LlamaIndex) was used; everything is implemented directly against the OpenAI and FAISS APIs.
