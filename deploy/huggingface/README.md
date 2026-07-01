---
title: RAG Assistant
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RAG Assistant

Upload PDF documents and ask questions answered from their content, with source
citations. Built with Flask, LangChain, FAISS, FastEmbed, and Groq.

## Setup

This Space needs one secret. Go to **Settings → Variables and secrets → New secret**
and add:

- `GROQ_API_KEY` — your free key from https://console.groq.com

That's it. Uploads and the vector index are stored in `/tmp` (they reset when the
Space restarts, which is expected on the free tier).
