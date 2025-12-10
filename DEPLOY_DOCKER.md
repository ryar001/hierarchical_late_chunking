# Deployment Guide

This project is containerized using Docker for easy deployment.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your machine.
[diff_block_start]
- API Keys for Google Gemini (`GOOGLE_API_KEY`).

## Docker Setup

### 1. Build the Image

Run the following command in the root directory of the project:

```bash
docker build -t hierarchical-rag .
```

### 2. Run the Container

You need to pass your API keys as environment variables.

**Option A: Using a `.env` file (Recommended)**

Ensure you have a `.env` file in your current directory with your keys (do not commit this file):

```env
GOOGLE_API_KEY=your_google_key
# CHROMA_HOST=... (only if using remote ChromaDB)
```

Run with `--env-file`:

```bash
docker run -p 8000:8000 --env-file .env hierarchical-rag
```

**Option B: Passing keys directly**

If the `.env` file method fails or you prefer explicit variables:

```bash
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY="your_google_key" \
  hierarchical-rag
```

### 3. Access the Application
...
      --set-env-vars GOOGLE_API_KEY=...
    ```
[diff_block_end]

    *Note: For Cloud Run, ensure you use a cloud-hosted ChromaDB or persist data to a bucket/volume, as local container storage is ephemeral.*
