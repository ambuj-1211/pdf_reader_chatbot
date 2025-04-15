FROM python:3.12.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LLM=gemma2:2b \
    EMBEDDING_MODEL=sentence_transformer \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "pdf_bot.py", "--server.port=8080", "--server.address=0.0.0.0"]