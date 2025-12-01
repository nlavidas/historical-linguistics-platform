#!/bin/bash
# Install Ollama (open-source vLLM alternative)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Qwen2.5-Coder model (open-source coding model)
ollama pull qwen2.5-coder:7b

# Pull embedding model for RAG
ollama pull nomic-embed-text

# Start Ollama service
nohup ollama serve > /tmp/ollama.log 2>&1 &

echo "Ollama setup complete"
