export OLLAMA_API_KEY="c3aff5d330844f0db4b3680f600ae12d.Z_YOLmtUIAbyiYoPjRi9QeQB"
curl https://ollama.com/api/chat \
  -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "model": "gemma4:31b", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "stream": false }'
