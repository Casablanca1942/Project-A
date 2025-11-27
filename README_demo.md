# Demo: generate_demo.py

This demo exercises the full flow described in the instructions:
- Vector DB lookup (textExtract/chroma_db)
- yfinance wrapper functions in `financeAnalysis` (snapshot, news)
- Optional HuggingFace model generation or a deterministic fallback for environments without models

Quick start
-----------
1. Ensure you created the vector DB: `python textExtract/build_vector_db.py` (this will create `textExtract/chroma_db`).
2. Install required packages (from root):

```powershell
conda activate finance   # or activate your env
pip install -r requirements.txt
# Optional: install transformers if you want model generation
pip install transformers
```

3. Run the demo (this version supports a model-driven function-calling flow; fallback generators run when dependencies are missing):

```powershell
python generate_demo.py --query "What is the margin of safety for AAPL?" --symbol AAPL

# If you run Ollama (llama3) locally you can prefer it by setting environment variables or passing a model name:
Setting OLLAMA_URL to the local URL (default http://localhost:11434) is supported.

```powershell

# example: prefer Ollama local model llama3
setx OLLAMA_URL "http://localhost:11434"
setx OLLAMA_MODEL "llama3"
python generate_demo.py --query "What is the margin of safety for AAPL?" --symbol AAPL

# Or use a small HF model instead (downloads model weights if necessary):
python generate_demo.py --query "What is the margin of safety for AAPL?" --symbol AAPL --model-name distilgpt2

Function-calling flow
---------------------
This demo will ask the model which tools (financeAnalysis functions) should be called to answer the user question. The model is asked to return a JSON object listing the tools and parameters. The program executes those tools, bundles their outputs, and then asks the model to produce a final, concise plain-text answer.

If the model or Ollama is not available, the demo will run a deterministic fallback path using minimal simulated tool outputs so it remains runnable for tests.

Notes about language and outputs
------------------------------
- Program strings and prompts are in English (suitable for English queries).
- Final output is plain text (no JSON) to make it easy for users and downstream clients.

Advanced
--------
If you prefer streaming token-by-token outputs from Ollama, you can modify `generate_demo.py` to request streaming and then assemble the NDJSON fragments into a final string. The script is designed to request non-stream responses by default to avoid partial JSON fragments in the output.
```

Notes
----
- If `textExtract/chroma_db` does not exist, run `python textExtract/build_vector_db.py` first.
- The demo is purposely lightweight and meant for local development/testing; it uses a templated fallback when no model is available.
