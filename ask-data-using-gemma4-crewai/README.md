# Ask Data Using Gemma4 + CrewAI

This spike is a fully agentic data chat prototype using:

- Gradio UI
- CrewAI agent framework
- Hugging Face Router, OpenRouter, or local Ollama (selectable in UI)
- DuckDB execution tool
- Content-based file detection and format conversion

## What it does

1. User uploads a supported file type.
2. Agent calls a tool to detect file type and extract schema.
3. The workflow runs in two sequential CrewAI tasks:
	- Task 1 inspects the file and returns the detected format plus schema.
	- Task 2 uses that schema to decide whether to analyze, export, or convert.
4. Agent sends the user query + schema context to the selected LLM backend (Hugging Face Router, OpenRouter, or local Ollama).
5. Agent either generates DuckDB SQL or chooses a direct file conversion path.
6. Agent decides how to execute based on intent:
	- Returns table results for analytical questions.
	- Exports and returns a downloadable file for update/filter/sort/export requests.
	- Converts the uploaded file to another format when explicitly requested.
	- Shows the first 10 rows of the uploaded file using a tabulated preview when the user explicitly asks for a preview or head rows.
	- Flattens deeply nested JSON/NDJSON attributes into dot-path columns **only when the user explicitly asks** (e.g. "flatten the JSON", "expand nested fields", "show nested attributes as columns").

No rule-based query routing or custom fallback logic is implemented.

## Supported file formats

- Input detection is content-based, not extension-based.
- Supported uploads: `.csv`, `.json`, `.ndjson`, `.jsonl`, `.parquet`, `.arrow`, `.ipc`
- CSV uses DuckDB auto-detection.
- JSON and NDJSON/JSONL are handled separately.
- Parquet is read directly by DuckDB.
- Arrow IPC files are loaded through PyArrow and registered into DuckDB.
- The inspection step includes a head preview of the first 10 rows rendered with `tabulate`.

## Supported outputs

- Query results can be exported as `.csv` or `.json`.
- Full file conversion supports `.csv`, `.json`, `.ndjson`, `.parquet`, and `.arrow`.

## LLM Backend Selection

The app supports three LLM backends, selectable directly in the Gradio UI:

### Hugging Face Router
- **Default models**: `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`
- **Why use it**: Free tier available, good for prototyping
- **Rate limits**: Shared queue; may experience delays during high traffic
- **UI behavior**: Select `huggingface` in the **LLM Backend** dropdown, then pick a model from the **Model Name** dropdown

### OpenRouter
- **Default models**: `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`
- **Why use it**: Unified API for many model providers, good for production
- **Rate limits**: Per-account limits; typically higher throughput than HF Router
- **UI behavior**: Select `openrouter` in the **LLM Backend** dropdown, then pick a model from the **Model Name** dropdown

### Ollama (Local)
- **Default models**: `gemma4:latest`, `gemma4:e2b-it-q4_K_M`
- **Why use it**: Fully local inference, no external API calls
- **Rate limits**: None from a hosted provider (limited by your machine resources)
- **UI behavior**: Select `ollama` in the **LLM Backend** dropdown, then pick a local model from the **Model Name** dropdown
- **Requirements**: Run Ollama locally and pull the model first

**Switching backends in the UI**:
1. Upload a file.
2. Open the **LLM Backend** dropdown and select `huggingface`, `openrouter`, or `ollama`.
3. The **Model Name** dropdown automatically updates to show models available for that backend.
4. (Optional) Override the model if you want a different variant.
5. Send your query. The agent will use the selected backend and model for that request.

Each chat request uses the backend/model selected at send time, so you can switch mid-conversation.

## Understanding the API Call Counter

The chat panel displays a **Call Counter** showing how many LLM API attempts were made. This is important because the counter tracks **API calls**, not **chat requests**.

### Why does one query result in multiple API calls?

The agent runs two sequential CrewAI tasks:
1. **Task 1 (Schema Inspection)**: Call `inspect_data_file` to detect format and extract schema.
2. **Task 2 (Execution)**: Use schema context to decide the right path (analysis, export, conversion, or flatten) and execute it.

Each task may trigger multiple LLM API attempts due to internal reasoning, retry logic, or JSON parsing. For example:
- Schema task: 1–2 API calls (inspect + validation)
- Execution task: 2–3 API calls (planning + execution + fallback)

**Result**: A typical single user query can trigger **4–6 API calls** total. This aligns with what you see on provider dashboards (Hugging Face Router/OpenRouter) or local call attempts when using Ollama.

### Example

You ask: *"What is the average value column in this CSV?"*

- Call 1: Schema task attempts to inspect the file.
- Call 2: Schema task completes and returns schema.
- Call 3: Execution task analyzes intent and plans SQL.
- Call 4: Execution task executes the SQL and returns the result.
- (Calls 5–6 may occur if retries or validation steps are triggered.)

**Counter display**: Shows `4–6` for this single query.

Do not be alarmed if the counter shows 4+ for a single question—it reflects actual LLM API usage, which helps you monitor token usage and costs across backends.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables. Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` and configure **at least one backend**:

**Shared settings**:
```bash
# Backend to use by default when the app starts
LLM_BACKEND=huggingface

# Token limit for all LLM calls
LLM_MAX_TOKENS=2048
```

**Hugging Face Router** (required if using HF):
```bash
HF_API_TOKEN=your_hugging_face_api_token
HF_MODEL_NAME=google/gemma-4-31B-it
HF_BASE_URL=https://router.huggingface.co/v1
```

**OpenRouter** (required if using OpenRouter):
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL_NAME=google/gemma-4-31b-it:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

**Ollama (local)** (required if using Ollama):
```bash
OLLAMA_API_KEY=ollama
OLLAMA_MODEL_NAME=gemma4:latest
OLLAMA_BASE_URL=http://localhost:11434
```

Pull local models before running:

```bash
ollama pull gemma4:latest
ollama pull gemma4:e2b-it-q4_K_M
```

You can configure both backends and switch between them in the UI, or just configure the one(s) you plan to use.

4. Start the app:

```bash
python app.py
```

Open http://localhost:7860 in your browser.

### Using the app

- **Upload a file**: Select CSV, JSON, NDJSON, Parquet, or Arrow format.
- **Choose backend & model**: Use the **LLM Backend** and **Model Name** dropdowns to select Hugging Face, OpenRouter, or local Ollama.
- **Ask questions**: Type your query in the chat box and press Enter. The agent will inspect the file schema and answer.
- **Track API usage**: Watch the **Call Counter** to see how many LLM API calls were made (typically 4–6 per query).
- **Follow-up questions**: Ask multiple questions on the same file without re-uploading.
- **Export results**: Ask the agent to export filtered/transformed data as CSV or JSON. The download link appears in the chat response.
- **Convert formats**: Ask to convert the file (e.g., "convert to parquet", "save as CSV"). The agent returns a downloadable file path.
- **Preview data**: Ask to see the first 10 rows (e.g., "show me the head", "preview this data"). The agent returns a markdown table.
- **Flatten nested JSON**: For nested JSON/NDJSON files, ask to flatten (e.g., "flatten the JSON", "expand nested fields", "show nested attributes as columns"). The agent expands nested structures into dot-path columns.

The app includes a colored UI header and a footer link to <https://databro.dev>. Gradio share mode is enabled, so launching the app generates a temporary public link for sharing.

## Run From GitHub

1. Clone the repository:

```bash
git clone https://github.com/ksaraboji/databro-lab.git
cd databro-lab/ask-data-using-gemma4-crewai
```

2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Copy `.env.example` to `.env` and fill in your credentials for at least one backend:

```bash
cp .env.example .env

# Edit and configure:
# - LLM_BACKEND (huggingface, openrouter, or ollama)
# - LLM_MAX_TOKENS (shared token limit)
# - HF_API_TOKEN, HF_MODEL_NAME, HF_BASE_URL (Hugging Face)
# - OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME, OPENROUTER_BASE_URL (OpenRouter)
# - OLLAMA_API_KEY, OLLAMA_MODEL_NAME, OLLAMA_BASE_URL (Ollama local)
```

See [Setup](#setup) above for complete configuration details.

5. Launch the app:

```bash
python app.py
```

## Sharing And Deployment

- For temporary sharing, the app already launches with Gradio `share=True`, which generates a public `gradio.live` URL while the process is running.
- For a more stable hosted setup, deploy the repo to Hugging Face Spaces or another Python hosting target and provide the same environment variables there.
- Keep `.env` local only. It is ignored by git and should not be committed.

## Workflow details

- The CrewAI crew runs with `Process.sequential` so schema inspection always happens before execution.
- The execution task chooses one of four paths:
	- **Analysis path (C)**: run DuckDB SQL and return a markdown table.
	- **Export path (B)**: run DuckDB SQL and write a downloadable CSV or JSON file.
	- **Conversion path (A)**: convert the original file to another supported format.
	- **Flatten path (D)**: flatten nested JSON/NDJSON attributes into dot-path columns — triggered only by explicit user intent (e.g. "flatten", "expand nested fields", "show dot-path columns"). Not applied automatically.

## Notes

- Supported upload types: `.csv`, `.json`, `.ndjson`, `.jsonl`, `.parquet`, `.arrow`, `.ipc`
- SQL is executed against a virtual table named `data`.
