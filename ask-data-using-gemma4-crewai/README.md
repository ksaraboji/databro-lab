# CSV/JSON/Parquet/Arrow Agentic SQL Prototype

This prototype is a simple, fully agentic workflow using:

- Gradio UI
- CrewAI agent framework
- Hugging Face Gemma endpoint (no fallback model)
- DuckDB execution tool
- Content-based file detection and format conversion

## What it does

1. User uploads a supported file type.
2. Agent calls a tool to detect file type and extract schema.
3. The workflow runs in two sequential CrewAI tasks:
	- Task 1 inspects the file and returns the detected format plus schema.
	- Task 2 uses that schema to decide whether to analyze, export, or convert.
4. Agent sends the user query + schema context to the Hugging Face Gemma endpoint.
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

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set a Hugging Face token and model name:

```bash
export HF_API_TOKEN=your_hugging_face_token
export HF_MODEL_NAME=google/gemma-4-31B-it
export HF_BASE_URL=https://router.huggingface.co/v1
```

4. Start the app:

```bash
python app.py
```

Open http://localhost:7860

Use the chat panel for follow-up questions on the same uploaded file. If you ask for an updated output file, it appears in the "Generated file (when requested)" section.

If you explicitly ask for a preview, head rows, or a sample, the agent includes the first 10 rows in the chat response as a markdown table rendered with `tabulate`.

The app also includes a colored UI header, a footer link to <https://databro.dev>, and Gradio share mode enabled so it can generate a public link when launched.

If you ask to convert the file, for example "convert this parquet file to csv" or "save as ndjson", the agent uses the conversion tool and returns a downloadable file path.

If you ask to flatten nested JSON attributes, for example "flatten the JSON", "expand nested fields", or "show nested attributes as columns", the agent calls the `flatten_nested_json` tool and returns a dot-path column preview. For all other queries, nested attributes are accessed directly using DuckDB dot-path SQL syntax (e.g. `user.address.city`).

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

```bash
export HF_API_TOKEN=your_hugging_face_token
export HF_MODEL_NAME=google/gemma-4-31B-it
export HF_BASE_URL=https://router.huggingface.co/v1
```

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
