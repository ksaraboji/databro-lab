import os
import threading

from crewai import Agent, Crew, LLM, Process, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallFailedEvent

from .tools import export_duckdb_sql_result, inspect_data_file, run_duckdb_sql, convert_file_format, flatten_nested_json

_call_counter: int = 0
_call_counter_lock = threading.Lock()
_callbacks_registered = False


def _increment_call_counter() -> None:
    global _call_counter
    with _call_counter_lock:
        _call_counter += 1


def _ensure_counter_callbacks() -> None:
    global _callbacks_registered
    with _call_counter_lock:
        if _callbacks_registered:
            return

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_llm_call_completed(source, event):
            _increment_call_counter()

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_llm_call_failed(source, event):
            _increment_call_counter()

        _callbacks_registered = True


def get_call_count() -> int:
    with _call_counter_lock:
        return _call_counter


def get_hf_api_key() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")


def get_hf_model_name() -> str:
    model_name = os.getenv("HF_MODEL_NAME", "google/gemma-4-31B-it")
    if model_name.startswith("huggingface/"):
        return model_name.removeprefix("huggingface/")
    return model_name


def get_hf_base_url() -> str:
    return os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")


def get_openrouter_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


def get_openrouter_model_name() -> str:
    return os.getenv("OPENROUTER_MODEL_NAME", "google/gemma-4-31b-it:free")


def get_openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def get_ollama_api_key() -> str | None:
    # Kept for backward compatibility with existing env files; Ollama local calls do not require it.
    return os.getenv("OLLAMA_API_KEY")


def get_ollama_model_name() -> str:
    return os.getenv("OLLAMA_MODEL_NAME", "gemma4:latest")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_default_llm_backend() -> str:
    backend = os.getenv("LLM_BACKEND", "huggingface").strip().lower()
    return backend if backend in {"huggingface", "openrouter", "ollama"} else "huggingface"


def get_max_tokens() -> int:
    value = os.getenv("LLM_MAX_TOKENS", os.getenv("HF_MAX_TOKENS", "2048"))
    try:
        return max(256, int(value))
    except ValueError:
        return 2048


def resolve_llm_config(llm_backend: str | None = None, model_override: str | None = None) -> tuple[str, str | None, str]:
    backend = (llm_backend or get_default_llm_backend()).strip().lower()
    if backend == "openrouter":
        model_name = (model_override or get_openrouter_model_name()).strip()
        return model_name, get_openrouter_api_key(), get_openrouter_base_url()

    if backend == "ollama":
        model_name = (model_override or get_ollama_model_name()).strip()
        if not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"
        return model_name, None, get_ollama_base_url()

    model_name = (model_override or get_hf_model_name()).strip()
    if model_name.startswith("huggingface/"):
        model_name = model_name.removeprefix("huggingface/")
    return model_name, get_hf_api_key(), get_hf_base_url()


def _build_llm(backend: str, model_override: str | None) -> LLM:
    model_name, api_key, base_url = resolve_llm_config(
        llm_backend=backend,
        model_override=model_override,
    )
    max_tokens = get_max_tokens()

    if backend in {"huggingface", "openrouter"} and not api_key:
        raise ValueError(f"{backend} backend requires its API key in .env before starting chat.")

    if backend == "ollama":
        return LLM(
            model=model_name,
            base_url=base_url,
            temperature=0,
            max_tokens=max_tokens,
        )

    return LLM(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        provider="openai",
        temperature=0,
        max_tokens=max_tokens,
    )


def _build_agent(llm: LLM) -> Agent:
    return Agent(
        role="DuckDB Data Analyst",
        goal="Answer user questions by creating and running DuckDB SQL on uploaded CSV, JSON, NDJSON, Parquet, or Arrow data. Also convert files between formats when requested.",
        backstory=(
            "You are a data analyst who always inspects schema first and then writes DuckDB SQL. "
            "You execute SQL with tools and report only tool-backed results. "
            "You work with multiple data formats: CSV, JSON, NDJSON, Parquet, and Arrow. "
            "You can convert files from one format to another when users request it."
        ),
        llm=llm,
        tools=[inspect_data_file, run_duckdb_sql, export_duckdb_sql_result, convert_file_format, flatten_nested_json],
        verbose=True,
    )


def _build_crew(agent: Agent) -> Crew:
    schema_task = Task(
        description=(
            "File path: {file_path}\n\n"
            "You MUST call inspect_data_file with this exact file_path and nothing else.\n"
            "Do not write SQL. Do not answer the user question. Just inspect the file."
        ),
        expected_output=(
            "A JSON object containing:\n"
            "- file_type: the detected format (csv, json, ndjson, parquet, or arrow)\n"
            "- row_count: exact total number of rows in the uploaded file\n"
            "- schema: a list of columns with name and type\n"
            "Do not include anything else."
        ),
        agent=agent,
    )

    execution_task = Task(
        description=(
            "File path: {file_path}\n"
            "User question: {user_query}\n\n"
            "The file schema is provided in the context from the previous task.\n"
            "Based on the user's intent, choose exactly ONE of the following paths:\n\n"
            "For any question about number of rows/records/entries/count, you MUST compute the exact value using SQL "
            "`SELECT COUNT(*) AS row_count FROM data` and report that result. Never estimate counts from previews.\n\n"
            "If the user explicitly asks for head rows, a preview, a sample, or the first 10 rows,\n"
            "include the first 10 rows in the final response using a markdown table.\n\n"
            "PATH A — Format conversion:\n"
            "  If user asks to convert the file to a different format (e.g. 'convert to csv', "
            "'save as parquet', 'to json'), call convert_file_format with file_path and target format.\n\n"
            "PATH B — Downloadable filtered/transformed file:\n"
            "  If user asks for an updated, filtered, sorted, or exported file to download, "
            "write the DuckDB SQL using the schema, then call export_duckdb_sql_result with "
            "file_path, SQL, and output_format ('csv' or 'json').\n\n"
            "PATH C — Analytical answer:\n"
            "  For all other questions (counts, averages, rankings, comparisons, summaries), "
            "write the DuckDB SQL using the schema, then call run_duckdb_sql with file_path and SQL.\n\n"
            "PATH D — Flatten nested JSON attributes:\n"
            "  ONLY if the user explicitly asks to flatten, expand nested fields, show dot-path columns,\n"
            "  or see nested attributes as separate columns (e.g. 'flatten the JSON', 'expand nested',\n"
            "  'show nested fields as columns', 'dot-path columns'), call flatten_nested_json with file_path.\n"
            "  Do NOT call this for any other intent — standard queries use PATH A/B/C instead."
        ),
        expected_output=(
            "A concise markdown response containing:\n"
            "- Chosen path (A, B, or C) and why\n"
            "- SQL query used (if applicable)\n"
            "- Query result as a markdown table (Path C) or confirmation with download file path (Path A/B)\n"
            "- Any count in the answer must come from SQL COUNT(*) output, not estimates\n"
            "- If requested, include a first-10-row preview as a markdown table"
        ),
        agent=agent,
        context=[schema_task],
    )

    return Crew(
        agents=[agent],
        tasks=[schema_task, execution_task],
        process=Process.sequential,
        verbose=True,
    )


def run_agentic_query(
    file_path: str,
    user_query: str,
    llm_backend: str | None = None,
    model_override: str | None = None,
) -> str:
    _ensure_counter_callbacks()

    backend = (llm_backend or get_default_llm_backend()).strip().lower()
    llm = _build_llm(backend=backend, model_override=model_override)
    agent = _build_agent(llm)
    crew = _build_crew(agent)

    response = crew.kickoff(inputs={"file_path": file_path, "user_query": user_query})
    return str(response)
