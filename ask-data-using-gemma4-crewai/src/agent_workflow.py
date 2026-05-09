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


def get_default_llm_backend() -> str:
    backend = os.getenv("LLM_BACKEND", "huggingface").strip().lower()
    return backend if backend in {"huggingface", "openrouter"} else "huggingface"


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

    model_name = (model_override or get_hf_model_name()).strip()
    if model_name.startswith("huggingface/"):
        model_name = model_name.removeprefix("huggingface/")
    return model_name, get_hf_api_key(), get_hf_base_url()


def run_agentic_query(
    file_path: str,
    user_query: str,
    llm_backend: str | None = None,
    model_override: str | None = None,
) -> str:
    _ensure_counter_callbacks()

    model_name, api_key, base_url = resolve_llm_config(
        llm_backend=llm_backend,
        model_override=model_override,
    )
    max_tokens = get_max_tokens()

    llm = LLM(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        provider="openai",
        temperature=0,
        max_tokens=max_tokens,
    )

    analyst = Agent(
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

    # Task 1: Schema Inspection
    # Deterministic — always calls inspect_data_file and returns file type + schema.
    # No SQL, no decisions. Just discovery.
    schema_task = Task(
        description=(
            "File path: {file_path}\n\n"
            "You MUST call inspect_data_file with this exact file_path and nothing else.\n"
            "Do not write SQL. Do not answer the user question. Just inspect the file."
        ),
        expected_output=(
            "A JSON object containing:\n"
            "- file_type: the detected format (csv, json, ndjson, parquet, or arrow)\n"
            "- schema: a list of columns with name and type\n"
            "Do not include anything else."
        ),
        agent=analyst,
    )

    # Task 2: Execution
    # Uses schema from Task 1 to decide the right tool and execute it.
    # Three exclusive paths: SQL analysis, SQL export, or format conversion.
    execution_task = Task(
        description=(
            "File path: {file_path}\n"
            "User question: {user_query}\n\n"
            "The file schema is provided in the context from the previous task.\n"
            "Based on the user's intent, choose exactly ONE of the following paths:\n\n"
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
            "- If requested, include a first-10-row preview as a markdown table"
        ),
        agent=analyst,
        context=[schema_task],
    )

    crew = Crew(
        agents=[analyst],
        tasks=[schema_task, execution_task],
        process=Process.sequential,  # schema_task always runs before execution_task
        verbose=True,
    )

    response = crew.kickoff(inputs={"file_path": file_path, "user_query": user_query})
    return str(response)
