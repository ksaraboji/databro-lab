"""
CrewAI workflow for generating technical documentation from GitHub repositories.
"""

import os
import shutil
import threading
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallFailedEvent

from .tools import (
    analyze_dependencies,
    clone_github_repo,
    detect_tech_stack,
    extract_entry_points,
    list_repo_structure,
    read_key_files,
)

_call_counter: int = 0
_call_counter_lock = threading.Lock()
_callbacks_registered = False
_temp_repos: list[str] = []  # Track cloned repos for cleanup


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


def cleanup_temp_repos() -> None:
    """Clean up temporary cloned repositories."""
    global _temp_repos
    for repo_path in _temp_repos:
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
        except Exception:
            pass
    _temp_repos = []


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
        return max(512, int(value))
    except ValueError:
        return 2048


def resolve_llm_config(
    llm_backend: str | None = None, model_override: str | None = None
) -> tuple[str, str | None, str]:
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


def generate_technical_docs(
    github_url: str,
    llm_backend: str | None = None,
    model_override: str | None = None,
) -> str:
    """
    Generate technical documentation for a GitHub repository.

    Args:
        github_url: GitHub repository URL
        llm_backend: LLM backend to use (huggingface, openrouter, or ollama)
        model_override: Override default model name

    Returns:
        Markdown-formatted technical documentation
    """
    _ensure_counter_callbacks()

    model_name, api_key, base_url = resolve_llm_config(
        llm_backend=llm_backend,
        model_override=model_override,
    )
    max_tokens = get_max_tokens()

    backend = (llm_backend or get_default_llm_backend()).strip().lower()
    if backend == "ollama":
        llm = LLM(
            model=model_name,
            base_url=base_url,
            temperature=0.3,  # Slightly more creative for documentation
            max_tokens=max_tokens,
        )
    else:
        llm = LLM(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            provider="openai",
            temperature=0.3,  # Slightly more creative for documentation
            max_tokens=max_tokens,
        )

    doc_analyst = Agent(
        role="Technical Documentation Specialist",
        goal="Generate clear, comprehensive technical documentation from GitHub repository analysis including architecture diagrams, flow diagrams, tech stack, and dependency information.",
        backstory=(
            "You are an expert technical writer who specializes in creating "
            "low-level technical documentation. You excel at analyzing code repositories, "
            "understanding architectures, and explaining technical concepts clearly. "
            "You create Mermaid diagrams for architecture, data flows, and component relationships. "
            "You provide detailed information about libraries, frameworks, databases, and tools used."
        ),
        llm=llm,
        tools=[
            clone_github_repo,
            analyze_dependencies,
            detect_tech_stack,
            list_repo_structure,
            extract_entry_points,
            read_key_files,
        ],
        verbose=True,
    )

    # Task 1: Clone and analyze repository structure
    clone_task = Task(
        description=(
            "GitHub URL: {github_url}\n\n"
            "Your task:\n"
            "1. Clone the repository using clone_github_repo tool\n"
            "2. List the repository structure using list_repo_structure\n"
            "3. Extract entry points using extract_entry_points\n"
            "4. Read key files (README, ARCHITECTURE, etc.) using read_key_files\n\n"
            "Provide a summary of the repository structure, main components, and entry points."
        ),
        expected_output=(
            "A detailed summary including:\n"
            "- Repository structure overview\n"
            "- Main entry points by language\n"
            "- Key files and their purposes\n"
            "- High-level component organization"
        ),
        agent=doc_analyst,
    )

    # Task 2: Analyze tech stack and dependencies
    stack_task = Task(
        description=(
            "GitHub URL: {github_url}\n\n"
            "Using the cloned repository from the previous task:\n"
            "1. Detect the tech stack using detect_tech_stack\n"
            "2. Analyze dependencies using analyze_dependencies\n\n"
            "Provide a comprehensive analysis of:\n"
            "- Programming languages used\n"
            "- Frameworks and libraries\n"
            "- Databases and data stores\n"
            "- Development tools and infrastructure\n"
            "- Package managers and build systems"
        ),
        expected_output=(
            "A detailed tech stack report including:\n"
            "- Primary and secondary programming languages\n"
            "- Framework versions and key libraries\n"
            "- Database technologies\n"
            "- Cloud/infrastructure tools\n"
            "- Complete dependency list with descriptions"
        ),
        agent=doc_analyst,
        context=[clone_task],
    )

    # Task 3: Generate architecture diagrams
    architecture_task = Task(
        description=(
            "GitHub URL: {github_url}\n\n"
            "Based on the repository structure and analysis from previous tasks:\n"
            "1. Create Mermaid architecture diagrams showing:\n"
            "   - Main components and modules\n"
            "   - Service/layer relationships\n"
            "   - Data flow between components\n"
            "2. Include component descriptions\n"
            "3. Highlight external dependencies\n\n"
            "Use Mermaid syntax for flowcharts, architecture diagrams, and component diagrams.\n"
            "Make diagrams clear and easy to understand."
        ),
        expected_output=(
            "Markdown content with:\n"
            "- System architecture diagram (Mermaid)\n"
            "- Component interaction diagram (Mermaid)\n"
            "- Data flow diagram (Mermaid)\n"
            "- Descriptions of each component's purpose"
        ),
        agent=doc_analyst,
        context=[clone_task, stack_task],
    )

    # Task 4: Generate flow diagrams
    flow_task = Task(
        description=(
            "GitHub URL: {github_url}\n\n"
            "Based on the repository analysis:\n"
            "1. Create Mermaid flow diagrams for:\n"
            "   - Request/response flow\n"
            "   - Data processing pipelines\n"
            "   - Application startup sequence\n"
            "2. Include decision points and branching logic\n"
            "3. Show error handling paths\n\n"
            "Use Mermaid flowchart syntax. Make diagrams detail-oriented but readable."
        ),
        expected_output=(
            "Markdown content with:\n"
            "- Main request flow diagram (Mermaid)\n"
            "- Data processing flow (Mermaid)\n"
            "- Application initialization flow (Mermaid)\n"
            "- Flow descriptions and key decision points"
        ),
        agent=doc_analyst,
        context=[clone_task, stack_task],
    )

    # Task 5: Compile comprehensive technical documentation
    compile_task = Task(
        description=(
            "GitHub URL: {github_url}\n\n"
            "Compile all the analysis from previous tasks into a comprehensive technical documentation.\n"
            "Structure:\n"
            "1. **Overview** - Brief description of the project\n"
            "2. **Tech Stack** - Languages, frameworks, databases, tools\n"
            "3. **Architecture** - High-level system design with diagrams\n"
            "4. **Components** - Detailed description of main components\n"
            "5. **Data Flow** - How data moves through the system (with diagrams)\n"
            "6. **Dependencies** - Key libraries and their purposes\n"
            "7. **Entry Points** - Main files and how the application starts\n"
            "8. **Configuration** - Environment setup and configuration\n\n"
            "Use clear Markdown formatting with proper headers, code blocks, and Mermaid diagrams.\n"
            "Make it suitable for test engineers to understand the system."
        ),
        expected_output=(
            "A complete Markdown document containing:\n"
            "- Professional overview section\n"
            "- Comprehensive tech stack table\n"
            "- Architecture diagrams with explanations\n"
            "- Data flow diagrams\n"
            "- Component descriptions\n"
            "- Dependencies documentation\n"
            "- Configuration and setup information\n"
            "- All information suitable for test case generation"
        ),
        agent=doc_analyst,
        context=[clone_task, stack_task, architecture_task, flow_task],
    )

    crew = Crew(
        agents=[doc_analyst],
        tasks=[clone_task, stack_task, architecture_task, flow_task, compile_task],
        process=Process.sequential,
        verbose=True,
    )

    response = crew.kickoff(inputs={"github_url": github_url})
    return str(response)
