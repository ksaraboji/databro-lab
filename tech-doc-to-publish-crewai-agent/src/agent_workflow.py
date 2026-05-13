"""Agentic workflow: technical documentation to curiosity-driven publish-ready articles."""

import base64
import json
import os
import re
import threading
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallFailedEvent

from .tools import (
    publish_to_devto,
    read_technical_document,
    save_markdown_artifact,
    save_publication_manifest,
)

_call_counter: int = 0
_call_counter_lock = threading.Lock()
_callbacks_registered = False


def _extract_mermaid_blocks(markdown_text: str) -> list[str]:
    pattern = re.compile(r"```mermaid\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    blocks = [match.strip() for match in pattern.findall(markdown_text)]
    return [block for block in blocks if block]


def _build_mermaid_image_url(mermaid_code: str) -> str:
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
    encoded = encoded.rstrip("=")
    return f"https://mermaid.ink/img/{encoded}?type=png"


def _prepare_mermaid_assets(technical_doc_path: str) -> tuple[list[dict[str, str]], str]:
    markdown_text = Path(technical_doc_path).read_text(encoding="utf-8", errors="replace")
    blocks = _extract_mermaid_blocks(markdown_text)
    assets: list[dict[str, str]] = []

    for index, block in enumerate(blocks, start=1):
        assets.append(
            {
                "name": f"diagram-{index}",
                "alt_text": f"Technical architecture diagram {index}",
                "image_url": _build_mermaid_image_url(block),
                "mermaid_code": block,
            }
        )

    if not assets:
        return assets, "[]"

    return assets, json.dumps(assets, indent=2)


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


def get_ollama_model_name() -> str:
    return os.getenv("OLLAMA_MODEL_NAME", "gemma4:latest")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_default_llm_backend() -> str:
    backend = os.getenv("LLM_BACKEND", "huggingface").strip().lower()
    return backend if backend in {"huggingface", "openrouter", "ollama"} else "huggingface"


def get_max_tokens() -> int:
    value = os.getenv("LLM_MAX_TOKENS", os.getenv("HF_MAX_TOKENS", "8192"))
    try:
        return max(512, int(value))
    except ValueError:
        return 8192


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


def generate_publishable_articles(
    technical_doc_path: str,
    llm_backend: str | None = None,
    model_override: str | None = None,
    publish_now: bool = False,
    tone_reference: str | None = None,
) -> str:
    """Generate and optionally publish a Dev.to article from an uploaded technical doc."""
    _ensure_counter_callbacks()
    mermaid_assets, mermaid_assets_json = _prepare_mermaid_assets(technical_doc_path)

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
            temperature=0.7,
            max_tokens=max_tokens,
        )
    else:
        llm = LLM(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            provider="openai",
            temperature=0.7,
            max_tokens=max_tokens,
        )

    intake_agent = Agent(
        role="Technical Document Intake Analyst",
        goal="Extract precise themes, novelty points, and evidence from uploaded technical docs.",
        backstory=(
            "You are an analytical editor who turns dense technical material into concise, "
            "citation-ready insights while preserving accuracy."
        ),
        llm=llm,
        tools=[read_technical_document],
        verbose=True,
    )

    curiosity_editor = Agent(
        role="Curiosity-Driven Story Architect",
        goal="Design irresistible headlines and narrative arcs that maximize curiosity without clickbait fluff.",
        backstory=(
            "You are a senior technical storyteller known for high-retention articles that blend "
            "technical depth with suspenseful structure and tasteful emojis."
        ),
        llm=llm,
        verbose=True,
    )

    channel_optimizer = Agent(
        role="Dev.to Markdown Optimizer",
        goal="Create a Dev.to-ready markdown package with tags, cover metadata, and YAML front matter.",
        backstory=(
            "You are a publication strategist who crafts high-performing Dev.to articles from technical content."
        ),
        llm=llm,
        verbose=True,
    )

    publisher_agent = Agent(
        role="Publishing Operations Agent",
        goal=(
            "Save article artifacts and publish to Dev.to. "
            "You have exactly three tools: save_markdown_artifact, save_publication_manifest, publish_to_devto. "
            "Never call any other tool name. Never invent tool names."
        ),
        backstory=(
            "You are a disciplined publishing operator. You call tools by their exact registered names. "
            "To publish to Dev.to you MUST call the tool named publish_to_devto with arguments: "
            "title (str), markdown_content (str), tags_csv (str), cover_image_url (str), publish_now (str). "
            "You never call publish_article, post_article, or any other invented name."
        ),
        llm=llm,
        tools=[
            save_markdown_artifact,
            save_publication_manifest,
            publish_to_devto,
        ],
        verbose=True,
    )

    intake_task = Task(
        description=(
            "Technical document path: {technical_doc_path}\n\n"
            "1. Use read_technical_document with the provided file path.\n"
            "2. Produce a structured intake summary with:\n"
            "   - core topic\n"
            "   - 5 strongest insights\n"
            "   - 3 surprising details\n"
            "   - practical takeaways\n"
            "   - target audience\n"
            "3. Keep it factual and traceable to the source document."
        ),
        expected_output=(
            "A structured markdown summary with headings: Topic, Insights, Surprises, Takeaways, Audience."
        ),
        agent=intake_agent,
    )

    story_task = Task(
        description=(
            "Using the intake summary context:\n"
            "1. Generate 10 curiosity-driven title options with emojis.\n"
            "2. Choose 1 final title and explain why it wins.\n"
            "3. Create a section blueprint that follows this flow:\n"
            "   Hook -> Why it matters -> Mental model -> Deep dive -> Real-world implications -> CTA.\n"
            "4. Blueprint must target a substantial article (minimum 8 sections and 1200+ words).\n"
            "5. Ensure tone resembles high-engagement technical explainers.\n"
            "6. Do not copy any source article text verbatim from references.\n"
            "Tone reference: {tone_reference}."
        ),
        expected_output=(
            "A markdown document containing title shortlist, chosen title, and final narrative blueprint."
        ),
        agent=curiosity_editor,
        context=[intake_task],
    )

    package_task = Task(
        description=(
            "Create a Dev.to article package from prior context.\n"
            "Requirements:\n"
            "- Must be curiosity-driven, technical, and emoji-rich.\n"
            "- Must be extensive and comprehensive (minimum 1200 words, minimum 8 headings).\n"
            "- Must include YAML front matter with title, published, tags (max 4), and cover_image.\n"
            "- Add cover concept details and suggested tags.\n\n"
            "- Include at least one comparison table in markdown.\n"
            "- Mermaid image assets from source doc are provided in {mermaid_assets_json}.\n"
            "- For each Mermaid asset, embed an image in markdown using syntax: ![alt text](image_url).\n"
            "- Place each image near the relevant explanation section.\n"
            "- Do not output raw ```mermaid blocks in final article; use the image URLs instead.\n\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            "  \"title\": string,\n"
            "  \"devto_tags_csv\": string,\n"
            "  \"cover_image_url\": string,\n"
            "  \"devto_markdown\": string,\n"
            "  \"summary_markdown\": string\n"
            "}"
        ),
        expected_output="A strict JSON object containing a Dev.to-ready markdown article and metadata.",
        agent=channel_optimizer,
        context=[intake_task, story_task],
    )

    publish_task = Task(
        description=(
            "Parse the JSON from the previous task and execute these steps in order.\n"
            "\n"
            "Step 1 — Save article:\n"
            "  Call tool: save_markdown_artifact\n"
            "  Arguments: file_stem='devto-article', markdown_content=<devto_markdown from JSON>\n"
            "\n"
            "Step 2 — Save report:\n"
            "  Call tool: save_markdown_artifact\n"
            "  Arguments: file_stem='publication-report', markdown_content=<summary_markdown from JSON>\n"
            "\n"
            "Step 3 — Publish (only if publish_now is 'true'):\n"
            "  Call tool: publish_to_devto\n"
            "  Arguments:\n"
            "    title=<title from JSON>\n"
            "    markdown_content=<devto_markdown from JSON>\n"
            "    tags_csv=<devto_tags_csv from JSON>\n"
            "    cover_image_url=<cover_image_url from JSON, or empty string>\n"
            "    publish_now='{publish_now}'\n"
            "  IMPORTANT: the tool name is publish_to_devto — do NOT call publish_article or any other name.\n"
            "\n"
            "Step 4 — Save manifest:\n"
            "  Call tool: save_publication_manifest\n"
            "  Arguments: run_name='publish-run', manifest_json=<JSON string of all outcomes>\n"
            "\n"
            "Step 5 — Return a final markdown report containing:\n"
            "  - saved artifact file paths\n"
            "  - Dev.to publish status (success/failure)\n"
            "  - resulting article URL or explicit error message\n"
            "publish_now: {publish_now}"
        ),
        expected_output="A final markdown execution report with artifact paths and Dev.to publish outcome.",
        agent=publisher_agent,
        context=[package_task],
    )

    crew = Crew(
        agents=[intake_agent, curiosity_editor, channel_optimizer, publisher_agent],
        tasks=[intake_task, story_task, package_task, publish_task],
        process=Process.sequential,
        verbose=True,
    )

    response = crew.kickoff(
        inputs={
            "technical_doc_path": technical_doc_path,
            "publish_now": str(bool(publish_now)).lower(),
            "mermaid_assets_json": mermaid_assets_json,
            "mermaid_asset_count": str(len(mermaid_assets)),
            "tone_reference": tone_reference
            or "https://dev.to/databro/apache-parquet-file-anatomy-row-groups-column-chunks-pages-and-metadata-explained-4ebg",
        }
    )
    return str(response)
