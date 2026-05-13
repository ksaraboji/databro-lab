"""Gradio UI for tech-doc-to-publish agentic spike."""

import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from src.agent_workflow import (
    generate_publishable_articles,
    get_call_count,
    get_default_llm_backend,
    get_hf_api_key,
    get_hf_model_name,
    get_ollama_base_url,
    get_ollama_model_name,
    get_openrouter_api_key,
    get_openrouter_model_name,
)

load_dotenv()

HF_MODEL_OPTIONS = [
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
]

OPENROUTER_MODEL_OPTIONS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]

OLLAMA_MODEL_OPTIONS = [
    "gemma4:latest",
    "gemma4:e2b-it-q4_K_M",
]


def _status_for_backend(backend: str, selected_model: str | None = None) -> str:
    active_model = (selected_model or "").strip()

    if backend == "ollama":
        model_name = active_model or get_ollama_model_name()
        base_url = get_ollama_base_url()
        return (
            "### Ollama Status\n"
            f"Configured for local model `{model_name}` via `{base_url}`."
        )

    if backend == "openrouter":
        model_name = active_model or get_openrouter_model_name()
        token = get_openrouter_api_key()
        if not token:
            return (
                "### OpenRouter Status\n"
                f"Configured for `{model_name}`, but no OpenRouter token was loaded."
            )
        return f"### OpenRouter Status\nConfigured for `{model_name}` with token loaded."

    model_name = active_model or get_hf_model_name()
    token = get_hf_api_key()
    if not token:
        return (
            "### Hugging Face Status\n"
            f"Configured for `{model_name}`, but no Hugging Face token was loaded."
        )
    return f"### Hugging Face Status\nConfigured for `{model_name}` with token loaded."


def get_backend_status(selected_backend: str, selected_model: str | None = None) -> str:
    backend = (selected_backend or get_default_llm_backend()).strip().lower()
    return _status_for_backend(backend, selected_model)


def get_default_model_for_backend(selected_backend: str) -> str:
    backend = (selected_backend or get_default_llm_backend()).strip().lower()
    if backend == "ollama":
        configured = get_ollama_model_name()
        return configured if configured in OLLAMA_MODEL_OPTIONS else OLLAMA_MODEL_OPTIONS[0]
    if backend == "openrouter":
        configured = get_openrouter_model_name()
        return configured if configured in OPENROUTER_MODEL_OPTIONS else OPENROUTER_MODEL_OPTIONS[0]
    configured = get_hf_model_name()
    return configured if configured in HF_MODEL_OPTIONS else HF_MODEL_OPTIONS[0]


def get_model_options_for_backend(selected_backend: str) -> list[str]:
    backend = (selected_backend or get_default_llm_backend()).strip().lower()
    if backend == "ollama":
        return OLLAMA_MODEL_OPTIONS
    return OPENROUTER_MODEL_OPTIONS if backend == "openrouter" else HF_MODEL_OPTIONS


def get_model_dropdown_update(selected_backend: str):
    options = get_model_options_for_backend(selected_backend)
    return gr.update(choices=options, value=get_default_model_for_backend(selected_backend))


def _extract_paths(response_text: str) -> list[str]:
    matches = re.findall(r"/[^\s\"'`]+\.(?:md|json)", response_text)
    ordered: list[str] = []
    for candidate in matches:
        if Path(candidate).exists() and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def get_counter_display() -> str:
    return f"🤖 **Gemma4 API calls this session:** `{get_call_count()}`"


def process_technical_doc(
    document_file,
    llm_backend: str,
    llm_model: str,
    publish_now: bool,
    tone_reference: str,
) -> tuple[str, str, list[str]]:
    if not document_file:
        return "Please upload a technical documentation file first.", get_counter_display(), []

    try:
        backend = (llm_backend or get_default_llm_backend()).strip().lower()
        model_override = (llm_model or "").strip() or None

        response_text = generate_publishable_articles(
            technical_doc_path=document_file,
            llm_backend=backend,
            model_override=model_override,
            publish_now=bool(publish_now),
            tone_reference=(tone_reference or "").strip() or None,
        )
        return response_text, get_counter_display(), _extract_paths(response_text)
    except Exception as exc:
        return (
            "The agentic workflow failed. Check model/provider tokens and publishing keys.\n\n"
            f"Error: {exc}",
            get_counter_display(),
            [],
        )


ui_css = """
body {
    background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.12), transparent 26%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 22%),
        linear-gradient(180deg, #f8fbff 0%, #edf7ff 48%, #fefefe 100%);
}

.gradio-container {
    max-width: 1100px;
    margin: 0 auto !important;
}

#hero-banner {
    background: linear-gradient(135deg, rgba(187, 247, 208, 0.82), rgba(191, 219, 254, 0.82));
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 20px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 18px 40px rgba(148, 163, 184, 0.22);
}

#panel {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.84);
    backdrop-filter: blur(8px);
    box-shadow: 0 14px 32px rgba(148, 163, 184, 0.14);
    padding: 0.95rem;
}

.gr-button-primary {
    background: linear-gradient(135deg, #10b981, #06b6d4) !important;
    border: none !important;
}
"""

with gr.Blocks(title="Tech Doc -> Publish | CrewAI", css=ui_css) as interface:
    gr.Markdown(
        "<div id='hero-banner'>"
        "<h1>📝✨ Technical Doc -> Dev.to Publisher</h1>"
        "<p>Upload technical documentation, generate curiosity-driven emoji-rich articles, and optionally publish through a fully agentic workflow.</p>"
        "</div>"
    )

    with gr.Column(elem_id="panel"):
        with gr.Row():
            llm_backend = gr.Dropdown(
                choices=["huggingface", "openrouter", "ollama"],
                value=get_default_llm_backend(),
                label="LLM Backend",
            )
            llm_model = gr.Dropdown(
                choices=get_model_options_for_backend(get_default_llm_backend()),
                value=get_default_model_for_backend(get_default_llm_backend()),
                label="Model",
            )

        backend_status = gr.Markdown(value=get_backend_status(get_default_llm_backend(), get_default_model_for_backend(get_default_llm_backend())))
        call_counter = gr.Markdown(value=get_counter_display())

        technical_doc_file = gr.File(
            label="Upload technical documentation (.md/.txt)",
            type="filepath",
        )
        tone_reference = gr.Textbox(
            label="Tone reference URL (optional)",
            value="https://dev.to/databro/apache-parquet-file-anatomy-row-groups-column-chunks-pages-and-metadata-explained-4ebg",
            lines=1,
        )
        publish_now = gr.Checkbox(
            label="Publish now (unchecked = draft/manual review mode)",
            value=False,
        )

        run_button = gr.Button("🚀 Run Agentic Publishing Workflow")

        result_markdown = gr.Markdown(
            value="Workflow output and publish status will appear here.",
            label="Run Output",
        )
        artifacts = gr.Files(label="Generated artifacts")

    llm_backend.change(fn=get_backend_status, inputs=[llm_backend, llm_model], outputs=backend_status)
    llm_backend.change(fn=get_model_dropdown_update, inputs=llm_backend, outputs=llm_model)
    llm_model.change(fn=get_backend_status, inputs=[llm_backend, llm_model], outputs=backend_status)

    run_button.click(
        fn=process_technical_doc,
        inputs=[technical_doc_file, llm_backend, llm_model, publish_now, tone_reference],
        outputs=[result_markdown, call_counter, artifacts],
    )

interface.queue()

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
