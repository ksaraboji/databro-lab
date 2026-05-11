from pathlib import Path
import re

import gradio as gr
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from src.agent_workflow import (
    get_call_count,
    get_default_llm_backend,
    get_hf_api_key,
    get_hf_base_url,
    get_hf_model_name,
    get_ollama_base_url,
    get_ollama_model_name,
    get_openrouter_api_key,
    get_openrouter_base_url,
    get_openrouter_model_name,
    run_agentic_query,
)


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
            f"Configured for local model `{model_name}` via `{base_url}`. "
            "Ensure Ollama is running and the model is pulled."
        )

    if backend == "openrouter":
        model_name = active_model or get_openrouter_model_name()
        token = get_openrouter_api_key()
        base_url = get_openrouter_base_url()
        if not token:
            return (
                "### OpenRouter Status\n"
                f"Configured for `{model_name}` via `{base_url}`, but no OpenRouter token was loaded.\n\n"
                "Set `OPENROUTER_API_KEY` in the `.env` file before submitting."
            )
        return (
            "### OpenRouter Status\n"
            f"Configured for `{model_name}` via `{base_url}` with an OpenRouter token loaded."
        )

    model_name = active_model or get_hf_model_name()
    token = get_hf_api_key()
    base_url = get_hf_base_url()
    if not token:
        return (
            "### Hugging Face Status\n"
            f"Configured for `{model_name}` via `{base_url}`, but no Hugging Face token was loaded.\n\n"
            "Set `HF_API_TOKEN` or `HF_TOKEN` in the `.env` file before submitting."
        )
    return (
        "### Hugging Face Status\n"
        f"Configured for `{model_name}` via `{base_url}` with a Hugging Face token loaded."
    )


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


def _extract_download_path(response_text: str) -> str | None:
    # Find a local exported csv/json/parquet/arrow path produced by the export tool.
    matches = re.findall(r"/[^\s\"'`]+\.(?:csv|json|ndjson|jsonl|parquet|pq|arrow|ipc)", response_text)
    for candidate in matches:
        if Path(candidate).exists():
            return candidate
    return None


def get_counter_display() -> str:
    count = get_call_count()
    return f"🤖 **Gemma4 API calls this session:** `{count}`"


def chat_with_file(message: str, history, file_obj, llm_backend: str, llm_model: str):
    if not file_obj:
        return "Please upload a supported file before asking questions.", None
    if not message.strip():
        return "Please enter a query.", None

    file_path = file_obj
    try:
        backend = (llm_backend or get_default_llm_backend()).strip().lower()
        model_override = (llm_model or "").strip() or None
        response_text = run_agentic_query(
            file_path=file_path,
            user_query=message,
            llm_backend=backend,
            model_override=model_override,
        )
        return response_text, _extract_download_path(response_text)
    except Exception as exc:
        error_text = str(exc)
        backend = (llm_backend or get_default_llm_backend()).strip().lower()

        if backend == "ollama" and (
            "Failed to connect" in error_text
            or "Connection error" in error_text
            or "OpenAI API call failed" in error_text
        ):
            return (
                "The agent could not reach your local Ollama server.\n\n"
                "Check these settings and retry:\n"
                f"- Backend: `{backend}`\n"
                f"- Model: `{(llm_model or '').strip() or 'gemma4:latest'}`\n"
                "- OLLAMA_BASE_URL: `http://localhost:11434` (or a host reachable from this runtime)\n"
                "- Ollama process must be running where this app runs\n\n"
                f"Raw error: {error_text}"
            ), None

        if "504" in error_text or "Gateway Timeout" in error_text:
            error_text += (
                "\n\nLLM endpoint timed out. Retry, or try a smaller/faster model for the selected backend."
            )
        return (
            "The agent could not complete the request. "
            "Ensure the selected backend token is set and the model name is valid.\n\n"
            f"Error: {error_text}"
        ), None


def reset_chat_session():
    return [], None, "### Chat ended\nStart a new chat by uploading a file and asking a question."


ui_css = """
body {
    background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 22%),
        linear-gradient(180deg, #f8fbff 0%, #edf7ff 48%, #fefefe 100%);
}

.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #0f172a;
    overflow-anchor: none;
}

html, body {
    scroll-behavior: auto;
}

#app-shell {
    max-width: 1100px;
    margin: 0 auto;
    padding-bottom: 1.5rem;
}

#control-panel, #chat-panel, #upload-panel, #generated-panel {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.82);
    backdrop-filter: blur(12px);
    box-shadow: 0 14px 32px rgba(148, 163, 184, 0.14);
    padding: 0.9rem;
}

#upload-panel, #generated-panel {
    min-height: 320px;
    display: flex;
    flex-direction: column;
}

#control-panel h3, #chat-panel h3, #upload-panel h3, #generated-panel h3 {
    margin: 0;
}

.control-help {
    margin-top: 0.2rem;
    color: #334155;
    font-size: 0.92rem;
}

.button-row {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-top: 0.3rem;
}

#chat-panel [data-testid="chatbot"] {
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.9);
}

#chat-panel form {
    border-top: 1px solid rgba(148, 163, 184, 0.24);
    background: rgba(241, 245, 249, 0.7);
    border-radius: 0 0 16px 16px;
}

#chat-panel textarea {
    background: rgba(241, 245, 249, 0.9) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
}

#chat-end-btn {
    border: 1px solid rgba(15, 23, 42, 0.18) !important;
}

#session-note {
    margin-top: 0.4rem;
}

#hero-banner {
    background: linear-gradient(135deg, rgba(187, 247, 208, 0.82), rgba(191, 219, 254, 0.82));
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 20px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 18px 40px rgba(148, 163, 184, 0.22);
}

#hero-banner h1 {
    margin: 0;
    font-size: 2rem;
    line-height: 1.15;
    color: #0f172a;
}

#hero-banner p {
    margin: 0.4rem 0 0;
    color: #334155;
}

#status-card, #footer-card {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.80);
    backdrop-filter: blur(12px);
    padding: 0.35rem 0.8rem;
    box-shadow: 0 10px 28px rgba(148, 163, 184, 0.12);
}

#footer-card {
    margin-top: 1rem;
    max-width: 1100px;
    margin-left: auto;
    margin-right: auto;
    background: linear-gradient(135deg, rgba(187, 247, 208, 0.82), rgba(191, 219, 254, 0.82));
    border: 1px solid rgba(148, 163, 184, 0.28);
    box-shadow: 0 18px 40px rgba(148, 163, 184, 0.22);
}

.gr-button-primary {
    background: linear-gradient(135deg, #10b981, #06b6d4) !important;
    border: none !important;
}

a {
    color: #0284c7;
}
"""


with gr.Blocks(
    title="Chat with Your Data | Gemma4 + CrewAI",
) as interface:

    with gr.Column(elem_id="app-shell"):
        gr.Markdown(
            "<div id='hero-banner'>"
            "<h1>✨ Chat with Your Data | Gemma4 + CrewAI</h1>"
            "<p>Upload CSV, JSON, NDJSON, Parquet, or Arrow files. Choose Hugging Face, OpenRouter, or local Ollama, then ask questions, export results, or convert formats with an agentic workflow.</p>"
            "</div>"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_id="control-panel"):
                gr.Markdown("### Session Setup")
                gr.Markdown("<div class='control-help'>Pick your backend, model, and file before chatting.</div>")
                llm_backend = gr.Dropdown(
                    choices=["huggingface", "openrouter", "ollama"],
                    value=get_default_llm_backend(),
                    label="LLM Backend",
                    info="Choose which API gateway to use for model calls.",
                )
                llm_model = gr.Dropdown(
                    choices=get_model_options_for_backend(get_default_llm_backend()),
                    value=get_default_model_for_backend(get_default_llm_backend()),
                    label="Model Name",
                    info="Select a model for the selected backend.",
                )

            with gr.Column(scale=3):
                huggingface_status = gr.Markdown(elem_id="status-card")
                call_counter = gr.Markdown(elem_id="status-card", value="🤖 **Gemma4 API calls this session:** `0`")
                session_note = gr.Markdown(
                    "### Session active\nYour chat context is currently active.",
                    elem_id="status-card",
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=4, elem_id="upload-panel"):
                gr.Markdown("### Upload File")
                gr.Markdown("<div class='control-help'>Upload one supported file to start or continue your chat session.</div>")
                file_input = gr.File(label="Upload CSV, JSON, NDJSON, Parquet, or Arrow", type="filepath")

            with gr.Column(scale=3, elem_id="generated-panel"):
                gr.Markdown("### Generated File")
                gr.Markdown("<div class='control-help'>When you request export or conversion, the generated file appears here.</div>")
                download_output = gr.File(label="Generated file (when requested)")

        with gr.Column(elem_id="chat-panel"):
            gr.Markdown("### Data Chat")
            gr.Markdown(
                "<div class='control-help'>Ask multiple questions on the uploaded file. "
                "Request exports or format conversions when needed.</div>"
            )

            chat = gr.ChatInterface(
                fn=chat_with_file,
                textbox=gr.Textbox(
                    placeholder="Ask a question about your data...",
                    lines=1,
                    submit_btn=True,
                    autofocus=False,
                ),
                additional_inputs=[file_input, llm_backend, llm_model],
                additional_outputs=[download_output],
                title=None,
                description=None,
            )

            with gr.Row(elem_classes=["button-row"]):
                end_chat_btn = gr.Button("End Chat", elem_id="chat-end-btn")
                gr.Markdown("<div id='session-note' class='control-help'>End Chat clears conversation history and generated-file output.</div>")

    interface.load(
        fn=get_backend_status,
        inputs=[llm_backend, llm_model],
        outputs=huggingface_status,
    )
    llm_backend.change(fn=get_backend_status, inputs=[llm_backend, llm_model], outputs=huggingface_status)
    llm_backend.change(fn=get_model_dropdown_update, inputs=llm_backend, outputs=llm_model)
    llm_model.change(fn=get_backend_status, inputs=[llm_backend, llm_model], outputs=huggingface_status)
    interface.load(fn=get_counter_display, outputs=call_counter)
    chat.chatbot.change(fn=get_counter_display, outputs=call_counter)
    end_chat_btn.click(
        fn=reset_chat_session,
        outputs=[chat.chatbot, download_output, session_note],
    )

    gr.Markdown(
        "<div id='footer-card'>"
        "<div style='text-align:center;'>Prototyped with ❤️ for <a href='https://databro.dev' target='_blank' rel='noopener noreferrer'>Databro</a>.</div>"
        "</div>"
    )

    _ = chat

interface.queue()


if __name__ == "__main__":
        # Reset scroll to top on first load and refresh; avoid preserving a deep scroll position.
    scroll_script = """
    <script>
    (() => {
      const resetTop = () => {
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
        window.scrollTo(0, 0);
      };

            if ('scrollRestoration' in history) {
                history.scrollRestoration = 'manual';
            }

            const scheduleResets = () => {
                [0, 25, 75, 160, 280, 450, 700, 1000].forEach((ms) => setTimeout(resetTop, ms));
            };

            window.addEventListener('load', scheduleResets);
            document.addEventListener('DOMContentLoaded', scheduleResets);

            window.addEventListener('pageshow', resetTop);
            window.addEventListener('beforeunload', resetTop);

            // Gradio can reflow after hydration; pin top briefly on first paint cycle.
            let ticks = 0;
            const pinTop = setInterval(() => {
                resetTop();
                ticks += 1;
                if (ticks >= 20) {
                    clearInterval(pinTop);
                }
            }, 80);
    })();
    </script>
    """
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="cyan", neutral_hue="slate"),
        css=ui_css,
        head=scroll_script,
    )
