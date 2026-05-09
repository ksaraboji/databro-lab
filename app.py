from pathlib import Path
import re

import gradio as gr
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from src.agent_workflow import (
    get_call_count,
    get_hf_api_key,
    get_hf_base_url,
    get_hf_model_name,
    run_agentic_query,
)


def get_huggingface_status() -> str:
    model_name = get_hf_model_name()
    hf_token = get_hf_api_key()
    hf_base_url = get_hf_base_url()

    if not hf_token:
        return (
            "### Hugging Face Status\n"
            f"Configured for `{model_name}` via `{hf_base_url}`, but no Hugging Face token was loaded.\n\n"
            "Set `HF_API_TOKEN` or `HF_TOKEN` in the `.env` file before submitting."
        )

    return (
        "### Hugging Face Status\n"
        f"Configured for `{model_name}` via `{hf_base_url}` with a Hugging Face token loaded."
    )


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


def chat_with_file(message: str, history, file_obj):
    if not file_obj:
        return "Please upload a supported file before asking questions.", None
    if not message.strip():
        return "Please enter a query.", None

    file_path = file_obj
    try:
        response_text = run_agentic_query(file_path=file_path, user_query=message)
        return response_text, _extract_download_path(response_text)
    except Exception as exc:
        error_text = str(exc)
        if "504" in error_text or "Gateway Timeout" in error_text:
            error_text += (
                "\n\nHugging Face endpoint timed out. Retry, or try a smaller/faster Gemma model in `HF_MODEL_NAME`."
            )
        return (
            "The agent could not complete the request. "
            "Ensure a Hugging Face token is set and the model name is valid.\n\n"
            f"Error: {error_text}"
        ), None


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
    gr.HTML(
        "<script>"
        "(() => {"
        "  const scrollOnce = () => {"
        "    document.documentElement.scrollTop = 0;"
        "    document.body.scrollTop = 0;"
        "    window.scrollTo(0, 0);"
        "  };"
        "  scrollOnce();"
        "  requestAnimationFrame(scrollOnce);"
        "  setTimeout(scrollOnce, 30);"
        "})();"
        "</script>"
    )
    
    gr.Markdown(
        "<div id='hero-banner'>"
        "<h1>✨ Chat with Your Data | Gemma4 + CrewAI</h1>"
        "<p>Upload CSV, JSON, NDJSON, Parquet, or Arrow files. Ask questions, export results, or convert formats with an agentic workflow.</p>"
        "</div>"
    )
    huggingface_status = gr.Markdown(elem_id="status-card")
    call_counter = gr.Markdown(elem_id="status-card", value="🤖 **Gemma4 API calls this session:** `0`")
    file_input = gr.File(label="Upload CSV, JSON, NDJSON, Parquet, or Arrow", type="filepath")
    download_output = gr.File(label="Generated file (when requested)")
    gr.Markdown("Start chatting below after uploading a file.")

    chat = gr.ChatInterface(
        fn=chat_with_file,
        textbox=gr.Textbox(
            placeholder="Ask a question about your data...",
            lines=1,
            submit_btn=True,
            autofocus=True,
        ),
        additional_inputs=[file_input],
        additional_outputs=[download_output],
        title="Data Chat",
        description="Ask multiple questions about the uploaded file in conversation style. When you ask for an updated file, the generated file appears below.",
    )

    interface.load(fn=get_huggingface_status, outputs=huggingface_status)
    interface.load(fn=get_counter_display, outputs=call_counter)
    chat.chatbot.change(fn=get_counter_display, outputs=call_counter)

    gr.Markdown(
        "<div id='footer-card'>"
        "<div style='text-align:center;'>Prototyped with ❤️ for <a href='https://databro.dev' target='_blank' rel='noopener noreferrer'>Databro</a>.</div>"
        "</div>"
    )

    _ = chat

interface.queue()


if __name__ == "__main__":
    # First-load-only scroll reset: force top at open, then allow normal scrolling.
    scroll_script = """
    <script>
    (() => {
      history.scrollRestoration = 'manual';

      const KEY = 'databro-initial-scroll-done';
      const resetTop = () => {
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
        window.scrollTo(0, 0);
      };

      if (sessionStorage.getItem(KEY) !== '1') {
        const delays = [0, 16, 60, 140, 260, 450];
        delays.forEach((ms) => setTimeout(resetTop, ms));
        sessionStorage.setItem(KEY, '1');
      } else {
        // On in-app refresh/navigation, do a lightweight one-shot reset only.
        setTimeout(resetTop, 0);
      }
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
