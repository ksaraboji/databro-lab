"""
Gradio UI for Repository to Technical Documentation Agent
"""

import os

import gradio as gr
from dotenv import load_dotenv

from src.agent_workflow import (
    get_hf_api_key,
    generate_technical_docs,
    get_call_count,
    get_default_llm_backend,
    get_hf_model_name,
    get_ollama_base_url,
    get_ollama_model_name,
    get_openrouter_api_key,
    get_openrouter_model_name,
    cleanup_temp_repos,
)

load_dotenv()


def get_backend_status(backend: str) -> str:
    """Generate status message for selected backend."""
    if backend == "openrouter":
        model_name = get_openrouter_model_name()
        has_key = bool(get_openrouter_api_key())
        status = "🟢 Configured" if has_key else "🔴 Not configured (add OPENROUTER_API_KEY to .env)"
        return f"**OpenRouter** (`{model_name}`) | {status}"

    if backend == "ollama":
        model_name = get_ollama_model_name()
        base_url = get_ollama_base_url()
        return f"**Ollama Local** (`{model_name}` via `{base_url}`) | 🟢 Local endpoint configured"

    model_name = get_hf_model_name()
    has_key = bool(get_hf_api_key())
    status = "🟢 Configured" if has_key else "🔴 Not configured (add HF_API_TOKEN or HF_TOKEN to .env)"
    return f"**Hugging Face Router** (`{model_name}`) | {status}"


def get_model_options_for_backend(backend: str) -> list[str]:
    """Get available models for selected backend."""
    if backend == "openrouter":
        return ["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"]
    if backend == "ollama":
        return ["gemma4:latest", "gemma4:e2b-it-q4_K_M"]
    else:  # huggingface
        return ["google/gemma-4-31B-it", "google/gemma-4-26B-A4B-it"]


def get_default_model_for_backend(backend: str) -> str:
    """Resolve default model from env while staying within known options."""
    options = get_model_options_for_backend(backend)
    if backend == "openrouter":
        configured = get_openrouter_model_name()
    elif backend == "ollama":
        configured = get_ollama_model_name()
    else:
        configured = get_hf_model_name()
    return configured if configured in options else options[0]


def update_model_options(backend: str) -> dict:
    """Update model dropdown choices based on selected backend."""
    models = get_model_options_for_backend(backend)
    return gr.Dropdown(choices=models, value=get_default_model_for_backend(backend))


def get_counter_display() -> str:
    """Get formatted call counter display."""
    count = get_call_count()
    if count == 0:
        return "📊 **API Calls**: 0"
    else:
        return f"📊 **API Calls**: {count}"


def process_repo_url(
    github_url: str,
    llm_backend: str = "huggingface",
    llm_model: str = "google/gemma-4-31B-it",
) -> tuple[str, str]:
    """Process GitHub URL and generate technical documentation."""
    if not github_url or not github_url.strip():
        return "❌ Please enter a GitHub repository URL", get_counter_display()

    try:
        # Validate GitHub URL format
        if "github.com" not in github_url:
            return (
                "❌ Invalid GitHub URL. Expected format: https://github.com/owner/repo or "
                "https://github.com/owner/repo/tree/main/path",
                get_counter_display(),
            )

        # Show processing status
        status = f"⏳ Processing repository: {github_url}\nBackend: {llm_backend} | Model: {llm_model}"

        # Generate documentation
        docs = generate_technical_docs(
            github_url=github_url,
            llm_backend=llm_backend,
            model_override=llm_model,
        )

        # Clean up temporary repositories
        cleanup_temp_repos()

        return docs, get_counter_display()

    except Exception as e:
        error_msg = f"❌ Error processing repository:\n\n{str(e)}"
        cleanup_temp_repos()
        return error_msg, get_counter_display()


# Custom CSS for bright, modern UI
custom_css = """
:root {
    --primary-color: #10b981;
    --secondary-color: #f97316;
    --accent-color: #06b6d4;
}

body {
    background: linear-gradient(135deg, #f0f9ff 0%, #f0fdf4 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell;
}

.gradio-container {
    max-width: 1100px;
    margin: 0 auto !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 12px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08) !important;
}

.hero-section {
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
    color: white;
    padding: 40px 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    text-align: center;
}

.hero-section h1 {
    font-size: 2.5em;
    margin: 0 0 10px 0;
    font-weight: 700;
}

.hero-section p {
    font-size: 1.1em;
    margin: 0;
    opacity: 0.95;
}

.status-card {
    background: linear-gradient(135deg, #ecfdf5 0%, #cffafe 100%);
    border-left: 4px solid #10b981;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}

.control-group {
    background: #f9fafb;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
}

button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(16, 185, 129, 0.2) !important;
}

.footer {
    text-align: center;
    padding: 30px 20px;
    margin-top: 30px;
    color: #0f172a;
    max-width: 1100px;
    margin-left: auto;
    margin-right: auto;
    background: linear-gradient(135deg, rgba(187, 247, 208, 0.82), rgba(191, 219, 254, 0.82));
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 18px;
    box-shadow: 0 18px 40px rgba(148, 163, 184, 0.22);
}

.footer a {
    color: #10b981;
    text-decoration: none;
    font-weight: 600;
}

.footer a:hover {
    text-decoration: underline;
}

.output-box {
    background: #f3f4f6;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

.counter-display {
    background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
    border-left: 4px solid #06b6d4;
    padding: 12px;
    border-radius: 6px;
    font-weight: 600;
    color: #0369a1;
}
"""

# Build the UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    # Hero section
    with gr.Group(elem_classes="hero-section"):
        gr.Markdown(
            "# 📚 Repo → Technical Docs Agent\n\n"
            "Input a GitHub repository URL and get comprehensive technical documentation with "
            "architecture diagrams, flow charts, tech stack analysis, and more."
        )

    # Main content
    with gr.Group(elem_classes="control-group"):
        with gr.Row():
            github_url_input = gr.Textbox(
                label="🔗 GitHub Repository URL",
                placeholder="https://github.com/owner/repo or https://github.com/owner/repo/tree/main/src",
                lines=1,
            )

        with gr.Row():
            llm_backend = gr.Dropdown(
                choices=["huggingface", "openrouter", "ollama"],
                value=get_default_llm_backend(),
                label="🤖 LLM Backend",
                interactive=True,
            )
            llm_model = gr.Dropdown(
                choices=get_model_options_for_backend(get_default_llm_backend()),
                value=get_default_model_for_backend(get_default_llm_backend()),
                label="🧠 Model Name",
                interactive=True,
            )

        # Update model options when backend changes
        llm_backend.change(
            fn=update_model_options,
            inputs=llm_backend,
            outputs=llm_model,
        )

        backend_status = gr.Markdown(
            get_backend_status(get_default_llm_backend()),
            elem_classes="status-card",
        )

        llm_backend.change(
            fn=get_backend_status,
            inputs=llm_backend,
            outputs=backend_status,
        )

        with gr.Row():
            submit_btn = gr.Button("📖 Generate Documentation", size="lg")

    # Results section
    with gr.Group():
        call_counter = gr.Markdown(get_counter_display(), elem_classes="counter-display")
        documentation_output = gr.Markdown(
            value="## Documentation will appear here\n\nEnter a GitHub URL and click 'Generate Documentation'",
            elem_classes="output-box",
            label="📄 Technical Documentation",
        )

    # Submit handler
    submit_btn.click(
        fn=process_repo_url,
        inputs=[github_url_input, llm_backend, llm_model],
        outputs=[documentation_output, call_counter],
    )

    # Footer
    with gr.Group():
        gr.Markdown(
            '<div class="footer">'
            '<p>Built with ❤️ using CrewAI, Gradio, and Gemma models</p>'
            '<p><a href="https://databro.dev" target="_blank">Visit databro.dev</a></p>'
            "</div>",
            elem_classes="footer",
        )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
