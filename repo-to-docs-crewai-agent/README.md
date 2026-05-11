# Repo → Technical Docs Agent

Generate comprehensive technical documentation from GitHub repositories using CrewAI and Gemma models.

## What it does

1. User provides a GitHub repository URL (optionally pointing to a specific folder).
2. CrewAI agent fetches the repository and analyzes:
   - Repository structure and components
   - Technology stack (languages, frameworks, databases, tools)
   - Dependencies and package managers
   - Entry points and main modules
   - Key configuration files
3. Agent generates professional technical documentation including:
   - **Tech Stack Overview**: Languages, frameworks, libraries, databases, tools
   - **Architecture Diagrams**: System design and component relationships (Mermaid)
   - **Flow Diagrams**: Request flows, data pipelines, initialization sequences (Mermaid)
   - **Component Descriptions**: Purpose and role of each major component
   - **Dependency Analysis**: Key libraries and their uses
   - **Configuration Information**: Setup and environment details
4. Output is a single Markdown document with embedded Mermaid diagrams.
5. Documentation is designed for test engineers to create test cases based on the technical design.

## Supported repositories

- **Python** projects (requirements.txt, pyproject.toml, setup.py)
- **JavaScript/Node.js** projects (package.json, npm/yarn/pnpm)
- **Java** projects (Maven, Gradle)
- **Go** projects (go.mod)
- **Mixed-language** monorepos

File detection is based on directory structure and package manager files, with smart skipping of non-essential directories (node_modules, __pycache__, .git, etc.).

## Workflow

The CrewAI crew runs with `Process.sequential` to ensure logical progression:

1. **Clone & Structure Task**: Clone repository, list structure, identify entry points
2. **Tech Stack Task**: Analyze dependencies and detect technology stack
3. **Architecture Task**: Generate system architecture and component diagrams (Mermaid)
4. **Flow Task**: Generate request/data flow diagrams (Mermaid)
5. **Compilation Task**: Combine all analysis into comprehensive Markdown documentation

Each task depends on previous results, ensuring accurate context for diagram generation and documentation writing.

## Setup

### 1. Create and activate a Python environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and configure **at least one backend**:

**Shared settings:**
```bash
# Backend to use by default when the app starts
LLM_BACKEND=huggingface

# Token limit for all LLM calls (optional, default: 2048)
LLM_MAX_TOKENS=2048
```

**Hugging Face Router** (required if using HF):
```bash
HF_API_TOKEN=your_hugging_face_api_token
HF_MODEL_NAME=google/gemma-4-31B-it
HF_BASE_URL=https://router.huggingface.co/v1
```

Get your HF token from: https://huggingface.co/settings/tokens

**OpenRouter** (required if using OpenRouter):
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL_NAME=google/gemma-4-31b-it:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Get your OpenRouter key from: https://openrouter.ai/

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

You can configure both backends and switch between them in the UI, or configure just one.

### 4. Start the app

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Using the app

### Basic workflow

1. **Enter GitHub URL**: Paste a repository URL:
   - Full repo: `https://github.com/owner/repo`
   - Specific folder: `https://github.com/owner/repo/tree/main/src`

2. **Select LLM Backend**: Choose between:
   - **Hugging Face Router**: Free tier, shared queue, good for prototyping
   - **OpenRouter**: Unified API, typically better throughput for production
   - **Ollama (Local)**: Local inference via your own machine

3. **Choose Model** (optional): Pick a model variant for your selected backend:
   - **HF models**: `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`
   - **OpenRouter models**: `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`
   - **Ollama models**: `gemma4:latest`, `gemma4:e2b-it-q4_K_M`

4. **Generate Documentation**: Click "Generate Documentation" and wait for analysis to complete.

5. **Review Output**: Technical documentation appears in Markdown with embedded diagrams.

6. **Track API Usage**: Watch the API call counter to monitor LLM backend usage.

### Tips

- **Large repositories**: The agent intelligently samples key files and directories
- **Specific folders**: Include `/tree/branch/path` in URL to analyze a subfolder
- **Private repos**: Not currently supported; public repositories only
- **Mermaid diagrams**: Rendered inline in Markdown, can be copied/edited
- **Test suite integration**: Copy Markdown content for test case generation agents

## Understanding the API call counter

The app displays an **API Call Counter** tracking LLM API attempts. This differs from chat requests:

- **Why multiple calls for one analysis?**
  - Five sequential CrewAI tasks, each using the LLM
  - Internal reasoning and diagram generation require multiple attempts
  - Retry logic for resilience

- **Typical count**: 8-15+ API calls per repository analysis
  - This aligns with provider dashboards (Hugging Face or OpenRouter)

- **What you're paying for**: Each API call on provider dashboards matches the counter

## Outputs

The generated documentation includes:

1. **Overview section**: Brief project description
2. **Tech Stack table**: Languages, frameworks, databases, tools
3. **Architecture section**:
   - System architecture diagram (Mermaid)
   - Component descriptions
   - External dependencies
4. **Data Flow section**:
   - Request flow diagram (Mermaid)
   - Data processing flow (Mermaid)
   - Application startup flow (Mermaid)
5. **Dependencies section**: Key libraries and purposes
6. **Configuration section**: Entry points, environment setup
7. **Test-relevant notes**: Information for test case generation

All content is in Markdown with embeddable Mermaid diagrams.

## Run from GitHub

```bash
git clone https://github.com/ksaraboji/databro-lab.git
cd databro-lab/repo-to-docs-agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API credentials

python app.py
```

## Sharing and deployment

- **Temporary sharing**: The app launches with Gradio `share=True`, generating a temporary public URL
- **Persistent hosting**: Deploy to Hugging Face Spaces or another Python hosting platform
- **Environment variables**: Provide the same `.env` configuration in your deployment platform

## Technical architecture

- **Framework**: CrewAI 0.102.0+ with five sequential tasks
- **LLM Backends**:
  - Hugging Face: `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`
  - OpenRouter: `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`
   - Ollama: `gemma4:latest`, `gemma4:e2b-it-q4_K_M`
- **UI**: Gradio 5.x with bright emerald/cyan theme
- **Repository Analysis**: GitPython for cloning, content-based detection
- **Diagrams**: Mermaid syntax embedded in Markdown
- **Call Tracking**: CrewAI LLM event hooks (per-attempt counting)

## Tools available to the agent

1. **`clone_github_repo`**: Fetch repository, support for specific folders/branches
2. **`detect_tech_stack`**: Identify languages, frameworks, databases, tools
3. **`analyze_dependencies`**: Parse package managers (pip, npm, Maven, Gradle, Bundler)
4. **`list_repo_structure`**: Generate directory tree (limited depth and breadth)
5. **`extract_entry_points`**: Identify main.py, app.js, package.json, etc.
6. **`read_key_files`**: Extract README, ARCHITECTURE, docker-compose, .env.example

## Limitations

- **Public repositories only**: No support for private repos yet
- **Large monorepos**: Agent intelligently samples; may miss some details
- **Complex frameworks**: Inferred from code patterns and dependencies
- **Diagram accuracy**: Generated based on code structure; may need manual refinement

## Notes

- Clone and analysis happens in temporary directories; automatically cleaned up
- API keys are never logged or exposed in output
- Markdown output is suitable for Git-based documentation
- Mermaid diagrams render natively in GitHub, GitLab, Notion, and many Markdown viewers

## Future improvements

- Support for private repositories (with GitHub token)
- Incremental analysis (update existing docs with changes)
- Custom prompt templates for specific documentation styles
- Export to other formats (PDF, HTML, Confluence)
- Integration with test suite generation agents
