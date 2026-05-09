# Databro Lab

This repository is a container for multiple spike projects and prototypes.

## Prototypes

- `ask-data-using-gemma4-crewai/`
  Agentic data chat prototype built with Gradio, CrewAI, DuckDB, and Hugging Face Gemma.

## Working With A Prototype

1. Clone the repository:

```bash
git clone https://github.com/ksaraboji/databro-lab.git
cd databro-lab
```

2. Change into the prototype folder you want to run.

Example:

```bash
cd ask-data-using-gemma4-crewai
```

3. Follow that prototype's local README for setup, environment variables, and launch steps.

## Conventions

- Each prototype keeps its own code, dependencies, and README inside its own subfolder.
- The root README stays short and acts as an index for all spikes.
- Local `.env` files, Gradio state, and Python cache directories stay ignored.
