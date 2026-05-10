# databro-lab

> Python-first prototyping lab for [databro.dev](https://databro.dev) — rapid experiments built with Gradio, Streamlit, and other modern UI stacks before production implementation.

`databro-lab` is the working sandbox for exploratory builds, proof-of-concepts, and rapid interface experiments that help shape features for [databro.dev](https://databro.dev).

The primary focus of this repository is Python-based prototyping using frameworks such as **Gradio**, **Streamlit**, and other leading UI stacks for data apps, AI workflows, interactive demos, and developer tooling.

## Purpose

This repository exists to validate ideas quickly before investing in production engineering. It is the place where concepts are tested, interaction patterns are explored, and technical approaches are compared before anything is hardened for the main portfolio site.

## How this repo is organized

Each folder in this repository should be treated as a **self-contained spike**. A folder represents one prototype, one experiment, one idea, or one implementation direction being evaluated on its own.

That means a folder may contain:

- A standalone Gradio prototype
- A Streamlit-based data app
- A Python-first UI experiment
- A proof of concept for an AI workflow
- A comparison spike for evaluating a framework or interaction pattern

The important principle is that **every folder stands on its own** as an isolated experiment rather than as part of a tightly coupled monorepo structure.

## What kind of work belongs here

- Rapid Python prototypes
- Gradio demos and model interfaces
- Streamlit dashboards and interactive tools
- Experimental UI ideas for data and AI products
- Framework evaluation spikes
- Pre-production concepts that may later move into `databro.dev`

## Working style

Code in this repository is intentionally experimental. It may be incomplete, rough around the edges, or optimized for speed of learning rather than production readiness.

A successful spike should help answer questions such as:

- Is this idea worth pursuing?
- Does this UI approach feel right?
- Is this framework a good fit?
- Should this concept graduate into the production portfolio site?

## Relationship to databro.dev

`databro-lab` is the experimentation layer.  
`databro.dev` is the polished production-facing portfolio site.

Work usually starts here as an isolated spike, gets evaluated, and only then moves into the main site if it proves useful enough to productionize.

## Notes

Where useful, each spike should document:

- What problem it is exploring
- Why a particular framework was chosen
- Key findings and trade-offs
- Whether the idea should move forward, be reworked, or be dropped

---

Built by [Kumaravelu Saraboji Mahalingam](https://github.com/ksaraboji)
