import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import requests
from crewai.tools import tool

OUTPUT_ROOT = Path(tempfile.gettempdir()) / "tech-doc-to-publish-crewai-agent"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_name(raw_name: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    cleaned = "".join(ch if ch in allowed else "-" for ch in raw_name.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "artifact"


def _parse_tags(tags_csv: str) -> list[str]:
    import re
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in tags_csv.split(","):
        # Dev.to tags: lowercase, alphanumeric only, no spaces or special chars
        sanitized = re.sub(r"[^a-z0-9]", "", raw.strip().lower())
        if sanitized and sanitized not in seen:
            deduped.append(sanitized)
            seen.add(sanitized)
    return deduped


@tool("read_technical_document")
def read_technical_document(file_path: str) -> str:
    """Read an uploaded technical documentation file and return metadata plus full content."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return json.dumps(
        {
            "file_name": path.name,
            "file_path": str(path),
            "char_count": len(content),
            "line_count": content.count("\n") + 1,
            "content": content,
        },
        indent=2,
    )


@tool("save_markdown_artifact")
def save_markdown_artifact(file_stem: str, markdown_content: str) -> str:
    """Save markdown content to a local file and return its absolute path."""
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_stem = _safe_name(file_stem)
    file_name = f"{timestamp}-{safe_stem}-{uuid.uuid4().hex[:8]}.md"
    output_path = OUTPUT_ROOT / file_name
    output_path.write_text(markdown_content, encoding="utf-8")
    return json.dumps({"output_path": str(output_path)}, indent=2)


@tool("save_publication_manifest")
def save_publication_manifest(run_name: str, manifest_json: str) -> str:
    """Persist a publication manifest JSON string and return the file path."""
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_name = _safe_name(run_name)
    file_name = f"{timestamp}-{safe_name}-{uuid.uuid4().hex[:8]}.json"
    output_path = OUTPUT_ROOT / file_name
    output_path.write_text(manifest_json, encoding="utf-8")
    return json.dumps({"output_path": str(output_path)}, indent=2)


@tool("publish_to_devto")
def publish_to_devto(
    title: str,
    markdown_content: str,
    tags_csv: str,
    cover_image_url: str = "",
    publish_now: str = "false",
) -> str:
    """Publish a markdown article to Dev.to using DEVTO_API_KEY from environment."""
    api_key = os.getenv("DEVTO_API_KEY")
    if not api_key:
        return json.dumps(
            {
                "platform": "devto",
                "success": False,
                "error": "DEVTO_API_KEY is not configured.",
            },
            indent=2,
        )

    tags = _parse_tags(tags_csv)[:4]
    payload = {
        "article": {
            "title": title,
            "published": publish_now.strip().lower() == "true",
            "body_markdown": markdown_content,
            "tags": tags,
        }
    }
    if cover_image_url.strip():
        payload["article"]["main_image"] = cover_image_url.strip()

    response = requests.post(
        "https://dev.to/api/articles",
        headers={"api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )

    if response.status_code >= 400:
        return json.dumps(
            {
                "platform": "devto",
                "success": False,
                "status_code": response.status_code,
                "error": response.text,
            },
            indent=2,
        )

    data = response.json()
    return json.dumps(
        {
            "platform": "devto",
            "success": True,
            "status_code": response.status_code,
            "id": data.get("id"),
            "url": data.get("url"),
            "path": data.get("path"),
        },
        indent=2,
    )



