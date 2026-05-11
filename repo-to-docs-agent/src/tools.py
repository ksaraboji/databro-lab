"""
Tools for analyzing GitHub repositories and extracting technical information.
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from crewai.tools import tool


def _parse_requirements_txt(file_path: str) -> list[str]:
    """Parse Python requirements.txt and extract package names."""
    packages = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name (before ==, >=, etc.)
                    pkg = re.split(r"[<>=!~]", line)[0].strip()
                    if pkg:
                        packages.append(pkg)
    except Exception:
        pass
    return packages


def _parse_package_json(file_path: str) -> list[str]:
    """Parse package.json and extract dependencies."""
    packages = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for key in ["dependencies", "devDependencies", "peerDependencies"]:
                if key in data:
                    packages.extend(data[key].keys())
    except Exception:
        pass
    return packages


def _parse_pom_xml(file_path: str) -> list[str]:
    """Parse Maven pom.xml and extract dependencies."""
    packages = []
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}
        for dep in root.findall(".//m:dependency/m:artifactId", ns):
            if dep.text:
                packages.append(dep.text)
    except Exception:
        pass
    return packages


def _parse_go_mod(file_path: str) -> list[str]:
    """Parse go.mod and extract dependencies."""
    packages = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            in_require = False
            for line in f:
                line = line.strip()
                if line == "require (":
                    in_require = True
                elif line == ")":
                    in_require = False
                elif in_require and line:
                    parts = line.split()
                    if parts:
                        packages.append(parts[0])
    except Exception:
        pass
    return packages


def _detect_tech_stack(repo_root: str) -> dict:
    """Detect technology stack from directory contents."""
    tech = {
        "languages": set(),
        "frameworks": set(),
        "databases": set(),
        "tools": set(),
        "package_managers": set(),
    }

    # File type indicators
    file_indicators = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".jsx": "React",
        ".tsx": "React+TypeScript",
        ".go": "Go",
        ".java": "Java",
        ".cs": "C#",
        ".rb": "Ruby",
        ".php": "PHP",
        ".rs": "Rust",
    }

    # Framework/library indicators
    framework_indicators = {
        "requirements.txt": "pip",
        "pyproject.toml": "Python/Poetry",
        "setup.py": "Python/setuptools",
        "package.json": "Node.js",
        "yarn.lock": "Yarn",
        "pnpm-lock.yaml": "pnpm",
        "package-lock.json": "npm",
        "go.mod": "Go Modules",
        "Gemfile": "Ruby/Bundler",
        "Cargo.toml": "Rust/Cargo",
        "pom.xml": "Maven",
        "build.gradle": "Gradle",
        "docker-compose.yml": "Docker",
        "Dockerfile": "Docker",
        ".github/workflows": "GitHub Actions",
    }

    try:
        for root, dirs, files in os.walk(repo_root):
            # Skip hidden and common non-essential directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in {"node_modules", "__pycache__", ".git", "venv", "vendor"}
            ]

            for file in files:
                # Detect by extension
                for ext, lang in file_indicators.items():
                    if file.endswith(ext):
                        tech["languages"].add(lang)

                # Detect by filename
                for filename, indicator in framework_indicators.items():
                    if file == filename or file.endswith(filename):
                        parts = indicator.split("/")
                        if len(parts) == 2:
                            tech["languages"].add(parts[0])
                            tech["tools"].add(parts[1])
                        else:
                            tech["tools"].add(indicator)

                # Detect databases
                db_indicators = {
                    "postgres": "PostgreSQL",
                    "mysql": "MySQL",
                    "mongodb": "MongoDB",
                    "redis": "Redis",
                    "sqlite": "SQLite",
                    "elasticsearch": "Elasticsearch",
                    "dynamodb": "DynamoDB",
                }

                for db_key, db_name in db_indicators.items():
                    if db_key in file.lower():
                        tech["databases"].add(db_name)

    except Exception:
        pass

    return {
        "languages": sorted(list(tech["languages"])),
        "frameworks": sorted(list(tech["frameworks"])),
        "databases": sorted(list(tech["databases"])),
        "tools": sorted(list(tech["tools"])),
    }


def _list_directory_tree(
    path: str, max_files: int = 100, prefix: str = ""
) -> str:
    """Generate a directory tree structure (limited depth)."""
    items = []
    count = 0

    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden and non-essential directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in {"node_modules", "__pycache__", ".git", "venv", "vendor"}
            ]

            level = root.replace(path, "").count(os.sep)
            if level > 3:  # Limit depth
                break

            indent = " " * 2 * level
            items.append(f"{indent}{os.path.basename(root)}/")

            for file in sorted(files)[:20]:  # Limit files per directory
                if count >= max_files:
                    break
                items.append(f"{indent}  {file}")
                count += 1

            if count >= max_files:
                break
    except Exception:
        pass

    return "\n".join(items)


@tool
def clone_github_repo(
    github_url: str, folder_path: Optional[str] = None
) -> str:
    """
    Clone a GitHub repository and optionally navigate to a specific folder.

    Args:
        github_url: Full GitHub URL (e.g., https://github.com/owner/repo or
                   https://github.com/owner/repo/tree/main/src)
        folder_path: Optional specific folder within the repo (e.g., "src/components")

    Returns:
        Path to the cloned repository root or specified folder
    """
    try:
        import git

        # Parse GitHub URL
        # Handle both full URLs and tree-based URLs
        url_match = re.match(r"https://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)/(.+))?", github_url)
        if not url_match:
            return f"Error: Invalid GitHub URL format. Expected https://github.com/owner/repo or https://github.com/owner/repo/tree/branch/path"

        owner, repo, branch, repo_folder = url_match.groups()
        repo_name = repo.rstrip("/")

        # Clone to temp directory
        temp_dir = tempfile.mkdtemp()
        clone_url = f"https://github.com/{owner}/{repo_name}.git"

        git.Repo.clone_from(clone_url, temp_dir, depth=1)

        # Navigate to specified folder if provided
        target_path = temp_dir
        if repo_folder:
            target_path = os.path.join(temp_dir, repo_folder)
        elif folder_path:
            target_path = os.path.join(temp_dir, folder_path)

        if not os.path.exists(target_path):
            return f"Error: Folder '{folder_path or repo_folder}' not found in repository"

        return target_path

    except ImportError:
        return "Error: GitPython not installed. Install with: pip install GitPython"
    except Exception as e:
        return f"Error cloning repository: {str(e)}"


@tool
def analyze_dependencies(repo_path: str) -> str:
    """
    Analyze project dependencies from various package manager files.

    Args:
        repo_path: Path to the repository root

    Returns:
        JSON-formatted string containing detected dependencies and package managers
    """
    try:
        deps = {"python": [], "javascript": [], "java": [], "go": [], "ruby": []}

        # Python dependencies
        req_file = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_file):
            deps["python"] = _parse_requirements_txt(req_file)

        pyproject = os.path.join(repo_path, "pyproject.toml")
        if os.path.exists(pyproject):
            deps["python"].append("(Poetry/setuptools project)")

        # JavaScript/Node dependencies
        pkg_json = os.path.join(repo_path, "package.json")
        if os.path.exists(pkg_json):
            deps["javascript"] = _parse_package_json(pkg_json)

        # Java dependencies
        pom_file = os.path.join(repo_path, "pom.xml")
        if os.path.exists(pom_file):
            deps["java"] = _parse_pom_xml(pom_file)

        # Go dependencies
        go_mod = os.path.join(repo_path, "go.mod")
        if os.path.exists(go_mod):
            deps["go"] = _parse_go_mod(go_mod)

        # Ruby dependencies
        gemfile = os.path.join(repo_path, "Gemfile")
        if os.path.exists(gemfile):
            deps["ruby"].append("(Bundler project)")

        # Remove empty categories
        deps = {k: v for k, v in deps.items() if v}

        return json.dumps(deps, indent=2)

    except Exception as e:
        return f"Error analyzing dependencies: {str(e)}"


@tool
def detect_tech_stack(repo_path: str) -> str:
    """
    Detect the technology stack used in the repository.

    Args:
        repo_path: Path to the repository root

    Returns:
        JSON-formatted string with detected languages, frameworks, databases, and tools
    """
    try:
        tech = _detect_tech_stack(repo_path)
        return json.dumps(tech, indent=2)

    except Exception as e:
        return f"Error detecting tech stack: {str(e)}"


@tool
def list_repo_structure(repo_path: str) -> str:
    """
    Generate a directory tree of the repository structure.

    Args:
        repo_path: Path to the repository root

    Returns:
        String representation of the directory tree (limited depth and files)
    """
    try:
        tree = _list_directory_tree(repo_path)
        if not tree:
            return "Empty or inaccessible repository"
        return tree

    except Exception as e:
        return f"Error listing repository structure: {str(e)}"


@tool
def extract_entry_points(repo_path: str) -> str:
    """
    Identify entry points and main files in the repository.

    Args:
        repo_path: Path to the repository root

    Returns:
        JSON-formatted string with detected entry points by language
    """
    try:
        entry_points = {
            "python": [],
            "javascript": [],
            "java": [],
            "go": [],
        }

        entry_indicators = {
            "python": [
                "main.py",
                "app.py",
                "run.py",
                "start.py",
                "setup.py",
                "__main__.py",
            ],
            "javascript": [
                "index.js",
                "server.js",
                "app.js",
                "main.js",
                "index.ts",
                "server.ts",
                "package.json",
            ],
            "java": ["Main.java", "Application.java", "pom.xml"],
            "go": ["main.go", "go.mod"],
        }

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in {"node_modules", "__pycache__", ".git", "venv"}
            ]

            level = root.replace(repo_path, "").count(os.sep)
            if level > 2:
                continue

            for lang, indicators in entry_indicators.items():
                for indicator in indicators:
                    if indicator in files:
                        rel_path = os.path.relpath(
                            os.path.join(root, indicator), repo_path
                        )
                        if rel_path not in entry_points[lang]:
                            entry_points[lang].append(rel_path)

        # Remove empty categories
        entry_points = {k: v for k, v in entry_points.items() if v}

        return json.dumps(entry_points, indent=2)

    except Exception as e:
        return f"Error extracting entry points: {str(e)}"


@tool
def read_key_files(repo_path: str, file_list: Optional[str] = None) -> str:
    """
    Read and summarize key files from the repository (README, docs, config files).

    Args:
        repo_path: Path to the repository root
        file_list: Comma-separated list of specific files to read (optional)

    Returns:
        Content of key files (truncated if too large)
    """
    try:
        files_to_check = []

        if file_list:
            files_to_check = [f.strip() for f in file_list.split(",")]
        else:
            files_to_check = [
                "README.md",
                "README.rst",
                "README.txt",
                "ARCHITECTURE.md",
                "DESIGN.md",
                "docs/overview.md",
                ".env.example",
                "docker-compose.yml",
            ]

        content = {}
        for file in files_to_check:
            file_path = os.path.join(repo_path, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read(2000)  # Limit to 2000 chars per file
                        content[file] = text
                except Exception:
                    content[file] = "(Unable to read)"

        return json.dumps(content, indent=2)

    except Exception as e:
        return f"Error reading key files: {str(e)}"
