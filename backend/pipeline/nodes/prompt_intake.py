"""
Agent 1 — Prompt Intake (§5, §6.2).

Responsibilities:
- Parse raw prompt for @file references
- Detect project context: git root, language, framework, CLAUDE.md summary
- Assign request_id, workspace_id
- Normalise attached_files into FileRef list
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import uuid
from pathlib import Path

import structlog

from backend.core.models import FileRef, ProjectContext
from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)

_FILE_REF_RE = re.compile(r"@([\w./\-]+)")

_LANG_EXTENSIONS: dict[str, str] = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".cs": "C#",
    ".cpp": "C++", ".c": "C", ".swift": "Swift", ".kt": "Kotlin",
}

_FRAMEWORK_FILES: list[tuple[str, str]] = [
    ("pyproject.toml", "FastAPI/Python"),
    ("package.json", "Node.js"),
    ("next.config.mjs", "Next.js"),
    ("next.config.js", "Next.js"),
    ("go.mod", "Go"),
    ("Cargo.toml", "Rust"),
    ("pom.xml", "Java/Maven"),
    ("build.gradle", "Java/Gradle"),
    ("mix.exs", "Elixir"),
    ("composer.json", "PHP"),
]


def _git_root(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=cwd, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else cwd
    except Exception:
        return cwd


def _detect_language(root: str) -> str:
    counts: dict[str, int] = {}
    total_files = 0
    try:
        for dirpath, dirs, files in os.walk(root):
            # Prune directories in-place to avoid traversing deep ignored trees
            dirs[:] = [d for d in dirs if not (d.startswith(".") or d in ("node_modules", ".venv", "venv", "dist", "build", "target"))]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in _LANG_EXTENSIONS:
                    lang = _LANG_EXTENSIONS[ext]
                    counts[lang] = counts.get(lang, 0) + 1
                    total_files += 1
            if total_files > 200:
                break
    except Exception as e:
        log.warning("prompt_intake.language_detect_error", error=str(e))
    return max(counts, key=counts.get) if counts else ""


def _detect_framework(root: str) -> str:
    for filename, label in _FRAMEWORK_FILES:
        try:
            path = Path(root, filename)
            if path.exists() and path.is_file():
                # Refine specific frameworks
                if filename == "package.json":
                    content = path.read_text(errors="ignore").lower()
                    if "next" in content: return "Next.js"
                    if "express" in content: return "Express"
                    if "react" in content: return "React"
                    if "vue" in content: return "Vue"
                elif filename == "pyproject.toml":
                    content = path.read_text(errors="ignore").lower()
                    if "fastapi" in content or "uvicorn" in content: return "FastAPI"
                    if "django" in content: return "Django"
                    if "flask" in content: return "Flask"
                return label
        except Exception as e:
            log.warning("prompt_intake.framework_detect_error", file=filename, error=str(e))
    return ""


def _claude_md_summary(root: str, max_chars: int = 800) -> str:
    for candidate in [Path(root, "CLAUDE.md"), Path(root, ".claude", "CLAUDE.md")]:
        if candidate.exists() and candidate.is_file():
            try:
                text = candidate.read_text(errors="ignore")
                return text[:max_chars].strip()
            except Exception as e:
                log.warning("prompt_intake.claude_md_read_error", error=str(e))
    return ""


def _read_file_ref(path: str, root: str) -> FileRef | None:
    try:
        full = Path(root).joinpath(path).resolve()
        root_path = Path(root).resolve()
        # Security: prevent directory traversal outside root
        if not str(full).startswith(str(root_path)):
            log.warning("prompt_intake.path_traversal_attempt", path=path)
            return None
            
        if full.exists() and full.is_file():
            if full.stat().st_size < 500_000:
                content = full.read_bytes()
                h = hashlib.sha256(content).hexdigest()
                return FileRef(path=str(full), content_hash=h)
            else:
                log.warning("prompt_intake.file_too_large", path=path)
    except Exception as e:
        log.warning("prompt_intake.file_read_error", path=path, error=str(e))
    return None


async def prompt_intake(state: OptiviaState) -> OptiviaState:
    """Agent 1 — Prompt Intake."""
    raw = state.get("raw_prompt", "").strip()
    if not raw:
        state["error"] = "empty prompt"
        return state

    # Assign IDs
    state.setdefault("request_id", str(uuid.uuid4()))
    state.setdefault("user_id", "anonymous")
    state.setdefault("workspace_id", str(uuid.uuid4()))
    state.setdefault("turn_index", 0)
    state.setdefault("clarification_round", 0)
    state.setdefault("consecutive_high_quality", 0)
    state.setdefault("execution_trace", [])
    state.setdefault("clarifications", [])
    state.setdefault("adaptation_actions", [])
    state.setdefault("obs_tokens", 0)
    state.setdefault("memory_tokens", 0)
    state.setdefault("plan_tokens", 0)
    state.setdefault("action_tokens", 0)

    # Detect working directory (passed via project_context or cwd)
    ctx = state.get("project_context") or ProjectContext()
    cwd = ctx.repo_root or os.getcwd()
    root = _git_root(cwd)

    # Parse @file references out of the prompt
    refs = _FILE_REF_RE.findall(raw)
    file_refs: list[FileRef] = list(state.get("attached_files", []))
    for ref_path in refs:
        fr = _read_file_ref(ref_path, root)
        if fr and fr not in file_refs:
            file_refs.append(fr)

    # Build / enrich project context
    language = ctx.language or _detect_language(root)
    framework = ctx.framework or _detect_framework(root)
    claude_md = ctx.claude_md_summary or _claude_md_summary(root)

    state["project_context"] = ProjectContext(
        workspace_id=state["workspace_id"],
        repo_root=root,
        language=language,
        framework=framework,
        claude_md_summary=claude_md,
    )
    state["attached_files"] = file_refs

    log.info(
        "prompt_intake.done",
        request_id=state["request_id"],
        language=language,
        framework=framework,
        file_refs=len(file_refs),
    )
    return state
