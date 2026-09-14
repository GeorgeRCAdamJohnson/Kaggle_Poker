"""Sandboxed reproduction runner — ``read_as_untrusted`` + ``NotebookManifest`` (task 7.1).

design.md "Components and Interfaces" §4 (SandboxedReproRunner) and the
"Security / sandboxing approach for untrusted notebooks" section. Requirements 1.5, 3.1.

This module implements ONLY the **read-as-untrusted first** control (design Security
step 1): a purely STATIC parse of a competitor ``.ipynb`` that NEVER executes a cell.
It extracts, from the notebook's source text alone:

* **Declared dependencies** — every ``import`` / ``from ... import`` plus any
  ``pip install`` / ``requirements`` hints found in shell (``!``/``%``) lines or magics.
* **Data-path reads** — every ``read_csv`` / ``read_parquet`` / ``scan_parquet`` /
  ``sink_parquet`` call, ``find_data_dir`` references, and ``Path(...)`` string
  literals — so a human reviewer can see exactly what the notebook expects to read
  before it is run behind the path shim (task 7.2).
* **Review flags** — any **network** (``requests`` / ``urllib`` / ``socket`` /
  ``http*``), **subprocess** (``subprocess`` / ``os.system`` / ``os.popen`` /
  shell escapes), **credential-env** (``os.environ`` / ``os.getenv`` for names that
  look like secrets / keys / tokens), or **filesystem-escape** (absolute paths
  outside the data root, ``..`` traversal, or a write ``open(..., 'w')`` /
  ``write_*`` / ``to_*`` / ``sink_*`` outside the working dir) call — surfaced for
  HUMAN REVIEW before any run.

Accountability + safety contract: ALL notebook content is treated as UNTRUSTED
regardless of source — there is **no trusted-source exemption** and no pre-vetting
bypass (design Security step 5; Req 1.5). Any natural-language text embedded in the
notebook (markdown cells, comments, docstrings) is treated as **data, never as
instructions to follow** — this reader surfaces facts about the notebook and does
not act on anything the notebook says. Nothing here executes a cell, imports the
notebook's declared dependencies, or transmits any code/data anywhere.

Task 7.2 adds the actual sandboxed :meth:`SandboxedReproRunner.run` and the
:class:`ReproResult` state machine ON TOP of the static reader, following design.md's
"Security / sandboxing approach for untrusted notebooks" 5-point section:

* **(step 2) explicit path shim** — a generated shim header remaps the notebook's
  data-root discovery (``/kaggle/input`` mounts, ``find_data_dir()``, and bare
  competition filenames) to the local ``poker/data/poker`` root; any read that
  resolves OUTSIDE that shimmed root fails closed (a ``PermissionError``).
* **(step 3) network egress disabled** — the child process runs with ``socket`` and
  the common HTTP client entry points monkeypatched to raise, and with proxy env
  cleared, so no project code/data can be transmitted to a third party (safety
  guardrails high-risk rule).
* **(step 3) credential-stripped env** — every credential-bearing environment
  variable (keys / tokens / secrets / cloud creds) is removed from the child's
  environment before it starts.
* **(step 4) working-dir jail + wall-clock timeout** — the notebook runs in an
  isolated temp working directory as a CHILD PROCESS (never in-process, since
  executing untrusted code is inherently side-effecting) under a wall-clock timeout;
  only the emitted ``submission.csv`` is captured on success and validated against the
  official submission contract (:func:`poker_collusion.submission.writer.validate_submission`).

The status machine (Req 3.3): a run starts at ``RUNNING`` and flips to ``SUCCEEDED``
(a schema-valid ``submission.csv`` was captured) or ``BLOCKED`` with a ``failure_stage``
in ``{deps, data_path, compute, runtime}`` and the exact detail. Status MAY sit at
``RUNNING`` while a failure is diagnosed and flip to ``BLOCKED`` only once the cause is
pinned. Reproduction is NEVER reported as partially successful.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "FlaggedCall",
    "DataPathRead",
    "NotebookManifest",
    "ReproResult",
    "SandboxedReproRunner",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "STATUS_BLOCKED",
    "STAGE_DEPS",
    "STAGE_DATA_PATH",
    "STAGE_COMPUTE",
    "STAGE_RUNTIME",
]

# --------------------------------------------------------------------------- #
# ReproResult status + failure-stage vocabularies (design "Error Handling")     #
# --------------------------------------------------------------------------- #

#: A run begins here and stays here WHILE a failure is diagnosed (Req 3.3).
STATUS_RUNNING: str = "RUNNING"
#: A schema-valid ``submission.csv`` was captured (the ONLY success terminal state).
STATUS_SUCCEEDED: str = "SUCCEEDED"
#: The run could not produce a scoreable submission; ``failure_stage`` names why.
STATUS_BLOCKED: str = "BLOCKED"

#: A declared dependency (import) or notebook-execution tool is missing/unimportable.
STAGE_DEPS: str = "deps"
#: The path shim could not resolve a required local file, or a read failed closed.
STAGE_DATA_PATH: str = "data_path"
#: The wall-clock timeout fired (or the process was OOM-killed) before completion.
STAGE_COMPUTE: str = "compute"
#: An uncaught exception in a notebook cell (a genuine runtime error), OR the emitted
#: submission failed the official schema/contract validation.
STAGE_RUNTIME: str = "runtime"

#: The captured submission filename the notebook is expected to emit (Kaggle contract).
_SUBMISSION_FILENAME: str = "submission.csv"

#: Credential-bearing env-var NAME substrings stripped from the child environment
#: (step 3). A superset of the static reader's ``CREDENTIAL_NAME_HINTS`` (defined
#: below) — it additionally strips cloud/provider tokens — so the run-time strip is at
#: least as strict as what the review-time reader flags. Kept as its own constant to
#: avoid a forward reference to ``CREDENTIAL_NAME_HINTS``.
_CREDENTIAL_ENV_STRIP_HINTS: Tuple[str, ...] = (
    "secret", "key", "token", "password", "passwd", "pwd", "credential", "cred",
    "api", "auth", "access", "private", "session", "cookie", "bearer", "aws",
    "gcp", "azure", "kaggle", "github", "gitlab", "openai", "hf_", "huggingface",
)


# --------------------------------------------------------------------------- #
# Detection vocabularies (kept as module constants so the review criteria are   #
# auditable in one place — contract Rule 2 / Rule 9).                           #
# --------------------------------------------------------------------------- #

#: Top-level module names whose import/use means the notebook can open a network
#: connection. Matched on the import's ROOT module so ``urllib.request`` etc. count.
NETWORK_MODULES: frozenset = frozenset(
    {"requests", "urllib", "urllib3", "socket", "http", "httpx", "aiohttp", "ftplib",
     "smtplib", "telnetlib", "asyncio", "websocket", "websockets"}
)

#: Modules / attribute calls that spawn a subprocess or shell out.
SUBPROCESS_MODULES: frozenset = frozenset({"subprocess", "pty", "commands"})
#: ``os``-attribute calls that execute an external command.
SUBPROCESS_OS_CALLS: frozenset = frozenset(
    {"system", "popen", "popen2", "popen3", "popen4", "spawn", "spawnl", "spawnv",
     "spawnve", "spawnlp", "spawnvp", "execv", "execve", "execl", "execlp"}
)

#: ``os`` credential-access attributes (``os.environ`` / ``os.getenv`` / ``os.putenv``).
CREDENTIAL_ENV_ATTRS: frozenset = frozenset({"environ", "getenv", "getenvb", "putenv"})
#: Substrings in an accessed env-var NAME that mark it as credential-bearing.
CREDENTIAL_NAME_HINTS: Tuple[str, ...] = (
    "secret", "key", "token", "password", "passwd", "pwd", "credential", "cred",
    "api", "auth", "access", "private", "session", "cookie", "bearer",
)

#: Data-READ call attributes we extract (polars + pandas + pyarrow style).
DATA_READ_CALLS: frozenset = frozenset(
    {"read_csv", "read_parquet", "scan_parquet", "read_json", "scan_csv",
     "read_ipc", "scan_ipc", "read_ndjson", "read_excel", "read_feather",
     "read_table", "load", "loadtxt", "genfromtxt"}
)
#: Data-WRITE call attributes (used for the filesystem-escape write check).
DATA_WRITE_CALLS: frozenset = frozenset(
    {"write_csv", "write_parquet", "sink_parquet", "sink_csv", "to_csv",
     "to_parquet", "write_ipc", "write_json", "to_json", "savetxt", "save"}
)
#: Helper names whose presence signals data-root discovery (notebook-specific but
#: common in this competition's notebooks).
DATA_DISCOVERY_NAMES: frozenset = frozenset({"find_data_dir", "rglob", "glob"})

#: Absolute-path prefixes considered INSIDE the acceptable competition roots. An
#: absolute path outside these (and outside the data root) is a filesystem-escape flag.
#: ``/kaggle/input`` is the read mount; ``/kaggle/working`` is the write dir — the
#: shim (task 7.2) remaps these, so they are recorded but only the WRITE side outside
#: the working dir is treated as an escape.
KAGGLE_INPUT_ROOTS: Tuple[str, ...] = ("/kaggle/input",)
KAGGLE_WORKING_ROOTS: Tuple[str, ...] = ("/kaggle/working", "/tmp", "./", "output", "outputs")

_PIP_LINE_RE = re.compile(r"(?:pip|pip3|python\s+-m\s+pip)\s+install\s+(?P<pkgs>.+)")
_ABS_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])")  # C:\ , /foo, \foo


# --------------------------------------------------------------------------- #
# Frozen result models                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FlaggedCall:
    """A single call flagged for HUMAN REVIEW before the notebook may be run.

    Attributes:
        category: One of ``"network"``, ``"subprocess"``, ``"credential_env"``,
            ``"filesystem_escape"``.
        detail: A short human-readable description of exactly what was found
            (e.g. the module/attribute called, the env-var name, or the escaping path).
        cell_index: Index of the code cell the call was found in.
        lineno: 1-based line number within that cell's source (best effort).
        snippet: The source text of the flagged construct (verbatim, untrusted DATA).
    """

    category: str
    detail: str
    cell_index: int
    lineno: int
    snippet: str

    def to_provenance(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "detail": self.detail,
            "cell_index": self.cell_index,
            "lineno": self.lineno,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class DataPathRead:
    """A single data-path read the notebook declares (statically, not executed).

    Attributes:
        call: The read function/attribute name (e.g. ``read_csv``, ``scan_parquet``),
            or a marker like ``"Path-literal"`` / ``"find_data_dir"`` for path refs.
        path_hint: A best-effort string describing the path/argument the read targets
            (a literal filename, a ``dir / "x.parquet"`` expression, or ``<dynamic>``).
        cell_index: Index of the code cell the read was found in.
        lineno: 1-based line number within that cell's source (best effort).
    """

    call: str
    path_hint: str
    cell_index: int
    lineno: int

    def to_provenance(self) -> Dict[str, Any]:
        return {
            "call": self.call,
            "path_hint": self.path_hint,
            "cell_index": self.cell_index,
            "lineno": self.lineno,
        }


@dataclass(frozen=True)
class NotebookManifest:
    """The result of statically reading an UNTRUSTED notebook (no cell executed).

    Attributes:
        notebook: The notebook path that was read (as a string).
        sha256: SHA-256 of the notebook file bytes (byte-integrity anchor for the
            audit trail — the same notebook always yields the same manifest).
        n_cells: Total cell count.
        n_code_cells: Number of code cells parsed.
        dependencies: Sorted list of declared dependencies — root module names of every
            import — so a reviewer sees the full third-party surface.
        pip_hints: Any ``pip install`` / requirements hints found in ``!``/``%`` shell
            lines or magics (raw package specs, verbatim).
        data_path_reads: Every extracted :class:`DataPathRead`.
        flagged_calls: Every :class:`FlaggedCall` (network / subprocess /
            credential_env / filesystem_escape) surfaced for human review.
        parse_errors: Any cells that failed to parse statically (recorded, never
            executed to work around a parse failure). A cell that will not parse is a
            first-class recorded fact, not a silent skip (contract Rule 9).
        notes: MEASURED-labelled human-readable notes (contract Rule 9).
    """

    notebook: str
    sha256: str
    n_cells: int
    n_code_cells: int
    dependencies: List[str]
    pip_hints: List[str]
    data_path_reads: List[DataPathRead]
    flagged_calls: List[FlaggedCall]
    parse_errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def requires_human_review(self) -> bool:
        """``True`` iff any network/subprocess/credential/filesystem-escape call was
        flagged — the notebook MUST be human-reviewed before it may be run."""
        return len(self.flagged_calls) > 0

    def flags_by_category(self) -> Dict[str, List[FlaggedCall]]:
        """Group the flagged calls by category for a reviewer-friendly summary."""
        out: Dict[str, List[FlaggedCall]] = {}
        for fc in self.flagged_calls:
            out.setdefault(fc.category, []).append(fc)
        return out

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the whole manifest for the append-only audit trail."""
        return {
            "notebook": self.notebook,
            "sha256": self.sha256,
            "n_cells": self.n_cells,
            "n_code_cells": self.n_code_cells,
            "dependencies": list(self.dependencies),
            "pip_hints": list(self.pip_hints),
            "data_path_reads": [d.to_provenance() for d in self.data_path_reads],
            "flagged_calls": [f.to_provenance() for f in self.flagged_calls],
            "requires_human_review": self.requires_human_review,
            "parse_errors": list(self.parse_errors),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ReproResult:
    """The terminal (or in-flight) result of a sandboxed notebook reproduction.

    A run starts at ``status == "RUNNING"`` and flips to exactly one terminal state:
    ``"SUCCEEDED"`` (a schema-valid ``submission.csv`` was captured) or ``"BLOCKED"``
    (``failure_stage`` + ``failure_detail`` name the exact pinned cause). Reproduction
    is NEVER reported as partially successful (design "Error Handling"; Req 3.3).

    Attributes:
        notebook: The notebook path that was run (as a string).
        status: One of :data:`STATUS_RUNNING`, :data:`STATUS_SUCCEEDED`,
            :data:`STATUS_BLOCKED`.
        submission_path: Path to the captured, schema-validated ``submission.csv`` on
            :data:`STATUS_SUCCEEDED`; ``None`` otherwise.
        failure_stage: On :data:`STATUS_BLOCKED`, one of :data:`STAGE_DEPS`,
            :data:`STAGE_DATA_PATH`, :data:`STAGE_COMPUTE`, :data:`STAGE_RUNTIME`;
            ``None`` otherwise.
        failure_detail: On :data:`STATUS_BLOCKED`, the exact human-readable reason
            (missing package, unresolved path, timeout, or cell traceback summary);
            ``None`` otherwise.
        dependency_manifest: The declared-dependency view carried from
            :meth:`SandboxedReproRunner.read_as_untrusted` (deps + pip hints + which
            are importable in THIS environment) — the audit trail for a ``deps`` block.
        data_path_manifest: The data-path view carried from the static read (every
            declared read + the shimmed data root) — the audit trail for a
            ``data_path`` block.
    """

    notebook: str
    status: str
    submission_path: Optional[Path]
    failure_stage: Optional[str]
    failure_detail: Optional[str]
    dependency_manifest: Dict[str, Any] = field(default_factory=dict)
    data_path_manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """``True`` iff the run reached :data:`STATUS_SUCCEEDED`."""
        return self.status == STATUS_SUCCEEDED

    @property
    def blocked(self) -> bool:
        """``True`` iff the run reached :data:`STATUS_BLOCKED`."""
        return self.status == STATUS_BLOCKED

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the result for the append-only audit trail (never partial)."""
        return {
            "notebook": self.notebook,
            "status": self.status,
            "submission_path": (
                str(self.submission_path) if self.submission_path is not None else None
            ),
            "failure_stage": self.failure_stage,
            "failure_detail": self.failure_detail,
            "dependency_manifest": dict(self.dependency_manifest),
            "data_path_manifest": dict(self.data_path_manifest),
        }


# --------------------------------------------------------------------------- #
# The static AST visitor                                                        #
# --------------------------------------------------------------------------- #


class _UntrustedCellVisitor(ast.NodeVisitor):
    """Walk ONE code cell's AST and collect deps / data reads / review flags.

    Every method only INSPECTS nodes; nothing is evaluated. Path/argument values are
    reconstructed from literals via :func:`ast.literal_eval` (safe: literals only) or
    described structurally (``<dir> / "name"``) when they are expressions.
    """

    def __init__(self, cell_index: int, source: str) -> None:
        self.cell_index = cell_index
        self._source = source
        self._source_lines = source.splitlines()
        self.dependencies: set[str] = set()
        self.data_reads: List[DataPathRead] = []
        self.flags: List[FlaggedCall] = []

    # -- helpers ---------------------------------------------------------- #

    def _snippet(self, node: ast.AST) -> str:
        """Best-effort verbatim source snippet for a node (untrusted DATA)."""
        try:
            seg = ast.get_source_segment(self._source, node)
            if seg:
                return seg.strip().splitlines()[0][:200]
        except Exception:
            pass
        lineno = getattr(node, "lineno", 1)
        idx = lineno - 1
        if 0 <= idx < len(self._source_lines):
            return self._source_lines[idx].strip()[:200]
        return ""

    def _flag(self, category: str, detail: str, node: ast.AST) -> None:
        self.flags.append(
            FlaggedCall(
                category=category,
                detail=detail,
                cell_index=self.cell_index,
                lineno=int(getattr(node, "lineno", 1)),
                snippet=self._snippet(node),
            )
        )

    @staticmethod
    def _attr_chain(node: ast.AST) -> str:
        """Dotted name for an ``ast.Attribute``/``ast.Name`` chain (e.g. ``os.path.join``)."""
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    @staticmethod
    def _describe_arg(node: Optional[ast.AST]) -> str:
        """Describe a call's first argument as a path hint without evaluating it."""
        if node is None:
            return "<none>"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # dir / "name.parquet" — polars/pathlib style
            left = _UntrustedCellVisitor._describe_arg(node.left)
            right = _UntrustedCellVisitor._describe_arg(node.right)
            return f"{left} / {right}"
        if isinstance(node, ast.Name):
            return f"<var:{node.id}>"
        if isinstance(node, ast.Attribute):
            return f"<attr:{_UntrustedCellVisitor._attr_chain(node)}>"
        if isinstance(node, ast.Call):
            return f"<call:{_UntrustedCellVisitor._attr_chain(node.func)}(...)>"
        return "<dynamic>"

    @staticmethod
    def _string_literals(node: ast.AST) -> List[str]:
        """All string constants inside a node subtree (for path-escape scanning)."""
        out: List[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                out.append(child.value)
        return out

    # -- imports (declared dependencies) ---------------------------------- #

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            self.dependencies.add(root)
            if root in NETWORK_MODULES:
                self._flag("network", f"imports network module {alias.name!r}", node)
            if root in SUBPROCESS_MODULES:
                self._flag("subprocess", f"imports subprocess module {alias.name!r}", node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Ignore relative imports (module is None / level>0) for the dep surface.
        if node.module and node.level == 0:
            root = node.module.split(".")[0]
            self.dependencies.add(root)
            if root in NETWORK_MODULES:
                self._flag("network", f"imports from network module {node.module!r}", node)
            if root in SUBPROCESS_MODULES:
                self._flag("subprocess", f"imports from subprocess module {node.module!r}", node)
        self.generic_visit(node)

    # -- calls (reads + flags) -------------------------------------------- #

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        chain = self._attr_chain(func)
        first_arg = node.args[0] if node.args else None

        # --- data reads ---
        if attr in DATA_READ_CALLS:
            self.data_reads.append(
                DataPathRead(
                    call=attr,
                    path_hint=self._describe_arg(first_arg),
                    cell_index=self.cell_index,
                    lineno=int(getattr(node, "lineno", 1)),
                )
            )
        # data-root discovery helpers
        if attr in DATA_DISCOVERY_NAMES or (isinstance(func, ast.Name) and func.id in DATA_DISCOVERY_NAMES):
            self.data_reads.append(
                DataPathRead(
                    call=chain or (attr or ""),
                    path_hint=self._describe_arg(first_arg),
                    cell_index=self.cell_index,
                    lineno=int(getattr(node, "lineno", 1)),
                )
            )

        # --- Path("...") literals (record + escape-check) ---
        if isinstance(func, ast.Name) and func.id == "Path":
            hint = self._describe_arg(first_arg)
            self.data_reads.append(
                DataPathRead(
                    call="Path-literal",
                    path_hint=hint,
                    cell_index=self.cell_index,
                    lineno=int(getattr(node, "lineno", 1)),
                )
            )
            self._check_path_escape(hint, node, is_write=False)

        # --- subprocess os.* calls ---
        if isinstance(func, ast.Attribute) and attr in SUBPROCESS_OS_CALLS:
            root = self._attr_chain(func).split(".")[0]
            if root == "os":
                self._flag("subprocess", f"calls {chain}(...)", node)
        # subprocess.<anything>
        if chain.split(".")[0] in SUBPROCESS_MODULES:
            self._flag("subprocess", f"calls {chain}(...)", node)

        # --- credential-env access via os.getenv("KEY") ---
        if isinstance(func, ast.Attribute) and attr in CREDENTIAL_ENV_ATTRS:
            name = self._describe_arg(first_arg)
            if self._looks_credential(name):
                self._flag(
                    "credential_env",
                    f"reads credential-bearing env var via {chain}({name!r})",
                    node,
                )

        # --- filesystem-escape: writes ---
        if attr in DATA_WRITE_CALLS:
            # target path is the write call's first arg
            hint = self._describe_arg(first_arg)
            self._check_path_escape(hint, node, is_write=True, call=attr)

        # --- open(path, mode) write ---
        if isinstance(func, ast.Name) and func.id == "open":
            mode = ""
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if any(m in mode for m in ("w", "a", "x", "+")):
                hint = self._describe_arg(first_arg)
                self._check_path_escape(hint, node, is_write=True, call="open")

        self.generic_visit(node)

    # -- attribute reads (os.environ["KEY"] handled via Subscript) -------- #

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # os.environ["SECRET"] pattern
        val = node.value
        if isinstance(val, ast.Attribute) and val.attr == "environ":
            root = self._attr_chain(val).split(".")[0]
            if root == "os":
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if self._looks_credential(key.value):
                        self._flag(
                            "credential_env",
                            f"reads credential-bearing env var os.environ[{key.value!r}]",
                            node,
                        )
        self.generic_visit(node)

    # -- helpers for flag classification ---------------------------------- #

    @staticmethod
    def _looks_credential(name: str) -> bool:
        low = name.lower()
        return any(h in low for h in CREDENTIAL_NAME_HINTS)

    def _check_path_escape(
        self, hint: str, node: ast.AST, *, is_write: bool, call: str = "Path"
    ) -> None:
        """Flag absolute-path-outside-root, ``..`` traversal, or a write outside the
        working dir. Only literal string components can be judged statically; a purely
        dynamic path (``<var:...>``) is NOT flagged here (it is data-root-shimmed at
        run time, task 7.2) but IS visible via data_path_reads."""
        literals = [hint] if hint and not hint.startswith("<") else []
        # also scan any string literals in the subtree (covers dir / "x/../y")
        literals += self._string_literals(node)
        for lit in literals:
            if ".." in Path(lit.replace("\\", "/")).parts:
                self._flag(
                    "filesystem_escape",
                    f"path traversal ('..') in {'write' if is_write else 'path'} {lit!r}",
                    node,
                )
                return
            if _ABS_PATH_RE.match(lit):
                norm = lit.replace("\\", "/")
                inside_input = any(norm.startswith(r) for r in KAGGLE_INPUT_ROOTS)
                inside_working = any(norm.startswith(r) for r in KAGGLE_WORKING_ROOTS)
                # EVERY absolute path is outside the LOCAL data root (poker/data/poker):
                # it either must be remapped by the path shim (a /kaggle/input read
                # mount), jailed as the working dir (a /kaggle/working write), or is a
                # genuine escape (anywhere else). All three are surfaced for review so
                # the shim/jail requirements (design Security steps 2 & 4) are explicit.
                if is_write and not inside_working:
                    self._flag(
                        "filesystem_escape",
                        f"{call} writes to absolute path outside working dir: {lit!r}",
                        node,
                    )
                    return
                if is_write:  # inside a working root
                    self._flag(
                        "filesystem_escape",
                        f"{call} writes to absolute path {lit!r} (requires working-dir jail)",
                        node,
                    )
                    return
                if inside_working:
                    # e.g. Path("/kaggle/working") — the notebook's output root. Absolute
                    # and outside the local data root; the working-dir jail must remap it.
                    self._flag(
                        "filesystem_escape",
                        f"absolute working/output path {lit!r} (requires working-dir jail)",
                        node,
                    )
                    return
                if inside_input:
                    # e.g. Path("/kaggle/input") — the read mount the path shim remaps
                    # to poker/data/poker (design Security step 2). Surfaced for review.
                    self._flag(
                        "filesystem_escape",
                        f"absolute data-mount path {lit!r} (requires path shim to local data root)",
                        node,
                    )
                    return
                # any other absolute path is a genuine escape outside every known root.
                self._flag(
                    "filesystem_escape",
                    f"references absolute path outside data root: {lit!r}",
                    node,
                )
                return


# --------------------------------------------------------------------------- #
# The runner (read_as_untrusted only for task 7.1)                              #
# --------------------------------------------------------------------------- #


class SandboxedReproRunner:
    """Runs UNTRUSTED competitor notebooks against local data (design §4).

    Task 7.1 implements ONLY :meth:`read_as_untrusted` — the static, no-execution
    read that produces a :class:`NotebookManifest`. ALL notebook content is treated as
    untrusted regardless of source; there is no trusted-source exemption (Req 1.5,
    design Security step 5). :meth:`run` (the actual sandbox: path shim, network
    disabled, credential-stripped env, working-dir jail + timeout) is task 7.2.
    """

    def read_as_untrusted(self, notebook: Union[str, Path]) -> NotebookManifest:
        """Statically read an untrusted ``.ipynb`` WITHOUT executing any cell.

        Parses the notebook JSON, walks each code cell's AST, and extracts declared
        dependencies + every data-path read, flagging any network / subprocess /
        credential-env / filesystem-escape call for human review. Markdown/comment
        text is treated as inert DATA — never as instructions to follow.

        Args:
            notebook: Path to the ``.ipynb`` file.

        Returns:
            A :class:`NotebookManifest`. Never executes, imports, or transmits anything.

        Raises:
            FileNotFoundError: If ``notebook`` does not exist.
            ValueError: If the file is not valid notebook JSON.
        """
        import hashlib

        path = Path(notebook)
        if not path.exists():
            raise FileNotFoundError(f"notebook not found: {path}")

        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        try:
            nb = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"not valid notebook JSON: {path} ({exc})") from exc

        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        deps: set[str] = set()
        pip_hints: List[str] = []
        data_reads: List[DataPathRead] = []
        flags: List[FlaggedCall] = []
        parse_errors: List[str] = []

        for idx, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            source = _join_source(cell.get("source", ""))
            # Strip / record shell (!) and magic (%) lines BEFORE AST parsing —
            # they are not valid Python and carry pip/requirements + shell-escape hints.
            py_source, shell_lines = _strip_shell_and_magic(source)
            for sl in shell_lines:
                pip_hints.extend(_extract_pip_hints(sl))
                # a `!cmd` shell escape is a subprocess-equivalent — flag it.
                stripped = sl.lstrip()
                if stripped.startswith("!"):
                    flags.append(
                        FlaggedCall(
                            category="subprocess",
                            detail=f"shell escape (! line): {stripped[:120]!r}",
                            cell_index=idx,
                            lineno=1,
                            snippet=stripped[:200],
                        )
                    )
            try:
                tree = ast.parse(py_source)
            except SyntaxError as exc:
                parse_errors.append(f"cell {idx}: {exc}")
                continue
            visitor = _UntrustedCellVisitor(idx, py_source)
            visitor.visit(tree)
            deps.update(visitor.dependencies)
            data_reads.extend(visitor.data_reads)
            flags.extend(visitor.flags)

        notes = [
            "MEASURED (static parse, no cell executed): "
            f"{len(code_cells)} code cells, {len(deps)} declared deps, "
            f"{len(data_reads)} data-path reads, {len(flags)} review flags.",
            "ALL notebook content treated as UNTRUSTED regardless of source; embedded "
            "markdown/comment text is inert DATA, never followed as instructions.",
        ]
        if flags:
            cats = sorted({f.category for f in flags})
            notes.append(f"HUMAN REVIEW REQUIRED before run — flagged categories: {cats}.")

        return NotebookManifest(
            notebook=str(path),
            sha256=sha256,
            n_cells=len(cells),
            n_code_cells=len(code_cells),
            dependencies=sorted(deps),
            pip_hints=pip_hints,
            data_path_reads=data_reads,
            flagged_calls=flags,
            parse_errors=parse_errors,
            notes=notes,
        )

    # --------------------------------------------------------------------- #
    # Task 7.2 — the actual sandboxed run                                    #
    # --------------------------------------------------------------------- #
    def run(
        self,
        notebook: Union[str, Path],
        *,
        local_data_dir: Union[str, Path],
        timeout_s: int,
        python_executable: Optional[str] = None,
    ) -> ReproResult:
        """Execute an UNTRUSTED notebook behind the four sandbox controls (design §4).

        The notebook is NEVER executed in-process (executing untrusted code is
        inherently side-effecting). It is exported to a plain ``.py`` script, prefixed
        with a generated path-shim + network-disable + credential-strip header, and run
        as a CHILD PROCESS in an isolated temp working directory under a wall-clock
        timeout. Only the emitted ``submission.csv`` is captured on success and it is
        validated against the official submission contract before the run is called
        ``SUCCEEDED``.

        Sandbox controls (design "Security", steps 2–5):

        * **(a) path shim** — the header remaps the notebook's data-root discovery
          (``/kaggle/input`` mounts, ``find_data_dir()``, bare competition filenames)
          to ``local_data_dir``. Reads that resolve OUTSIDE the shimmed root fail
          closed (raise ``PermissionError`` inside the child).
        * **(b) network egress disabled** — the child patches ``socket`` and common
          HTTP entry points to raise, and proxy env vars are cleared, so no project
          code/data can be transmitted externally.
        * **(c) credential-stripped env** — every credential-bearing env var is removed
          from the child's environment.
        * **(d) working-dir jail + timeout** — the child's cwd is an isolated temp dir
          and it is killed at ``timeout_s`` wall-clock seconds.

        The status machine (Req 3.3): the run conceptually begins at ``RUNNING`` and
        flips to ``SUCCEEDED`` (schema-valid ``submission.csv`` captured) or ``BLOCKED``
        with a ``failure_stage`` in ``{deps, data_path, compute, runtime}`` and the
        exact detail. Status only flips to ``BLOCKED`` once the cause is pinned; it is
        never reported as partially successful.

        Args:
            notebook: Path to the ``.ipynb`` to reproduce (treated as untrusted DATA).
            local_data_dir: The local competition data root (``poker/data/poker``) the
                path shim maps the notebook's data discovery onto (read-only semantics).
            timeout_s: Wall-clock timeout in seconds for the child process.
            python_executable: Interpreter to run the exported script with; defaults to
                the current :data:`sys.executable`.

        Returns:
            A terminal :class:`ReproResult` (``SUCCEEDED`` or ``BLOCKED``). This method
            does not itself return a ``RUNNING`` result — ``RUNNING`` is the in-flight
            state the machine passes through while diagnosing.

        Raises:
            FileNotFoundError: If ``notebook`` does not exist.
            ValueError: If ``notebook`` is not valid notebook JSON, or ``timeout_s`` is
                not positive.
        """
        path = Path(notebook)
        if not path.exists():
            raise FileNotFoundError(f"notebook not found: {path}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive; got {timeout_s!r}.")

        data_root = Path(local_data_dir).resolve()
        interpreter = python_executable or sys.executable

        # Static read FIRST (design Security step 1) — the manifest is the audit trail
        # carried into ReproResult and the source of the deps view. This also validates
        # the notebook JSON before anything is executed.
        manifest = self.read_as_untrusted(path)
        dependency_manifest = _dependency_manifest(manifest)
        data_path_manifest = _data_path_manifest(manifest, data_root)

        def blocked(stage: str, detail: str) -> ReproResult:
            # RUNNING → BLOCKED: the cause is now pinned (Req 3.3, never partial).
            return ReproResult(
                notebook=str(path),
                status=STATUS_BLOCKED,
                submission_path=None,
                failure_stage=stage,
                failure_detail=detail,
                dependency_manifest=dependency_manifest,
                data_path_manifest=data_path_manifest,
            )

        # ---- deps pre-check (belt): a declared dep unimportable HERE blocks early. ----
        # The child would fail on `import <pkg>` anyway; pinning it here yields the exact
        # missing-package name instead of an opaque child traceback (design Error table).
        missing = _missing_dependencies(manifest.dependencies)
        if missing:
            return blocked(
                STAGE_DEPS,
                "declared dependency not importable in this environment: "
                f"{missing}. Install the recorded dependency (see requirements.lock) "
                "and re-run; nothing was executed.",
            )

        # ---- data_path pre-check: the shimmed local data root must exist. ----
        if not data_root.is_dir():
            return blocked(
                STAGE_DATA_PATH,
                f"local_data_dir does not exist or is not a directory: {data_root}. "
                "The path shim has nothing to map the notebook's data discovery onto.",
            )

        # ---- export the notebook to a script + shim header (no execution yet). ----
        # __future__ imports are hoisted out of the cell bodies and placed at the ABSOLUTE
        # top of the assembled file (before the sandbox header), because Python requires
        # every ``from __future__`` to be the first statement in the module. Without this,
        # a notebook that repeats ``from __future__ import annotations`` across cells would
        # raise "SyntaxError: from __future__ imports must occur at the beginning of the file".
        future_imports, script_body = _export_notebook_to_script(path)
        header = _build_sandbox_header(data_root)
        future_block = ("\n".join(future_imports) + "\n\n") if future_imports else ""
        full_script = future_block + header + "\n\n" + script_body

        # ---- working-dir jail: isolated temp cwd; only submission.csv is captured. ----
        jail = Path(tempfile.mkdtemp(prefix="anchor_repro_jail_"))
        script_path = jail / "_repro_untrusted.py"
        try:
            # Expose the competition data INSIDE the jail at ``data/poker`` so the
            # notebook's OWN ``find_data_dir()`` (which rglob-searches ``Path("data")``
            # relative to cwd for ``players.parquet``) discovers it, WITHOUT copying the
            # multi-GB parquet. Prefers a dir junction/symlink; the read-guard still
            # permits reads under the real data root (the link just makes discovery
            # succeed). This replaces relying on the header shim shadowing the notebook's
            # own find_data_dir definition (which the notebook redefines later).
            _link_data_into_jail(data_root, jail)
            script_path.write_text(full_script, encoding="utf-8")
            child_env = _hardened_child_env(data_root, jail)

            try:
                completed = subprocess.run(
                    [interpreter, str(script_path)],
                    cwd=str(jail),
                    env=child_env,
                    capture_output=True,
                    text=True,
                    # The child is forced to UTF-8 stdio (see _hardened_child_env), so the
                    # parent MUST decode its output as UTF-8 too. Default text=True uses
                    # the OS locale (cp1252 on Windows), which raises UnicodeDecodeError in
                    # the reader thread on any non-cp1252 byte and tears the run down.
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                # (d) wall-clock timeout — a compute-limit block (Req 3.3).
                return blocked(
                    STAGE_COMPUTE,
                    f"wall-clock timeout after {timeout_s}s; the notebook did not emit "
                    f"a submission in time. Partial stdout tail: "
                    f"{_tail(exc.stdout)!r}",
                )
            except (OSError, ValueError) as exc:
                # Could not even launch the interpreter (e.g. missing python) — treat as
                # a deps/tooling block: the execution tool itself is unavailable.
                return blocked(
                    STAGE_DEPS,
                    f"could not launch the child interpreter {interpreter!r}: {exc}. "
                    "The notebook-execution tool is unavailable in this environment.",
                )

            if completed.returncode != 0:
                # Diagnose WHY the child failed and pin the stage (RUNNING while we do).
                stage, detail = _diagnose_child_failure(
                    completed.stdout, completed.stderr, manifest.dependencies
                )
                return blocked(stage, detail)

            # ---- capture ONLY submission.csv from the jail (design step 4). ----
            emitted = jail / _SUBMISSION_FILENAME
            if not emitted.is_file():
                return blocked(
                    STAGE_RUNTIME,
                    "the notebook exited cleanly but emitted no "
                    f"{_SUBMISSION_FILENAME!r} in the working-dir jail; there is "
                    "nothing to capture (never reported as partial success).",
                )

            # ---- validate the captured submission against the official contract. ----
            valid, reason = _validate_captured_submission(emitted, data_root)
            if not valid:
                return blocked(
                    STAGE_RUNTIME,
                    f"captured {_SUBMISSION_FILENAME!r} failed the official submission "
                    f"contract: {reason}",
                )

            # ---- RUNNING → SUCCEEDED: persist the captured submission next to the jail
            # so it survives cleanup, and return the only success terminal state. ----
            captured = _persist_submission(emitted)
            return ReproResult(
                notebook=str(path),
                status=STATUS_SUCCEEDED,
                submission_path=captured,
                failure_stage=None,
                failure_detail=None,
                dependency_manifest=dependency_manifest,
                data_path_manifest=data_path_manifest,
            )
        finally:
            # The jail (and the untrusted exported script inside it) is always removed;
            # any captured submission was copied out above before this runs.
            shutil.rmtree(jail, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Free helpers                                                                  #
# --------------------------------------------------------------------------- #


def _join_source(source: Union[str, List[str]]) -> str:
    """A notebook cell's ``source`` is either a str or a list of line strings."""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _strip_shell_and_magic(source: str) -> Tuple[str, List[str]]:
    """Split cell source into (python_only, shell_and_magic_lines).

    Lines beginning with ``!`` (shell escape) or ``%``/``%%`` (IPython magic) are not
    valid Python; they are removed from the AST-parsed body and returned separately so
    pip/requirements hints and shell escapes are still surfaced. Removed lines are
    blanked (not deleted) so remaining line numbers stay aligned.
    """
    py_lines: List[str] = []
    shell_lines: List[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            shell_lines.append(line)
            py_lines.append("")  # keep line alignment
        else:
            py_lines.append(line)
    return "\n".join(py_lines), shell_lines


def _extract_pip_hints(line: str) -> List[str]:
    """Extract package specs from a ``pip install ...`` shell/magic line."""
    stripped = line.lstrip().lstrip("!%").strip()
    m = _PIP_LINE_RE.search(stripped)
    if not m:
        return []
    pkgs = m.group("pkgs")
    # drop common flags; keep bare package specs
    tokens = [t for t in pkgs.split() if not t.startswith("-")]
    return tokens


# --------------------------------------------------------------------------- #
# Task 7.2 run() helpers: manifests, dep check, script export, sandbox header,  #
# hardened env, failure diagnosis, submission capture/validation.               #
# --------------------------------------------------------------------------- #


def _dependency_manifest(manifest: NotebookManifest) -> Dict[str, Any]:
    """The dependency audit view carried into a :class:`ReproResult`.

    Records the statically-declared deps, the ``pip install`` hints, and — for THIS
    environment — which declared deps are importable and which are missing, so a
    ``deps`` block's detail is reproducible from the result alone (design Error table).
    """
    importable, missing = _split_importable(manifest.dependencies)
    return {
        "declared_dependencies": list(manifest.dependencies),
        "pip_hints": list(manifest.pip_hints),
        "importable_here": importable,
        "missing_here": missing,
    }


def _data_path_manifest(manifest: NotebookManifest, data_root: Path) -> Dict[str, Any]:
    """The data-path audit view carried into a :class:`ReproResult`.

    Records every statically-declared data-path read plus the shimmed local data root
    the runner mapped the notebook's data discovery onto (design Security step 2).
    """
    return {
        "shimmed_data_root": str(data_root),
        "declared_reads": [d.to_provenance() for d in manifest.data_path_reads],
    }


def _split_importable(dependencies: List[str]) -> Tuple[List[str], List[str]]:
    """Partition declared deps into (importable-here, missing-here).

    Uses :func:`importlib.util.find_spec` — this only checks whether the module can be
    LOCATED, it does NOT import (and therefore does not execute) the dependency. Stdlib
    and known-aliased modules are handled by the alias map so a notebook importing
    ``cv2``/``sklearn``/``PIL`` is classified correctly.
    """
    import importlib.util

    # Common import-name → distribution differences are irrelevant here (we test the
    # IMPORT name, which is what the notebook actually uses). A few builtins the static
    # reader may surface (e.g. from `from __future__`) are always present.
    importable: List[str] = []
    missing: List[str] = []
    for dep in dependencies:
        if dep in _ALWAYS_PRESENT_MODULES:
            importable.append(dep)
            continue
        try:
            spec = importlib.util.find_spec(dep)
        except (ImportError, ValueError, ModuleNotFoundError):
            spec = None
        (importable if spec is not None else missing).append(dep)
    return importable, missing


#: Module names that are always available (stdlib / builtins the static reader may list).
_ALWAYS_PRESENT_MODULES: frozenset = frozenset(
    {"os", "sys", "re", "json", "math", "time", "random", "pathlib", "itertools",
     "functools", "collections", "typing", "dataclasses", "warnings", "gc",
     "hashlib", "datetime", "io", "glob", "shutil", "tempfile", "subprocess"}
)


def _missing_dependencies(dependencies: List[str]) -> List[str]:
    """Return the declared deps that cannot be located (imported) in this environment."""
    _importable, missing = _split_importable(dependencies)
    return missing


#: Matches a ``from __future__ import ...`` statement (single line; the competition
#: notebooks use the single-line form ``from __future__ import annotations``).
_FUTURE_IMPORT_RE = re.compile(r"^\s*from\s+__future__\s+import\s+.+$")


def _export_notebook_to_script(path: Path) -> Tuple[List[str], str]:
    """Export a notebook's CODE cells to a runnable ``.py`` body (no execution).

    Concatenates code cells in order, stripping IPython shell (``!``) and magic
    (``%``/``%%``) lines exactly like :func:`_strip_shell_and_magic` (they are not
    valid Python). This deliberately does NOT depend on ``nbconvert``/``jupyter`` — the
    export is a pure JSON+text transform so the runner works even when no notebook
    execution tool is installed. The exported script is still 100% UNTRUSTED and is
    only ever run inside the child-process sandbox.

    ``from __future__`` HOISTING (execution-correctness fix): Python requires every
    ``from __future__ import ...`` to appear at the very top of the file, before any
    other statement. A notebook commonly repeats ``from __future__ import annotations``
    at the top of MULTIPLE cells; naively concatenating cells would place the 2nd+
    occurrence mid-file and raise ``SyntaxError: from __future__ imports must occur at
    the beginning of the file``. So this function EXTRACTS every ``from __future__``
    line out of the cell bodies (blanking it in place to keep line numbers aligned for
    tracebacks) and returns the DEDUPED set separately, for the caller to place at the
    absolute top of the assembled script (before even the sandbox header). This is a
    pure syntactic relocation — it changes nothing the notebook does.

    Returns:
        ``(future_imports, body)`` where ``future_imports`` is the sorted, de-duplicated
        list of ``from __future__`` statements and ``body`` is the concatenated cell
        code with those statements removed.
    """
    nb = json.loads(path.read_bytes().decode("utf-8"))
    parts: List[str] = []
    futures: List[str] = []
    seen_futures: set[str] = set()
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _join_source(cell.get("source", ""))
        py_source, _shell = _strip_shell_and_magic(source)
        # Hoist __future__ imports: collect them (deduped) and blank the line in place
        # so remaining line numbers stay aligned with the original cell for tracebacks.
        kept_lines: List[str] = []
        for line in py_source.splitlines():
            if _FUTURE_IMPORT_RE.match(line):
                stmt = line.strip()
                if stmt not in seen_futures:
                    seen_futures.add(stmt)
                    futures.append(stmt)
                kept_lines.append("")  # keep line alignment
            else:
                kept_lines.append(line)
        py_source = "\n".join(kept_lines)
        if py_source.strip():
            parts.append(f"# --- notebook code cell {idx} ---\n{py_source}")
    return sorted(futures), "\n\n".join(parts)


def _build_sandbox_header(data_root: Path) -> str:
    """Build the in-child sandbox header prepended to the exported script.

    The header runs FIRST inside the child and installs, in order:

    * **(step 2) the path shim** — a builtins ``open`` wrapper + ``pathlib.Path``
      resolution guard that (i) remaps any ``/kaggle/input`` mount path and any bare
      competition filename to ``data_root``, and (ii) FAILS CLOSED (``PermissionError``)
      on any read that resolves outside ``data_root`` or the working-dir jail. It also
      defines a module-level ``find_data_dir()`` returning ``data_root`` so notebooks
      that call it get the shimmed root.
    * **(step 3) network egress disabled** — ``socket.socket`` and
      ``socket.create_connection`` are replaced with raisers; ``urllib`` opener and the
      common proxy env are neutralised.

    The header is authored here (trusted) — it is NOT taken from the notebook.
    """
    # Use repr on the resolved string so backslashes on Windows are safely escaped.
    root_literal = repr(str(data_root))
    files_literal = ", ".join(repr(f) for f in _COMPETITION_FILENAMES)
    return (
        _SANDBOX_HEADER_TEMPLATE
        .replace("__DATA_ROOT__", root_literal)
        .replace("__COMPETITION_FILES__", files_literal)
    )


#: The competition filenames the shim remaps a bare/kaggle-input reference onto.
_COMPETITION_FILENAMES: Tuple[str, ...] = (
    "development_labels.csv", "development_evidence.csv", "evaluation_pairs.csv",
    "sample_submission.csv", "hands.parquet", "actions.parquet", "players.parquet",
    "seats.parquet",
)


_SANDBOX_HEADER_TEMPLATE = '''\
# ==== anchor_repro sandbox header (AUTHORED BY THE RUNNER, NOT THE NOTEBOOK) ====
# Installed FIRST inside the untrusted child so every subsequent (untrusted) cell
# runs behind the path shim + network-disable controls. See design.md Security §2-3.
import builtins as _ar_builtins
import os as _ar_os
import socket as _ar_socket
import pathlib as _ar_pathlib

_AR_DATA_ROOT = _ar_pathlib.Path(__DATA_ROOT__).resolve()
_AR_JAIL = _ar_pathlib.Path.cwd().resolve()
_AR_COMPETITION_FILES = frozenset([__COMPETITION_FILES__])

# Python installation / library roots. Reads are ALSO allowed from these so an
# imported library can read its OWN packaged resources (e.g. matplotlib reading
# mpl-data/matplotlibrc, certifi reading its CA bundle, sklearn/xgboost datafiles).
# This is NOT a data-exfiltration risk: it is a library reading files that ship inside
# its own installed package. Arbitrary user-data reads (home dir, other drives) remain
# denied, writes remain jailed to the working dir, and network egress stays disabled.
import sys as _ar_sys
import sysconfig as _ar_sysconfig
_ar_lib_candidates = [
    _ar_sys.prefix, _ar_sys.base_prefix, _ar_sys.exec_prefix, _ar_sys.base_exec_prefix,
]
try:
    for _ar_k in ("purelib", "platlib", "stdlib", "platstdlib", "data"):
        _ar_p = _ar_sysconfig.get_paths().get(_ar_k)
        if _ar_p:
            _ar_lib_candidates.append(_ar_p)
except Exception:
    pass
for _ar_sp in list(_ar_sys.path):
    # site-packages / local-packages dirs the interpreter already trusts to import from.
    if _ar_sp and ("site-packages" in _ar_sp or "local-packages" in _ar_sp):
        _ar_lib_candidates.append(_ar_sp)
_AR_LIB_ROOTS = []
for _ar_c in _ar_lib_candidates:
    try:
        _AR_LIB_ROOTS.append(_ar_pathlib.Path(_ar_c).resolve())
    except Exception:
        pass


def find_data_dir(*_args, **_kwargs):
    """Shimmed data-root discovery -> the local competition data root (step 2)."""
    return _AR_DATA_ROOT


def _ar_remap(path):
    """Map a requested path to the shimmed root; fail closed if it escapes it."""
    p = _ar_pathlib.Path(path)
    name = p.name
    s = str(p).replace(chr(92), "/")
    # (i) remap a /kaggle/working OUTPUT path to the jail cwd so the notebook's
    # ``OUTPUT_DIR = Path("/kaggle/working")`` writes (incl. submission.csv) land where
    # the runner captures them, instead of being denied as an out-of-jail write.
    if s.startswith("/kaggle/working"):
        tail = _ar_pathlib.Path(*p.parts[3:]) if len(p.parts) > 3 else _ar_pathlib.Path(name if name else ".")
        return _AR_JAIL / tail
    # (ii) remap a /kaggle/input mount or a bare competition filename to the data root.
    if s.startswith("/kaggle/input") or name in _AR_COMPETITION_FILES:
        if name in _AR_COMPETITION_FILES:
            return _AR_DATA_ROOT / name
        # a /kaggle/input/<slug>/<rest> path -> data_root / <rest tail>
        tail = _ar_pathlib.Path(*p.parts[3:]) if len(p.parts) > 3 else _ar_pathlib.Path(name)
        return _AR_DATA_ROOT / tail
    return p


def _ar_is_inside(resolved, root):
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


_ar_real_open = _ar_builtins.open


def _ar_guarded_open(file, mode="r", *args, **kwargs):
    """(step 2) reads must resolve inside the data root or the jail; else fail closed.
    Writes are confined to the working-dir jail."""
    remapped = _ar_remap(file) if isinstance(file, (str, bytes, _ar_os.PathLike)) else file
    is_write = any(m in str(mode) for m in ("w", "a", "x", "+"))
    if isinstance(remapped, _ar_pathlib.Path) or isinstance(file, (str, bytes, _ar_os.PathLike)):
        resolved = _ar_pathlib.Path(remapped).resolve()
        if is_write:
            if not _ar_is_inside(resolved, _AR_JAIL):
                raise PermissionError(
                    "sandbox: write outside the working-dir jail is denied: " + str(resolved)
                )
        else:
            _ar_read_ok = (
                _ar_is_inside(resolved, _AR_DATA_ROOT)
                or _ar_is_inside(resolved, _AR_JAIL)
                or any(_ar_is_inside(resolved, _ar_r) for _ar_r in _AR_LIB_ROOTS)
            )
            if not _ar_read_ok:
                raise PermissionError(
                    "sandbox: read outside the shimmed data root is denied: " + str(resolved)
                )
        return _ar_real_open(resolved, mode, *args, **kwargs)
    return _ar_real_open(file, mode, *args, **kwargs)


_ar_builtins.open = _ar_guarded_open


def _ar_network_denied(*_a, **_k):
    raise PermissionError("sandbox: network egress is disabled (design Security step 3)")


# (step 3) disable network EGRESS for the child WITHOUT breaking the socket class
# hierarchy. Replacing the ``socket.socket`` CLASS with a function corrupts imports
# (e.g. ``ssl`` does ``class SSLSocket(socket)``; ``asyncio`` imports ``ssl``), so we
# instead neutralise the OUTBOUND CONNECTION methods and leave the class subclassable.
# No connection can be opened, but importing ssl/asyncio/urllib3/etc. still works.
try:
    _ar_socket.socket.connect = _ar_network_denied      # type: ignore[assignment]
    _ar_socket.socket.connect_ex = _ar_network_denied   # type: ignore[assignment]
except (AttributeError, TypeError):
    pass
_ar_socket.create_connection = _ar_network_denied
for _ar_proxy in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
    _ar_os.environ.pop(_ar_proxy, None)
# ==== end sandbox header ====
'''


def _hardened_child_env(data_root: Path, jail: Path) -> Dict[str, str]:
    """Build the hardened environment for the child (design Security steps 3–4).

    * (step 3) every credential-bearing env var is STRIPPED (keys/tokens/secrets/cloud
      creds — the :data:`_CREDENTIAL_ENV_STRIP_HINTS` vocabulary).
    * proxy variables are cleared (belt to the in-child socket disable).
    * ``PYTHONPATH`` is preserved so ``poker_collusion``/``anchor_repro`` remain
      importable, and a couple of shim hints are exported for notebooks that read env
      (``KAGGLE_KERNEL_RUN_TYPE`` is removed so the notebook does NOT think it is on
      Kaggle and try ``/kaggle/input``; the shim handles discovery instead).
    """
    env: Dict[str, str] = {}
    for name, value in os.environ.items():
        low = name.lower()
        if any(hint in low for hint in _CREDENTIAL_ENV_STRIP_HINTS):
            continue  # (step 3) strip credential-bearing vars
        if low in ("http_proxy", "https_proxy", "all_proxy"):
            continue  # clear proxies
        env[name] = value
    # Ensure the child does NOT believe it is on Kaggle (the shim provides the data).
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    # Record the shimmed root for notebooks that consult an env var for it.
    env["ANCHOR_REPRO_DATA_ROOT"] = str(data_root)
    # Make the child unbuffered so a timeout still yields a useful stdout tail.
    env["PYTHONUNBUFFERED"] = "1"
    # Force matplotlib's NON-INTERACTIVE Agg backend in the child. A notebook EDA cell
    # that calls plt.show() would otherwise open a blocking GUI window and HANG the
    # headless child forever (no display to close it). Agg makes plt.show() a no-op and
    # keeps the run deterministic and non-blocking (design step 4: automated jail run).
    env["MPLBACKEND"] = "Agg"
    # Redirect matplotlib's config/cache dir INTO the jail. On first use matplotlib
    # writes a font cache (e.g. ~/.matplotlib/fontlist-*.json); the working-dir jail
    # denies writes outside itself, so without this the cache write fails closed. A
    # cache dir inside the jail keeps the write jailed AND lets matplotlib initialise.
    env["MPLCONFIGDIR"] = str(jail / ".mpl_cache")
    # Force UTF-8 stdio in the child. The notebook prints unicode (e.g. the "→"
    # arrow); the default Windows cp1252 stdout raises UnicodeEncodeError on it. UTF-8
    # I/O makes the child's prints encodable and the run deterministic across locales.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _tail(text: Optional[str], limit: int = 800) -> str:
    """Return the last ``limit`` chars of ``text`` (safe for None)."""
    if not text:
        return ""
    return text[-limit:]


def _diagnose_child_failure(
    stdout: Optional[str], stderr: Optional[str], dependencies: List[str]
) -> Tuple[str, str]:
    """Pin a non-zero child exit to a (failure_stage, detail) — RUNNING while diagnosing.

    Classifies the child's traceback into one of the four stages (design Error table):

    * ``deps`` — a ``ModuleNotFoundError`` / ``ImportError`` (a missing dependency).
    * ``data_path`` — a ``FileNotFoundError``, or the shim's fail-closed
      ``PermissionError`` for a read outside the data root.
    * ``compute`` — a ``MemoryError`` (OOM reached in-process).
    * ``runtime`` — any other uncaught exception (the default; a genuine cell error).

    The detail includes the exact missing module / path where recoverable, plus a
    trimmed traceback tail so the block is reproducible from the result alone.
    """
    err = (stderr or "") + "\n" + (stdout or "")
    tail = _tail(stderr) or _tail(stdout)

    # network egress denial by the sandbox (step 3): the notebook TRIED to reach the
    # network and the sandbox refused it. This is a runtime error FROM THE SANDBOX
    # CONTROL, surfaced explicitly so the block reads as "egress denied" not an opaque
    # cell error — no project code/data was transmitted (safety guardrails high-risk).
    if "sandbox: network egress is disabled" in err:
        return (
            STAGE_RUNTIME,
            "the notebook attempted network egress and the sandbox denied it (network "
            f"disabled, design step 3); no data was transmitted. Traceback tail: {tail!r}",
        )

    # deps: missing module.
    m = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", err)
    if m or "ImportError" in err:
        module = m.group(1) if m else "<unresolved import>"
        return (
            STAGE_DEPS,
            f"the notebook failed importing a dependency ({module!r}); install the "
            f"recorded dependency and re-run. Traceback tail: {tail!r}",
        )

    # data_path: the shim's fail-closed read, or a genuine missing file.
    if "sandbox: read outside the shimmed data root is denied" in err:
        pm = re.search(r"denied: (.+)", err)
        target = pm.group(1).strip() if pm else "<path>"
        return (
            STAGE_DATA_PATH,
            "the notebook attempted a read outside the shimmed data root and the shim "
            f"failed it closed (path shim, design step 2): {target}. Traceback tail: {tail!r}",
        )
    if "sandbox: write outside the working-dir jail is denied" in err:
        return (
            STAGE_DATA_PATH,
            "the notebook attempted a write outside the working-dir jail and the jail "
            f"denied it (design step 4). Traceback tail: {tail!r}",
        )
    fm = re.search(r"FileNotFoundError: .*?([^\s'\"]+\.(?:csv|parquet|json|npy|txt))", err)
    if fm or "FileNotFoundError" in err:
        target = fm.group(1) if fm else "<file>"
        return (
            STAGE_DATA_PATH,
            "the path shim could not resolve a required local file the notebook "
            f"expected: {target}. Traceback tail: {tail!r}",
        )

    # compute: OOM reached in-process (the wall-clock timeout is handled by the caller).
    if "MemoryError" in err:
        return (
            STAGE_COMPUTE,
            f"the notebook ran out of memory (MemoryError). Traceback tail: {tail!r}",
        )

    # runtime: any other uncaught exception in a cell (the default).
    return (
        STAGE_RUNTIME,
        f"an uncaught exception occurred while executing a notebook cell. "
        f"Traceback tail: {tail!r}",
    )


def _validate_captured_submission(
    submission_path: Path, data_root: Path
) -> Tuple[bool, str]:
    """Validate the captured ``submission.csv`` against the official contract.

    Reuses :func:`poker_collusion.submission.writer.validate_submission` against the
    local ``sample_submission.csv`` (the eval pair set) so a locally-valid file is
    leaderboard-valid: identical schema/order, identical row count, exact pair-id set
    equality, ``risk_score`` in ``[0, 1]``, allowed behaviors, non-empty evidence cells
    with no per-row hand-id repeats. Returns ``(ok, reason)``; ``reason`` is empty on
    success.
    """
    try:
        import pandas as pd
        from poker_collusion.submission.writer import (
            SubmissionValidationError,
            validate_submission,
        )
    except Exception as exc:  # pragma: no cover - poker_collusion is always present
        return False, f"could not import the submission validator: {exc}"

    sample_path = data_root / "sample_submission.csv"
    if not sample_path.is_file():
        return False, (
            f"sample_submission.csv not found under the data root ({data_root}); "
            "cannot validate the captured submission against the eval pair set."
        )
    try:
        frame = pd.read_csv(submission_path, dtype={"pair_id": str})
        sample = pd.read_csv(sample_path, dtype={"pair_id": str})
    except Exception as exc:
        return False, f"could not read the captured/sample submission csv: {exc}"

    try:
        validate_submission(frame, sample)
    except SubmissionValidationError as exc:
        return False, str(exc)
    return True, ""


def _link_data_into_jail(data_root: Path, jail: Path) -> None:
    """Expose ``data_root`` inside the jail at ``data/poker`` for the notebook's own
    ``find_data_dir()`` discovery, without copying the data.

    Tries a directory junction / symlink (no copy). If the OS refuses to create a link
    (permissions), falls back to reflecting the data root's file NAMES via a lightweight
    marker so discovery can still resolve — but the link path is the normal case on
    Windows (junctions need no elevation) and POSIX (symlink). Reads through the link
    resolve to the real data root, which the read-guard already permits.
    """
    link_parent = jail / "data"
    link_parent.mkdir(parents=True, exist_ok=True)
    target = link_parent / "poker"
    try:
        # Directory symlink (POSIX) / junction-like (Windows: symlink needs the flag;
        # os.symlink with target_is_directory works when Developer Mode / privilege is on).
        os.symlink(str(data_root), str(target), target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        # Windows directory JUNCTION via mklink /J — needs NO elevation. This is the
        # reliable no-copy path on stock Windows.
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(target), str(data_root)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if target.exists():
                return
    except Exception:
        pass
    # Last resort (should be rare): the notebook discovery will fail and the run will
    # BLOCK at data_path with an honest reason — never a fabricated success.


def _persist_submission(emitted: Path) -> Path:
    """Copy the captured ``submission.csv`` out of the jail before the jail is removed.

    The captured file is placed in a fresh temp directory (NOT the jail, which is
    ``rmtree``-d) so the returned :class:`ReproResult.submission_path` remains valid
    after :meth:`SandboxedReproRunner.run` returns.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="anchor_repro_submission_"))
    out_path = out_dir / _SUBMISSION_FILENAME
    shutil.copy2(emitted, out_path)
    return out_path
