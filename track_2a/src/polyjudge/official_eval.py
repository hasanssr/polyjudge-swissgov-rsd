"""Shim for the organizers' `compute_correlations` (their code is NOT vendored: their repo has no licence).

Nothing is imported at module load, so `make run` (predictions only) never needs the organizers' code, nlpstats or scipy.
`make eval` clones github.com/ZurichNLP/SwissGov-RSD at the pinned commit inside the container and points ORGANIZERS_REPO at it;
the function is then extracted from their script (AST) and run unchanged, exactly as in the development project.
"""
import ast
import os
from pathlib import Path

_fn = None


def _load():
    global _fn
    if _fn is None:
        repo = os.environ.get("ORGANIZERS_REPO")
        script = Path(repo or "/nonexistent") / "scripts" / "evaluate_predictions_admin.py"
        if not repo or not script.is_file():
            raise RuntimeError("official scoring needs the organizers' repo: run `make eval` (it clones it and sets ORGANIZERS_REPO)")
        import numpy as np
        from nlpstats.correlations import bootstrap, correlate
        src = script.read_text(encoding="utf-8")
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compute_correlations")
        ns = {"np": np, "correlate": correlate, "bootstrap": bootstrap}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), ns)
        _fn = ns["compute_correlations"]
    return _fn


def official_spearman(pred_labels: list[float], gold_labels: list[float]) -> float:
    """Their function returns NaN when predictions are constant on the non-punctuation tokens; we report 0.0 (= no signal) then."""
    import numpy as np
    kept = [p for p, g in zip(pred_labels, gold_labels) if g != -1]
    if not kept or min(kept) == max(kept):
        return 0.0
    (rho, _), _ = _load()(pred_labels, gold_labels, show_std=False)
    return float(np.asarray(rho).squeeze())
