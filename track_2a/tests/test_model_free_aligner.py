"""The sentence pairing used inside M1 and M2 must be completely model-free (challenge-owner confirmation of 2026-10-05, see docs/prereg.md addendum):
no LLM client, no embedding model, no API call path anywhere in its import closure, statically and at runtime."""
import ast
import inspect
import subprocess
import sys
from pathlib import Path

import polyjudge
from polyjudge import predict as predict_mod
from polyjudge.aligner import BEST_V2, HeuristicAlignerV2

PKG = Path(polyjudge.__file__).parent
PAIRING = {"aligner", "segmentation", "heuristic", "groups"}                    # sentence segmentation + alignment + bead construction
FLAGS = PAIRING | {"spans", "detlayer", "candidates"}                            # + model-free candidate flags used by M1 / M2
FORBIDDEN_LOCAL = {"llm", "config", "verifier", "debate", "lla", "judge", "pipeline", "prompts", "prompts_v2", "diffs", "predict", "inputs", "data"}
FORBIDDEN_THIRD_PARTY = {"openai", "httpx", "requests", "aiohttp", "urllib3", "anthropic", "torch", "transformers", "sentence_transformers", "sklearn",
                         "tokenizers", "huggingface_hub", "datasets", "numpy", "scipy", "pandas"}
FORBIDDEN_TOKENS = ("LLMClient", "AsyncOpenAI", "load_llm_config", ".chat(", "api_key", "base_url", "embedding")


def _imports(module: str):
    """(local polyjudge modules, third-party roots) imported by polyjudge/<module>.py (top level and function level)."""
    tree = ast.parse((PKG / f"{module}.py").read_text(encoding="utf-8"))
    local, third = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:                                               # from .x import y   /   from . import x
                if node.module:
                    local.add(node.module.split(".")[0])
                else:
                    local.update(a.name for a in node.names)
            elif node.module:
                root = node.module.split(".")[0]
                (local.add(node.module.split(".")[1]) if root == "polyjudge" and "." in node.module else third.add(root))
        elif isinstance(node, ast.Import):
            for a in node.names:
                third.add(a.name.split(".")[0])
    return local, third


def _closure(start: set[str]):
    seen, todo, third_all = set(), list(start), set()
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        local, third = _imports(m)
        third_all |= third
        todo += [x for x in local if (PKG / f"{x}.py").exists()]
    return seen, third_all


def test_pairing_stack_import_closure_is_model_free():
    closure, third = _closure(PAIRING)
    assert closure <= PAIRING, f"the pairing stack imports other polyjudge modules: {closure - PAIRING}"
    assert not closure & FORBIDDEN_LOCAL
    assert not third & FORBIDDEN_THIRD_PARTY, third & FORBIDDEN_THIRD_PARTY


def test_flag_generation_stack_is_model_free_too():
    closure, third = _closure(FLAGS)
    assert closure <= FLAGS and not closure & FORBIDDEN_LOCAL
    assert not third & {"openai", "httpx", "requests", "aiohttp", "torch", "transformers", "sentence_transformers", "sklearn", "huggingface_hub", "datasets"}


def test_no_model_or_api_tokens_in_the_pairing_sources():
    for m in FLAGS:
        src = (PKG / f"{m}.py").read_text(encoding="utf-8")
        for tok in FORBIDDEN_TOKENS:
            assert tok not in src, f"{m}.py mentions {tok!r}"


def test_runtime_import_and_alignment_load_no_model_or_network_code():
    code = (
        "import sys\n"
        "from polyjudge.aligner import BEST_V2, HeuristicAlignerV2\n"
        "import polyjudge.groups, polyjudge.detlayer, polyjudge.candidates\n"
        "from polyjudge.segmentation import segment_document\n"
        "en = segment_document('A first sentence . A second one .', 'A first sentence. A second one.', 'en')\n"
        "tg = segment_document('Ein erster Satz . Ein zweiter .', 'Ein erster Satz. Ein zweiter.', 'de')\n"
        "links = HeuristicAlignerV2(**BEST_V2).align(en, tg, 'de')\n"
        "assert links\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in %r or m in %r)\n"
        "print('LOADED_FORBIDDEN=' + ','.join(bad))\n"
    ) % (FORBIDDEN_THIRD_PARTY - {"numpy", "scipy"}, {f"polyjudge.{m}" for m in FORBIDDEN_LOCAL})
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PKG.parent, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().endswith("LOADED_FORBIDDEN="), out.stdout


def test_m1_and_m2_use_the_v2_aligner_and_never_the_llm_alignment_code():
    assert isinstance(predict_mod.AL, HeuristicAlignerV2)
    for fn in (predict_mod.run_m1, predict_mod.run_m2, predict_mod._det_candidates):
        src = inspect.getsource(fn)
        assert "AL" in src
        for name in ("run_items", "build_items", "run_judge", "presence_labels"):          # the model-based alignment belongs to M3 only
            assert name not in src, f"{fn.__name__} uses {name}"
    assert "run_items" in inspect.getsource(predict_mod.run_m3) and BEST_V2["complex_penalty"] == 1.5
