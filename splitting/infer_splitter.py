#!/usr/bin/env python3
#infer_splitter.py
# -*- coding: utf-8 -*-

import argparse, os, re, json, sys
from typing import List, Dict, Any, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# -----------------------------
# Cache helpers (avoid disk quota in $HOME)
# -----------------------------
def get_cache_dir() -> Optional[str]:
    """
    Returns a preferred cache directory if configured via env.
    Priority:
      1) HF_HOME
      2) HUGGINGFACE_HUB_CACHE
    Otherwise, None -> default HF cache.
    """
    return os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or None


# -----------------------------
# Persona-aware instruction & exemplars
# -----------------------------
INSTR_TEMPLATE = (
    "Split the paper into {n_sections} logical sections tailored to the specified persona.\n"
    "For each section, return a JSON object with 'title' and 'description'.\n"
    "Titles and descriptions MUST reflect the persona's knowledge level and goals.\n"
    "Respond ONLY with a JSON list (no extra text)."
)

PERSONA_RUBRICS: Dict[str, Dict[str, Any]] = {
    "General Public": {
        "style": ("Avoid jargon; use accessible, curiosity-driven titles; "
                  "explain why the section matters and real-world impact."),
        "must_have_any": ["Background", "Real-world impact", "Why it matters"],
        "avoid": ["Ablation", "Complexity analysis"]
    },
    "Student": {
        "style": ("Use student-friendly titles; briefly define terms; connect ideas; "
                  "add learning takeaways in the description."),
        "must_have_any": ["Background", "Key Concepts", "Takeaways"],
        "avoid": []
    },
    "Teacher": {
        "style": ("Organize for teaching; emphasize learning objectives, datasets, evaluation setup, and limitations "
                  "to facilitate classroom discussion."),
        "must_have_any": ["Learning Objectives", "Evaluation Setup", "Limitations"],
        "avoid": []
    },
    "Researcher": {
        "style": ("Emphasize novelty, model/algorithm design choices, dataset composition, baselines, metrics, "
                  "ablations, and threats to validity."),
        "must_have_any": ["Model Architecture", "Training and Evaluation", "Ablation", "Results"],
        "avoid": []
    },
    "Engineer": {
        "style": ("Highlight implementation details, data formats, preprocessing, reproducibility, and integration concerns."),
        "must_have_any": ["Implementation Details", "Reproducibility", "Data Pipeline"],
        "avoid": []
    },
    "Clinician": {
        "style": ("Prioritize clinical relevance, cohorts, endpoints, inclusion/exclusion criteria, and validation on clinical outcomes."),
        "must_have_any": ["Clinical Relevance", "Validation", "Cohort/Endpoints"],
        "avoid": []
    },
    "Product Manager": {
        "style": ("Focus on user value, risks, constraints, scalability, and regulatory considerations; avoid low-level math."),
        "must_have_any": ["Value Proposition", "Risks & Constraints", "Scalability"],
        "avoid": ["Training Loss Curves"]
    },
    "Investor": {
        "style": ("Prioritize market context, competitive landscape, potential applications, risks, and differentiators. "
                  "Keep technical depth shallow unless directly tied to moat or scale."),
        "must_have_any": ["Background", "Market/Applications", "Risks", "Differentiators"],
        "avoid": ["Ablation", "Proofs"]
    },
    "Reviewer": {
        "style": ("Be critical; surface assumptions, confounders, missing ablations, generalization claims, and threats to validity."),
        "must_have_any": ["Assumptions", "Limitations", "Ablations"],
        "avoid": []
    },
}

FEW_SHOT = """
### Few-shot Persona Exemplars (for shaping section granularity)

[Persona: Investor]
Output (example):
[
  {"title":"Background on Generative Models","description":"Brief context and progress relevant for non-experts."},
  {"title":"Jukebox Architecture","description":"High-level components tied to differentiation."},
  {"title":"Controlling Music Generation","description":"What knobs exist and why they matter commercially."},
  {"title":"Sampling Methods","description":"What affects output quality/time-to-sample."},
  {"title":"Applications & Moat","description":"Potential markets, risks, differentiators."}
]

[Persona: Researcher]
Output (example):
[
  {"title":"Model Architecture","description":"VQ-VAE hierarchy, priors, upsamplers."},
  {"title":"Training and Evaluation","description":"Objectives, spectral loss, datasets, metrics."},
  {"title":"Controlling Generation","description":"Conditioning details and limitations."},
  {"title":"Results and Ablations","description":"Comparisons, ablations, failure modes."},
  {"title":"Related Work","description":"Positioning vs prior art."}
]
""".strip()

def _persona_block(persona: str, style_strength: int, n_sections: int) -> str:
    spec = PERSONA_RUBRICS.get(persona, PERSONA_RUBRICS["General Public"])
    must_any = spec.get("must_have_any", [])
    avoid = spec.get("avoid", [])
    style = spec.get("style", "")

    lines = [
        f"Persona: {persona}",
        f"Persona guidelines (strength {style_strength}/5): {style}",
        f"Target sections: {n_sections}",
        "All section titles/descriptions MUST be grounded in the paper text; no inventions.",
    ]
    if must_any:
        lines.append(f"At least one section title MUST include one of: {', '.join(must_any)}.")
    if avoid:
        lines.append(f"Avoid section titles about: {', '.join(avoid)}.")
    return "\n".join(lines)


# -----------------------------
# Markdown cleaning
# -----------------------------
COMMENT_RE = re.compile(r'^\s*<!--\s*(\{.*?\})\s*-->\s*$', re.IGNORECASE)
H1_RE = re.compile(r'^\s*#\s+(.*)')
HX_RE = re.compile(r'^\s*#{1,6}\s+(.*)')
STOP_HEAD_RE = re.compile(
    r'^\s*#{1,6}\s+(references?|bibliography|appendix|acknowledgements?)\b',
    re.IGNORECASE
)

def _is_table_line(line: str) -> bool:
    if line.strip().startswith("|"):
        return True
    s = line.strip().replace(" ", "")
    if set(s) <= {"|", "-", ":"} and ("|" in s or "-" in s):
        return True
    return False

def _looks_like_figure_caption(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r'^(fig(\.|ure)?|tab(\.|le)?)\s*[\d:.\- ]', s, re.IGNORECASE))

def fix_line_wraps(text: str) -> str:
    """
    Fix common PDF line-wrapping artifacts without harming real hyphenated compounds.
    - Remove soft hyphen (U+00AD).
    - De-hyphenate word breaks across newlines: 'intro-\\nduces' -> 'introduces'.
    - De-hyphenate word breaks with spaces: 'intro-   duces' -> 'introduces'.
    - Merge single newlines inside sentences: 'models,\\nwhich' -> 'models, which'.
    """
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)[-\u2010\u2011]\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\b[A-Za-z]{2,})-\s+([a-z]{2,}\b)", r"\1\2", text)
    text = re.sub(r"([a-z,;:])\n([a-z])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

DROP_HEADING_RE = re.compile(r'^\s*#{1,6}\s*(table|fig(?:\.|ure)?)\b', re.IGNORECASE)

def normalize_spacing_and_punct(text: str) -> str:
    text = re.sub(r'\s+([,.;:!?%])', r'\1', text)
    text = re.sub(r' +([\)\]\}])', r'\1', text)
    text = re.sub(r'([\(\[\{]) +', r'\1', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'\s+’s\b', "’s", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r'\bbest\s*ranked\b', 'best-ranked', text, flags=re.IGNORECASE)
    return text

def fix_common_academic_artifacts(text: str) -> str:
    pairs = {
        r'\blargescale\b': 'large-scale',
        r'\binterrater\b': 'inter-rater',
        r'\bwithing\b': 'within',
        r'\bnnU-\s*Net\b': 'nnU-Net',
    }
    for pat, rep in pairs.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text

def drop_figure_table_headings(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if DROP_HEADING_RE.match(ln):
            continue
        lines.append(ln)
    return "\n".join(lines)

def clean_pdf_markdown(md_text: str) -> Tuple[Optional[str], str]:
    """
    - Removes blocks marked by HTML comments (table/image/figure/header/footer)
    - Removes markdown table lines and inline images ![](...)
    - Stops at '## References/Appendix/...'
    - Extracts the H1 as title (if present)
    """
    title = None
    out_lines: List[str] = []

    skip_types = {"table", "image", "figure", "header", "footer"}
    current_block_type: Optional[str] = None
    stop_now = False

    lines = md_text.splitlines()
    for line in lines:
        if stop_now:
            break

        m = COMMENT_RE.match(line)
        if m:
            try:
                meta = json.loads(m.group(1))
                t = str(meta.get("type", "")).lower()
                current_block_type = t if t else None
            except Exception:
                current_block_type = None
            continue

        if STOP_HEAD_RE.match(line):
            stop_now = True
            break

        if current_block_type in skip_types:
            continue

        if _is_table_line(line):
            continue
        if re.search(r'!\[[^\]]*\]\([^)]+\)', line):
            continue
        if _looks_like_figure_caption(line):
            continue

        m1 = H1_RE.match(line)
        if m1 and title is None:
            title = m1.group(1).strip()
            continue

        out_lines.append(line)

        if HX_RE.match(line):
            current_block_type = None

    text = "\n".join(out_lines).strip()
    text = fix_line_wraps(text)
    text = drop_figure_table_headings(text)
    text = normalize_spacing_and_punct(text)
    text = fix_common_academic_artifacts(text)
    return title, text


# -----------------------------
# Prompt building
# -----------------------------
TEXT_KEYS = ["paper_text", "text", "content", "full_text", "body"]
TITLE_KEYS = ["paper_title", "title"]

def build_prompt(persona: str, title: str, paper_text: str,
                 n_sections: int = 5, style_strength: int = 3,
                 include_few_shot: bool = False) -> str:
    persona = persona or "General Public"
    user_lines = []
    if title:
        user_lines.append(f"Title: {title}")
    user_lines.append(_persona_block(persona, style_strength, n_sections))
    user_lines.append("Paper text:")
    user_lines.append(paper_text)
    user_text = "\n".join(user_lines)

    instr = INSTR_TEMPLATE.format(n_sections=n_sections if isinstance(n_sections, int) else "~5")
    few_shot = (f"\n\n### Exemplars:\n{FEW_SHOT}\n" if include_few_shot else "")

    prompt = (
        f"### Instruction:\n{instr}{few_shot}\n"
        f"### Input:\n{user_text}\n\n"
        f"### Response:\n"
    )
    return prompt


def extract_json_list(text: str) -> List[Dict[str, Any]]:
    """
    Extracts the FIRST plausible JSON list [...] from generated text.
    """
    s = text.strip()
    if "### Response:" in s:
        s = s.split("### Response:", 1)[1].strip()
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return []


# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(base_model: str, adapter_path: str):
    cache_dir = get_cache_dir()
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0",
        cache_dir=cache_dir,
    )
    try:
        model = PeftModel.from_pretrained(base, adapter_path, cache_dir=cache_dir)
    except TypeError:
        # Older PEFT may not support cache_dir
        model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tok, model


# -----------------------------
# Generation
# -----------------------------
def generate_sections(
    tok, model,
    persona: str, title: str, paper_text: str,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.9,
    n_sections: int = 5,
    style_strength: int = 3,
    include_few_shot: bool = False):

    prompt = build_prompt(
        persona, title, paper_text,
        n_sections=n_sections, style_strength=style_strength,
        include_few_shot=include_few_shot
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    do_sample = temperature > 0.0
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_p=float(top_p) if do_sample else 1.0,
        do_sample=do_sample,
        repetition_penalty=1.05 if do_sample else 1.0,
        typical_p=0.95 if do_sample else None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    text_out = tok.decode(out[0], skip_special_tokens=True)
    sections = extract_json_list(text_out)

    cleaned = []
    for it in sections:
        if isinstance(it, dict):
            t = str(it.get("title", "")).strip()
            d = str(it.get("description", "")).strip()
            if t or d:
                cleaned.append({"title": t, "description": d})
    return cleaned


# -----------------------------
# Input handling
# -----------------------------
def _load_inputs_any(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - .jsonl with records containing: persona + (paper_text|text|content) OR markdown in 'md'
      - .md: requires --persona
      - .txt: same as .md
    """
    if path.lower().endswith(".jsonl"):
        ds = load_dataset("json", data_files=path, split="train")
        records = []
        for ex in ds:
            persona = ex.get("persona") or "General Public"
            md = ex.get("md")
            raw_text = None
            if md:
                raw_text = md
            else:
                for k in TEXT_KEYS:
                    if k in ex and ex[k]:
                        raw_text = ex[k]
                        break
            if raw_text is None:
                continue
            title = None
            for tk in TITLE_KEYS:
                if tk in ex and ex[tk]:
                    title = ex[tk]
                    break
            rec_id = ex.get("id") or ex.get("id_paper") or ex.get("paper_id") or None
            records.append({"id": rec_id, "persona": persona, "md": raw_text, "title": title})
        return records
    else:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{"id": None, "persona": None, "md": content, "title": None}]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True,
                    help="e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", type=str, required=True,
                    help="e.g. out-splitter-qwen7b/checkpoint-100")
    ap.add_argument("--input_path", type=str, required=True,
                    help="Markdown/TXT file OR JSONL with persona+text or md")
    ap.add_argument("--output_jsonl", type=str, required=True,
                    help="Output JSONL with sections and metadata")

    ap.add_argument("--persona", type=str, default=None,
                    help="Required if input_path is a single .md/.txt file")
    ap.add_argument("--id", type=str, default=None,
                    help="Optional id for single file input")

    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Creativity of section generation (0.0 = deterministic)")
    ap.add_argument("--top_p", type=float, default=0.9,
                help="Nucleus sampling for splitter when temperature > 0.")

    # Persona-aware knobs
    ap.add_argument("--sections", type=int, default=5,
                    help="Target number of sections to return.")
    ap.add_argument("--style_strength", type=int, default=3,
                    help="How strongly to bias titles/descriptions toward persona (1–5).")
    ap.add_argument("--few_shot", action="store_true",
                    help="Include short persona exemplars in the prompt.")

    args = ap.parse_args()

    # Load model
    tok, model = load_model_and_tokenizer(args.base_model, args.adapter_path)

    # Load inputs
    inputs = _load_inputs_any(args.input_path)

    # For single file inputs, require persona/id from CLI
    if len(inputs) == 1 and inputs[0]["persona"] is None:
        if not args.persona:
            print("Error: --persona is required for .md/.txt input", file=sys.stderr)
            sys.exit(2)
        inputs[0]["persona"] = args.persona
        inputs[0]["id"] = args.id

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    n_ok = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for ex in inputs:
            raw_md = ex["md"]
            persona = ex["persona"] or "General Public"
            pre_title = ex.get("title") or ""
            rid = ex.get("id")

            # Clean markdown and get title
            title_md, cleaned_text = clean_pdf_markdown(raw_md)
            title = pre_title or title_md or ""

            # se l'utente NON ha specificato style_strength, ricavalo dalla temperature
            style_strength = args.style_strength
            if style_strength is None or str(style_strength).strip() == "":
                # mappa 0.0 → 2, 0.5 → 4, 1.0 → 5 (clamp 1..5)
                style_strength = max(1, min(5, int(round(2 + args.temperature * 3))))


            # Generate sections
            sections = generate_sections(
                tok, model, persona, title, cleaned_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,                  # <— importante
                n_sections=args.sections,
                style_strength=style_strength,     # se l’hai derivato dalla temp, usa la variabile
                include_few_shot=bool(args.few_shot),
            )

            # Record out
            rec_out = {
                "id": rid,
                "persona": persona,
                "paper_title": title,
                "sections": sections,
                "num_sections": len(sections),
                "temperature": args.temperature,
                "cleaned_text": cleaned_text,
                # extra metadata for audit/debug
                "n_sections_target": args.sections,
                "style_strength": args.style_strength,
                "few_shot": bool(args.few_shot),
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[✓] Saved -> {args.output_jsonl} ({n_ok} records)")


if __name__ == "__main__":
    main()
