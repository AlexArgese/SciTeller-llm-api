#!/usr/bin/env python3
# infer_from_splits.py — Storyteller per-sezione con retrieval robusto, prompt persona-aware e parsing resistente

import os, re, json, argparse, math, random, sys
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

# ---- lightweight tracing (console + optional NDJSON file from env) ----
from datetime import datetime

TRACE_LOG_FILE = os.environ.get("TRACE_LOG_FILE")
TRACE_REQ_ID = os.environ.get("TRACE_REQ_ID", "-")

def _now_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def trace(event: str, message: str = "", **data):
    rec = {"ts": _now_iso(), "req_id": TRACE_REQ_ID, "event": event, "message": message, **data}
    line = json.dumps(rec, ensure_ascii=False)
    # write to file (if provided)
    if TRACE_LOG_FILE:
        try:
            with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
    # always echo to console (stderr) so you see progress immediately
    try:
        print(f"[trace] {event} {message} :: {json.dumps(data, ensure_ascii=False)}", file=sys.stderr, flush=True)
    except Exception:
        pass

# ===============================
# CONFIG BASE
# ===============================
BASE_MODEL  = "Qwen/Qwen2.5-32B-Instruct"
ADAPTER_DIR = "qwen32b_storyteller_lora/final_best"

# BF16 se possibile
BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

# ===============================
# SEED & UTIL
# ===============================
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def max_ctx_len(model) -> int:
    return int(getattr(model.config, "max_position_embeddings", 8192))

def decode_clean_line(s: str) -> str:
    s = s.strip()
    s = s.splitlines()[0].strip()
    s = s.strip('"').strip("'").strip("`")
    s = re.sub(r'^(Human|User|Assistant|System)\s*:\s*', '', s, flags=re.I)
    s = re.sub(r'\s{2,}', ' ', s)
    # rimuovi CJK
    s = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', s)
    return s.strip()

# ===============================
# PERSONA GUIDANCE (8 ruoli)
# ===============================
PERSONA_GUIDE: Dict[str, Dict[str, str]] = {
    "General Public": {
        "expertise": "Low",
        "goal": "Understand what AI is and why it matters.",
        "style": ("Use simple, curiosity-driven language. Avoid jargon and equations. "
                  "Give 1–2 relatable examples or analogies and explain why this matters.")
    },
    "Investor": {
        "expertise": "Low–Medium",
        "goal": "Spot AI trends for business or funding decisions.",
        "style": ("Focus on market potential, differentiation, scalability, and risks. "
                  "Explain technical ideas only when tied to business value.")
    },
    "Student": {
        "expertise": "Medium",
        "goal": "Learn AI fundamentals and expand technical knowledge.",
        "style": ("Use educational tone with short definitions and an intuitive example. "
                  "Highlight motivation, key concepts, and takeaways.")
    },
    "Journalist": {
        "expertise": "Medium",
        "goal": "Report clearly and accurately on AI developments.",
        "style": ("Explain for an informed non-technical audience. "
                  "Emphasize significance, evidence, and societal implications.")
    },
    "Developer": {
        "expertise": "Medium–High",
        "goal": "Apply models, tools, and frameworks in real-world projects.",
        "style": ("Be practical: implementation details, datasets, evaluation settings, reproducibility, APIs/tools.")
    },
    "Policy Maker": {
        "expertise": "Medium–High",
        "goal": "Assess the social, ethical, and legal implications of AI.",
        "style": ("Prioritize governance, transparency, accountability, risks, and societal impact. "
                  "Avoid deep technical dives unless necessary.")
    },
    "Teacher": {
        "expertise": "High",
        "goal": "Teach AI concepts and methods effectively to others.",
        "style": ("Organize clearly: learning objectives, examples, misconceptions and limitations "
                  "to foster critical thinking.")
    },
    "Researcher": {
        "expertise": "High",
        "goal": "Produce or track cutting-edge AI research and experimental results.",
        "style": ("Be concise and technical. Focus on novelty, methodology, datasets, metrics, results, and limitations.")
    },
}

def persona_block(persona: str) -> str:
    spec = PERSONA_GUIDE.get(persona, PERSONA_GUIDE["General Public"])
    return (
        f"Persona: {persona}\n"
        f"Expertise: {spec['expertise']}\n"
        f"Goal: {spec['goal']}\n"
        f"Persona style: {spec['style']}\n"
        "Each section must serve the Persona Goal above.\n"
    )

# ===============================
# JSON / TEXT HELPERS
# ===============================
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    return s.strip()

def extract_first_json_object(s: str) -> str:
    s = _strip_code_fences(s)
    start = s.find("{")
    if start < 0: return s.strip()
    i, depth, in_str, esc = start, 0, False, False
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return s[start:i+1]
        i += 1
    return s[start:].strip()

def clean_plain_text(txt: str) -> str:
    t = _strip_code_fences(str(txt))
    t = t.strip().strip('"').strip("'")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', t)
    return t.strip()

# ===============================
# ANTI-HALLUCINATION (leggera)
# ===============================
CAP_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
SAFE_WORDS = {"json","section","introduction","conclusion","authors","framework"}

def cap_entities(text:str):
    return set(CAP_RE.findall(text))

def too_many_new_entities(gen_text:str, ctx:str, max_new_ratio=0.05, min_total=1):
    ctx_cap = cap_entities(ctx)
    out_cap = cap_entities(gen_text)
    if not out_cap: return False
    new_ents = [e for e in out_cap if (e not in ctx_cap and e.lower() not in SAFE_WORDS)]
    return (len(out_cap) >= min_total) and (len(new_ents)/max(1,len(out_cap)) > max_new_ratio)

def strip_unknown_entities(txt, ctx):
    wl = cap_entities(ctx)
    sents = re.split(r'(?<=[.!?])\s+', txt)
    keep = []
    for s in sents:
        ents = cap_entities(s)
        if not ents or all((e in wl) or (e.lower() in SAFE_WORDS) for e in ents):
            keep.append(s)
    return " ".join(keep).strip()

# ===============================
# RETRIEVAL
# ===============================
def segment_text(text:str, max_words:int=180, overlap:int=60) -> List[str]:
    words = (text or "").split()
    if not words: return []
    segs, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        seg = " ".join(words[i:i+max_words]).strip()
        if seg: segs.append(seg)
        i += step
    return segs

class Retriever:
    """
    Usa sentence-transformers (se disponibile) altrimenti TF-IDF.
    Riutilizzabile su più paper (ricarica la corpus per ciascun paper).
    """
    def __init__(self, method:str="auto", model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        self.method_in = method
        self.method = method
        self.model_name = model_name
        self.st_model = None
        self.vec = None
        self.corpus_mat = None
        self.corpus: List[str] = []
        if method in ("auto","emb"):
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(model_name)
                self.method = "emb"
            except Exception:
                self.method = "tfidf"
        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1,2), max_features=120000, lowercase=True)

    def fit(self, corpus: List[str]):
        self.corpus = corpus[:]
        if not corpus:
            self.corpus_mat = None
            return
        if self.method == "emb":
            embs = self.st_model.encode(corpus, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            self.corpus_mat = np.asarray(embs, dtype=np.float32)
        else:
            self.corpus_mat = self.vec.fit_transform(corpus)

    def topk(self, query:str, k:int=5) -> List[Tuple[int,float]]:
        if self.corpus_mat is None or self.corpus_mat.shape[0] == 0:
            return []
        if not self.corpus or self.corpus_mat is None: return []
        if self.method == "emb":
            q = self.st_model.encode([query], normalize_embeddings=True)[0]
            sims = (self.corpus_mat @ q)
            idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in idx]
        else:
            qv = self.vec.transform([query])
            sims = (qv @ self.corpus_mat.T).toarray().ravel()
            idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in idx]

def build_section_context(sec_title: str, sec_desc: str, ret: Retriever, k: int = 6, max_chars: int = 2500) -> str:
    query = (sec_title.strip() + " — " + (sec_desc or "")).strip()

    # Guardie sul retriever/corpus
    if ret is None or getattr(ret, "corpus_mat", None) is None:
        return ""
    n = getattr(ret.corpus_mat, "shape", (0, 0))[0]
    if n <= 0:
        return ""

    # limita k alla size del corpus
    k_eff = min(max(1, int(k)), n)

    hits = ret.topk(query, k=k_eff)  # <-- fix: 'ret', non 'self'
    ctx = ""
    for i, _score in hits:
        frag = ret.corpus[i].strip()
        if not frag:
            continue
        if len(ctx) + len(frag) + 2 > max_chars:
            break
        ctx += frag + "\n\n"
    return ctx.strip()


# ===============================
# PROMPT BUILDERS
# ===============================
# Regole base (concise) — singola sezione
BASE_RULES = (
    "- Output ONLY a single JSON object (no markdown, no preface).\n"
    "- Schema:\n"
    "  {\"title\": \"<exact section title>\", \"text\": \"<2–4 short paragraphs separated by one blank line>\"}\n"
    "- Use English only.\n"
    "- Be faithful to the paper context; NEVER invent citations, names, affiliations, or numbers.\n"
    "- If information is missing from the context, write 'not specified in the paper'.\n"
    "- Avoid bullet lists; write smooth narrative prose.\n"
)

def build_section_prompt(
    persona:str, paper_title:str, target_title:str, target_desc:str,
    context_chunk:str, length_preset:str="medium", prev_title:Optional[str]=None
) -> str:
    # Preset di lunghezza
    preset = (length_preset or "medium").lower()
    if preset == "short":
        length_rule = "- Keep this section short: ~2 short paragraphs.\n"
    elif preset == "long":
        length_rule = "- Write a longer section: ~3–4 paragraphs.\n"
    else:
        length_rule = "- Keep it concise: ~2–3 short paragraphs.\n"

    transition_line = ""
    if prev_title:
        transition_line = (f"- The opening sentence should connect naturally from the previous section "
                           f"('{prev_title}') without repeating it.\n")

    pblock = persona_block(persona)

    prompt = (
        "You are an AI Scientist Storyteller.\n"
        f"{pblock}\n"
        f"Paper title: {paper_title}\n"
        "Task: Write ONE coherent section of the story, strictly matching the given target title.\n"
        "Rules:\n"
        f"{BASE_RULES}"
        f"{length_rule}"
        f"{transition_line}"
        "- Do NOT change the section title.\n"
        "- Paraphrase faithfully instead of copying long passages.\n"
        "\nTarget section:\n"
        f"- Title: {target_title}\n"
    )
    if target_desc:
        prompt += f"- Description: {target_desc}\n"
    prompt += (
        "\nPaper context (retrieved, factual grounding):\n"
        f"\"\"\"{context_chunk if context_chunk else '[NO CONTEXT FOUND]'}\"\"\"\n"
        "\nReturn ONLY the JSON object for this single section.\n"
    )
    return prompt

def build_title_prompt(persona:str, paper_title:str, outline_titles:List[str]) -> str:
    avoid = "; ".join(t for t in outline_titles if t)
    pblock = persona_block(persona)
    return (
        "You are an AI Scientist Storyteller.\n"
        f"{pblock}\n"
        f"Paper title: {paper_title}\n\n"
        "Task: Propose ONE catchy, persona-tailored story title (<= 10 words).\n"
        "Constraints:\n"
        "- Do NOT reuse any section title verbatim.\n"
        "- No quotes, no markdown, English only.\n"
        f"- Avoid these exact strings: {avoid}\n"
        "Return ONLY the title text.\n"
    )

# ===============================
# MODEL LOAD
# ===============================
def load_model_and_tokenizer(adapter_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(torch.bfloat16 if BF16 else torch.float16),
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=(torch.bfloat16 if BF16 else torch.float16),
        quantization_config=bnb,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return tok, model

# ===============================
# GENERATION HELPERS
# ===============================
@torch.no_grad()
def generate_once(tokenizer, model, prompt: str, gen_cfg: GenerationConfig, safety_margin: int = 128) -> str:
    # 1) Budget: prompt + max_new <= max_ctx - margin
    max_pos = int(getattr(model.config, "max_position_embeddings", 8192))
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    budget_input = max_pos - int(gen_cfg.max_new_tokens or 0) - safety_margin
    if budget_input <= 0:
        gen_cfg.max_new_tokens = max(64, int(gen_cfg.max_new_tokens or 128))
        budget_input = max(256, max_pos - int(gen_cfg.max_new_tokens) - safety_margin)

    # 2) Trim in token-space poi RITOKENIZZA testo (evita mismatch con cache_position)
    if len(ids) > budget_input:
        ids = ids[-budget_input:]
    prompt_trim = tokenizer.decode(ids, skip_special_tokens=True)

    enc = tokenizer(
        prompt_trim,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_pos - safety_margin
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # 3) Qwen2 safe: niente min_new_tokens e cache disattivata in generate
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_cfg,
        use_cache=True,      
    )

    gen_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

# ===============================
# MAIN
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl",  required=True, help="JSONL from splitter (outline + cleaned_text + persona/title)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with the generations")
    ap.add_argument("--adapter",    default=ADAPTER_DIR, help="LoRA adapter dir (final_best)")
    # generazione
    ap.add_argument("--preset", default="medium", help="short|medium|long")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=700)
    # retrieval
    ap.add_argument("--retriever", default="auto", help="auto|emb|tfidf")
    ap.add_argument("--retriever_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=3, help="top-k paragrafi per sezione (default 3)")
    ap.add_argument("--max_ctx_chars", type=int, default=1400)
    ap.add_argument("--seg_words", type=int, default=180)
    ap.add_argument("--overlap_words", type=int, default=60)
    # altri
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # seed
    set_all_seeds(args.seed)

    # modello
    tok, model = load_model_and_tokenizer(args.adapter)

    trace("story.model_loaded", "Storyteller model loaded.",
          base_model=BASE_MODEL, adapter=args.adapter, preset=args.preset,
          temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)


    # retrieval riutilizzabile
    ret = Retriever(method=args.retriever, model_name=args.retriever_model)

    # config generazione base
    temp  = float(args.temperature)
    top_p = float(args.top_p)
    do_sample = temp > 0.0

    base_cfg = dict(
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # lettura input
    n_items, ok = 0, 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:

        lines = [ln for ln in fin if ln.strip()]
        for ln in tqdm(lines, desc="Inferencing", unit="item"):

            n_items += 1
            item = json.loads(ln)

            persona = item.get("persona") or "General Public"
            paper_title = item.get("paper_title") or "Untitled"
            outline = item.get("sections") or []
            trace("story.item.start", "Starting story generation for item.",
                  persona=persona, paper_title=paper_title,
                  outline_count=len(outline))
            ctx_full = item.get("cleaned_text") or ""

            # BYPASS PROMPT DIRETTO (rigenerazione paragrafo, ecc.)
            custom_prompt = item.get("prompt")
            if custom_prompt:
                # cfg breve e sampling controllato
                gen_cfg_direct = GenerationConfig(
                    max_new_tokens=384,
                    do_sample=(temp > 0.0),
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=(1.10 if temp > 0 else 1.0),
                    **base_cfg,
                )
                gen_full = generate_once(tok, model, custom_prompt, gen_cfg_direct)
                gen_json_text = extract_first_json_object(gen_full)
                fout.write(json.dumps({
                    "id": item.get("id"),
                    "persona": persona,
                    "generation": {"text": gen_json_text}
                }, ensure_ascii=False) + "\n")
                continue

            # --- Build retrieval corpus una volta per paper ---
            paragraphs = segment_text(ctx_full, max_words=args.seg_words, overlap=args.overlap_words)
            ret.fit(paragraphs)

            # numero sezioni target
            SEC_COUNT = 5
            target_n = len(outline) if isinstance(outline, list) and len(outline)>0 else SEC_COUNT

            # budget per sezione (senza min_new_tokens)
            per_sec_max = max(220, int(args.max_new_tokens // max(1, target_n)))
            if args.preset == "long":
                per_sec_min = max(120, int(0.50 * per_sec_max))
            elif args.preset == "short":
                per_sec_min = 0
            else:
                per_sec_min = max(60, int(0.35 * per_sec_max))

            if do_sample:
                gen_cfg = GenerationConfig(
                    max_new_tokens=per_sec_max,
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=1.10,
                    **base_cfg,
                )
            else:
                gen_cfg = GenerationConfig(
                    max_new_tokens=per_sec_max,
                    do_sample=False,
                    **base_cfg,
                )


            sections_out, valid_count = [], 0

            for i in range(target_n):
                sec = outline[i] if i < len(outline) else {}
                sec_title = (sec.get("title") or f"Section {i+1}").strip()
                sec_desc  = (sec.get("description") or "").strip()
                prev_title = (outline[i-1]["title"].strip() if i > 0 and isinstance(outline[i-1], dict) and outline[i-1].get("title") else None)

                trace("story.retriever.ready", "Retriever ready.",
                  method=ret.method, corpus_segments=len(paragraphs))

                # === CONTEXT RETRIEVAL per sezione ===
                ctx_i = build_section_context(
                    sec_title, sec_desc, ret,
                    k=args.k, max_chars=args.max_ctx_chars
                )


                prompt = build_section_prompt(
                    persona=persona,
                    paper_title=paper_title,
                    target_title=sec_title,
                    target_desc=sec_desc,
                    context_chunk=ctx_i,
                    length_preset=args.preset,
                    prev_title=prev_title,
                )
                trace("story.section.context", "Context retrieved.",
                  index=i, ctx_chars=len(ctx_i), top_k=args.k)

                tries = 0
                while True:
                    gen_full = generate_once(tok, model, prompt, gen_cfg)
                    gen_json_text = extract_first_json_object(gen_full)
                    if not too_many_new_entities(gen_json_text, ctx_i, max_new_ratio=0.05, min_total=1):
                        break
                    tries += 1
                    if tries > 3:
                        gen_json_text = strip_unknown_entities(gen_json_text, ctx_i)
                        break
                    # rafforza promemoria anti-hallu
                    prompt += (
                        "\nSTRICT REMINDER:\n"
                        "- Use only entities present verbatim in the retrieved context above.\n"
                        "- If authors or affiliations are missing, say 'the authors' / 'the framework'.\n"
                        "- If something is not in the context, write 'not specified in the paper'.\n"
                    )
                    trace("story.section.warning", "High ratio of new entities; reinforcing constraints.",
                      index=i, try_no=tries)

                out_sec = {"title": sec_title, "text": ""}

                # parsing robusto
                try:
                    obj = json.loads(gen_json_text)
                    if isinstance(obj, dict):
                        # Caso corretto: {"title": "...", "text": "..."}
                        if isinstance(obj.get("text"), str):
                            out_sec["text"] = obj["text"].strip()

                        # Caso array sezioni: {"sections":[{title,text},...]}
                        elif isinstance(obj.get("sections"), list):
                            for it in obj["sections"]:
                                if (
                                    isinstance(it, dict)
                                    and isinstance(it.get("title"), str)
                                    and it["title"].strip() == sec_title
                                    and isinstance(it.get("text"), str)
                                ):
                                    out_sec["text"] = it["text"].strip()
                                    break

                        # Caso annidato con singola chiave
                        if not out_sec["text"]:
                            if len(obj) == 1:
                                only_v = next(iter(obj.values()))
                                if isinstance(only_v, dict) and isinstance(only_v.get("text"), str):
                                    out_sec["text"] = only_v["text"].strip()

                        # 'text' è JSON string -> unwrap
                        if out_sec["text"] and out_sec["text"].lstrip().startswith("{"):
                            try:
                                inner = json.loads(out_sec["text"])
                                if isinstance(inner, dict):
                                    if isinstance(inner.get("text"), str):
                                        out_sec["text"] = inner["text"].strip()
                                    elif len(inner) == 1:
                                        only_v = next(iter(inner.values()))
                                        if isinstance(only_v, dict) and isinstance(only_v.get("text"), str):
                                            out_sec["text"] = only_v["text"].strip()
                            except Exception:
                                pass

                        # hard-unwrap finale
                        if out_sec["text"] and "{" in out_sec["text"] and "}" in out_sec["text"] and ":" in out_sec["text"]:
                            try:
                                inner2 = json.loads(out_sec["text"])
                                if isinstance(inner2, dict) and isinstance(inner2.get("text"), str):
                                    out_sec["text"] = inner2["text"].strip()
                            except Exception:
                                out_sec["text"] = re.sub(r"[{}\[\]\"]", "", out_sec["text"]).strip()

                except Exception:
                    # fallback: prendi i primi paragrafi puliti dalla generazione completa
                    raw = _strip_code_fences(gen_full).strip()
                    paras = re.split(r"\n\s*\n", raw)
                    paras = [p.strip() for p in paras if p.strip()]
                    out_sec["text"] = "\n\n".join(paras[:4]) if paras else raw[:1200]


                # seed se vuoto
                if not out_sec["text"].strip():
                    seed = sec_desc or ctx_i[:600].strip() or ""
                    out_sec["text"] = seed

                # pulizia finale prosa
                out_sec["text"] = clean_plain_text(out_sec["text"])
                paras = [p.strip() for p in re.split(r"\n{2,}", out_sec["text"]) if p.strip()]
                if not paras:
                    paras = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])', out_sec["text"])
                out_sec["text"] = "\n\n".join([p for p in paras if p])

                if out_sec["text"].strip():
                    valid_count += 1
                sections_out.append(out_sec)

                trace("story.section.done", f"Section '{sec_title}' generated.",
                    index=i, text_chars=len(out_sec["text"]))

            # Titolo storia
            outline_titles = [ (sec.get("title") or "").strip() for sec in outline ]
            title_prompt = build_title_prompt(persona, paper_title, outline_titles)
            title_cfg = GenerationConfig(max_new_tokens=32, do_sample=False, **base_cfg)
            title_raw = generate_once(tok, model, title_prompt, title_cfg)
            title_line = decode_clean_line(title_raw)
            if outline_titles and title_line.lower() == outline_titles[0].lower():
                title_line = f"{title_line}: An Overview"
            out_obj = {"title": title_line or None, "sections": sections_out}

            trace("story.title.generated", f"Story title proposed: '{title_line}'", title=title_line)

            rec = {
                "id": item.get("id"),
                "persona": persona,
                "paper_title": paper_title,
                "outline": outline,
                "paper_context_len": len(ctx_full),
                "generation": out_obj,
                "story_title": None
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trace("story.item.done", "Story item completed.",
                  sections=len(sections_out))

            if valid_count >= max(1, target_n - 1):
                ok += 1

    print(f"[DONE] items={n_items} ok_ratio={ok/max(1,n_items):.3f}")
    trace("story.output.saved", "Storyteller output saved.", path=args.out_jsonl, items=n_items)

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
