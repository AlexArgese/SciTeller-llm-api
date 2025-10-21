#!/usr/bin/env python3
# infer_from_splits.py — storyteller per-sezione con RETRIEVAL (semantico o TF-IDF)
import os, re, json, argparse, math
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

# ===============================
# CONFIG
# ===============================
BASE_MODEL  = "Qwen/Qwen2.5-32B-Instruct"
ADAPTER_DIR = "qwen32b_storyteller_lora/final_best"

STRICT_RULES_TEMPLATE = (
    "You are an AI Scientist Storyteller.\n"
    "Persona: {persona}\n"
    "Paper title: {paper_title}\n\n"
    "Rules (STRICT):\n"
    "- Output ONLY a single JSON object, no markdown, no extra text.\n"
    "- Schema:\n"
    "  {{\n"
    "    \"title\": \"<concise, persona-tailored story title (<= 10 words)>\",\n"
    "    \"sections\": [\n"
    "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
    "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
    "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
    "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
    "       {{\"title\": \"...\",\"text\": \"...\"}}\n"
    "    ]\n"
    "  }}\n"
    "- The title must be catchy and informative for the Persona, not just the paper title.\n"
    "- Prefer <= 10 words; avoid leading 'Introduction to ...'.\n"
    "- Exactly 5 sections and the same titles as in the target outline, in order.\n"
    "- Be faithful to the facts in the paper context; NO fabrication.\n"
    "- Use ONLY information provided in the paper context; do not invent citations, names, affiliations and numbers.\n"
    "- Tone/complexity MUST fit the Persona.\n"
    "- Clear prose in paragraphs; avoid bullet lists.\n"
    "{length_rule}\n"
    "- Do NOT copy long passages verbatim; paraphrase faithfully.\n"
    "- No prefaces or explanations; return only the JSON object.\n"
    "- Use ONLY English.\n"
    "- IMPORTANT: Never nest the section object under a key named after the section (e.g., no {{\\\"Background\\\": {{...}}}}).\n"
)

BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

def length_rule_from_preset(preset: str) -> str:
    p = (preset or "medium").lower()
    if p == "short": return "- Keep each section short: aim for ~2 short paragraphs."
    if p == "long":  return "- Write longer sections: aim for ~3–4 paragraphs each."
    return "- Keep sections concise: ~2–3 short paragraphs each."

# ===============================
# JSON / text helpers
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

def _clean_plain_text(txt: str) -> str:
    # pulizia leggera finale per prosa: niente fence, niente doppi apici isolati, compattazione newline
    t = _strip_code_fences(str(txt))
    t = t.strip().strip('"').strip("'")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

# ===============================
# Anti-hallucination helpers
# ===============================
def cap_entities(text:str):
    return set(re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b", text))

def too_many_new_entities(gen_text:str, ctx:str, max_new_ratio=0.05, min_total=1):
    ctx_cap = cap_entities(ctx)
    out_cap = cap_entities(gen_text)
    if not out_cap: return False
    new_ents = [e for e in out_cap if e not in ctx_cap and e.lower() not in {"json","section","introduction","conclusion"}]
    return (len(out_cap) >= min_total) and (len(new_ents)/max(1,len(out_cap)) > max_new_ratio)

def reinforce_no_hallu(prompt:str, chunk_note:str="context chunk"):
    extra = (
        "\n\nSTRICT REMINDER:\n"
        f"- DO NOT invent names/affiliations.\n"
        f"- If authors are anonymized, say 'the authors' / 'the framework'.\n"
        f"- Use only entities explicitly present in the {chunk_note}.\n"
        "- If the context does not include an information, write 'not specified in the paper'."
    )
    return prompt + extra

def _strip_unknown_entities(txt, ctx):
    wl = cap_entities(ctx)
    sents = re.split(r'(?<=[.!?])\s+', txt)
    keep = []
    for s in sents:
        ents = cap_entities(s)
        if not ents or all((e in wl) or (e.lower() in {"json","section"}) for e in ents):
            keep.append(s)
    return " ".join(keep).strip()

# ===============================
# Retrieval: segmentation + embeddings + top-k
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
    Usa sentence-transformers (se disponibile), altrimenti TF-IDF con cosine.
    """
    def __init__(self, method:str="auto", model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        self.method = method
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
                if method == "emb":
                    raise
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
        if not self.corpus or self.corpus_mat is None: return []
        if self.method == "emb":
            q = self.st_model.encode([query], normalize_embeddings=True)[0]
            sims = (self.corpus_mat @ q)  # cosine già normalizzato
            idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in idx]
        else:
            qv = self.vec.transform([query])
            sims = (qv @ self.corpus_mat.T).A[0]
            idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in idx]

def build_section_context(sec_title:str, sec_desc:str, ret:Retriever, k:int=6, max_chars:int=2500) -> str:
    query = (sec_title.strip() + " — " + (sec_desc or "")).strip()
    hits = ret.topk(query, k=max(1,k))
    ctx = ""
    for i, _score in hits:
        frag = ret.corpus[i].strip()
        if not frag: continue
        if len(ctx) + len(frag) + 2 > max_chars: break
        ctx += frag + "\n\n"
    return ctx.strip()

# ===============================
# Prompt builders
# ===============================
def build_section_prompt(persona:str, paper_title:str, target_title:str, target_desc:str, context_chunk:str, length_preset:str="medium") -> str:
    rules_full = STRICT_RULES_TEMPLATE.format(
        persona=persona, paper_title=paper_title, length_rule=length_rule_from_preset(length_preset),
    )
    rules_single = rules_full.replace(
        "- Schema:\n"
        "  {{\n"
        "    \"title\": \"<concise, persona-tailored story title (<= 10 words)>\",\n"
        "    \"sections\": [\n"
        "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
        "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
        "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
        "       {{\"title\": \"...\",\"text\": \"...\"}},\n"
        "       {{\"title\": \"...\",\"text\": \"...\"}}\n"
        "    ]\n"
        "  }}",
        "- Schema (SINGLE section):\n"
        "  {{\n"
        "    \"title\": \"" + target_title.replace('\"','\\\"') + "\",\n"
        "    \"text\": \"<2–4 short paragraphs separated by a single blank line>\"\n"
        "  }}\n"
        "- Use EXACTLY the provided section title above (do NOT change it)."
    )
    rules_single = re.sub(
        r"^- Exactly 5 sections.*\n",
        "- Generate exactly ONE section object matching the provided title.\n",
        rules_single, flags=re.M,
    )
    rules_single += "\n- Do NOT nest the section under a key named after the section (e.g., no {\"%s\": {...}}). Return ONLY {\"title\":\"%s\",\"text\":\"...\"}.\n" % (target_title.replace('"','\\"'), target_title.replace('"','\\"'))
    rules_single += "\n- IMPORTANT: Write in English only, never use Chinese or other characters.\n"
    parts = [
        rules_single,
        "\nTarget section:\n- Title: " + target_title
    ]
    if target_desc:
        parts.append("- Description: " + target_desc)
    parts.append("\nPaper context (retrieved for this section):\n" + (context_chunk or "[NO CONTEXT FOUND]"))
    parts.append("\nReturn ONLY the JSON object for this single section.")
    parts.append(
        "\nSTRICT ENTITY RULES:\n"
        "- Mention organizations/people ONLY if they appear verbatim in the retrieved context above.\n"
        "- If not present, use neutral phrases like 'the authors' or 'the framework'.\n"
        "- If some information is missing from the context, write 'not specified in the paper'."
    )
    parts.append("\nLanguage requirement: Write in English only. Never use Chinese/Japanese/Korean characters.")
    return "\n".join(parts)

def build_title_prompt(persona:str, paper_title:str, outline_titles:List[str], length_preset:str="medium") -> str:
    avoid = "; ".join(t for t in outline_titles if t)
    return (
        "You are an AI Scientist Storyteller.\n"
        f"Persona: {persona}\n"
        f"Paper title: {paper_title}\n\n"
        "Task: Propose ONE catchy, persona-tailored story title (<= 10 words).\n"
        "Constraints:\n"
        "- Do NOT reuse any section title verbatim.\n"
        "- No quotes, no punctuation-only lines, no markdown.\n"
        "- English only.\n"
        f"- Avoid these exact strings: {avoid}\n"
        "Return ONLY the title text.\n"
    )

# ===============================
# Model load
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
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=(torch.bfloat16 if BF16 else torch.float16),
        quantization_config=bnb,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tok, model

# ===============================
# Inference
# ===============================
@torch.no_grad()
def generate_once(tokenizer, model, prompt: str, gen_cfg: GenerationConfig) -> str:
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, generation_config=gen_cfg)
    gen_ids = out[0][inp["input_ids"].shape[1]:]
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
    ap.add_argument("--min_new_tokens", type=int, default=360)
    # retrieval
    ap.add_argument("--retriever", default="auto", help="auto|emb|tfidf")
    ap.add_argument("--retriever_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=3, help="top-k paragrafi per sezione (default 3 per meno rumore)")
    ap.add_argument("--max_ctx_chars", type=int, default=1400)
    ap.add_argument("--seg_words", type=int, default=180)
    ap.add_argument("--overlap_words", type=int, default=60)
    args = ap.parse_args()

    # modello
    tok, model = load_model_and_tokenizer(args.adapter)

    # clamp anti-hallucination (utile anche se chiamato standalone)
    temp  = float(args.temperature)
    top_p = float(args.top_p)

    # config generazione
    do_sample = temp > 0.0
    base_cfg = dict(
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

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
            ctx_full = item.get("cleaned_text") or ""

            custom_prompt = item.get("prompt")
            if custom_prompt:
                # bypass: prompt diretto (caso regen_paragraph_vm)
                gen_full = generate_once(tok, model, custom_prompt, gen_cfg)
                gen_json_text = extract_first_json_object(gen_full)
                fout.write(json.dumps({
                    "id": item.get("id"),
                    "persona": persona,
                    "generation": {"text": gen_json_text}
                }, ensure_ascii=False) + "\n")
                continue


            # --- Build retrieval corpus una volta per paper ---
            paragraphs = segment_text(ctx_full, max_words=args.seg_words, overlap=args.overlap_words)
            ret = Retriever(method=args.retriever, model_name=args.retriever_model)
            ret.fit(paragraphs)

            # numero sezioni target
            SEC_COUNT = 5
            target_n = len(outline) if isinstance(outline, list) and len(outline)>0 else SEC_COUNT

            # budget per sezione
            per_sec_max = max(220, int(args.max_new_tokens // max(1, target_n)))
            # preset-aware minimums
            preset = (args.preset or "medium").lower()
            if preset == "short":
                per_sec_min = 0
            elif preset == "long":
                per_sec_min = max(120,  int(args.min_new_tokens // max(1, target_n)))
            else:
                per_sec_min = max(60,   int(args.min_new_tokens // max(1, target_n)))

            if do_sample:
                gen_cfg = GenerationConfig(
                    max_new_tokens=per_sec_max,
                    min_new_tokens=per_sec_min if per_sec_min > 0 else None,
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=1.10,
                    **base_cfg,
                )
            else:
                gen_cfg = GenerationConfig(
                    max_new_tokens=per_sec_max,
                    min_new_tokens=per_sec_min if per_sec_min > 0 else None,
                    do_sample=False,
                    **base_cfg,
                )

            sections_out, valid_count = [], 0

            for i in range(target_n):
                sec = outline[i] if i < len(outline) else {}
                sec_title = (sec.get("title") or f"Section {i+1}").strip()
                sec_desc  = (sec.get("description") or "").strip()

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
                )

                tries = 0
                while True:
                    gen_full = generate_once(tok, model, prompt, gen_cfg)
                    gen_json_text = extract_first_json_object(gen_full)
                    if not too_many_new_entities(gen_json_text, ctx_i, max_new_ratio=0.05, min_total=1):
                        break
                    tries += 1
                    if tries > 3:
                        gen_json_text = _strip_unknown_entities(gen_json_text, ctx_i)
                        break
                    prompt = reinforce_no_hallu(prompt, chunk_note="retrieved context")

                out_sec = {"title": sec_title, "text": ""}

                try:
                    obj = json.loads(gen_json_text)
                    if isinstance(obj, dict):
                        # 1) caso corretto: {"title": "...", "text": "..."}
                        if isinstance(obj.get("text"), str):
                            out_sec["text"] = obj["text"].strip()

                        # 2) caso array sezioni: {"sections": [{title,text}, ...]}
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

                        # 3) caso annidato con singola chiave: {"Background": {"title":"Background","text":"..."}}
                        if not out_sec["text"]:
                            if len(obj) == 1:
                                only_v = next(iter(obj.values()))
                                if isinstance(only_v, dict) and isinstance(only_v.get("text"), str):
                                    out_sec["text"] = only_v["text"].strip()

                        # 4) se 'text' è a sua volta un JSON string -> unwrap annidato
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

                        # 5) hard-unwrap finale: se restano graffe/JSON-like nel text, ripulisci
                        if out_sec["text"] and "{" in out_sec["text"] and "}" in out_sec["text"] and ":" in out_sec["text"]:
                            # prova ancora un parse
                            try:
                                inner2 = json.loads(out_sec["text"])
                                if isinstance(inner2, dict) and isinstance(inner2.get("text"), str):
                                    out_sec["text"] = inner2["text"].strip()
                            except Exception:
                                # fallback: rimuovi delimitatori lasciando plain text
                                out_sec["text"] = re.sub(r"[{}\[\]\"]", "", out_sec["text"]).strip()

                except Exception:
                    raw = _strip_code_fences(gen_full).strip()
                    paras = re.split(r"\n\s*\n", raw)
                    paras = [p.strip() for p in paras if p.strip()]
                    out_sec["text"] = "\n\n".join(paras[:4]) if paras else raw[:1200]

                # seed se ancora vuoto
                if not out_sec["text"].strip():
                    seed = sec_desc or ctx_i[:600].strip() or ""
                    out_sec["text"] = seed

                # pulizia finale prosa
                out_sec["text"] = _clean_plain_text(out_sec["text"])
                out_sec["text"] = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', out_sec["text"]).strip()
                paras = [p.strip() for p in re.split(r"\n{2,}", out_sec["text"]) if p.strip()]
                if not paras:
                    paras = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])', out_sec["text"])
                out_sec["text"] = "\n\n".join([p for p in paras if p])


                if out_sec["text"].strip(): valid_count += 1
                sections_out.append(out_sec)

            outline_titles = [ (sec.get("title") or "").strip() for sec in outline ]
            title_prompt = build_title_prompt(persona, paper_title, outline_titles, args.preset)
            title_cfg = GenerationConfig(max_new_tokens=32, **base_cfg)
            title_raw = generate_once(tok, model, title_prompt, title_cfg).strip()
            title_line = title_raw.splitlines()[0].strip().strip('"').strip("'")
            # hard clean
            title_line = re.sub(r'^(Human|User|Assistant|System)\s*:\s*', '', title_line, flags=re.I)
            title_line = re.sub(r'["`]+.*$', '', title_line).strip()  
            title_line = re.sub(r'\s{2,}', ' ', title_line).strip()
            title_line = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', title_line).strip()

            if outline_titles and title_line.lower() == outline_titles[0].lower():
                title_line = f"{title_line}: An Overview"

            out_obj = {"title": title_line or None, "sections": sections_out}

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

            if valid_count >= max(1, target_n - 1):
                ok += 1

    print(f"[DONE] items={n_items}  ok(items with most sections filled)={ok/max(1,n_items):.3f}")

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
