# two_stage_app.py — Orchestratore SPLITTER → STORYTELLER (FastAPI su VM)
# run: uvicorn two_stage_app:app --host 127.0.0.1 --port 8000 --workers 1

import os, re, json, uuid, time, tempfile, subprocess, shutil, sys
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PY = sys.executable

# =========================
# Config da ENV (override)
# =========================
API_KEY = os.getenv("API_KEY", "").strip()

SPLITTER_SCRIPT = os.getenv("SPLITTER_SCRIPT", "/docker/argese/clean_dataset/splitting/infer_splitter.py")
SPLITTER_BASE_MODEL = os.getenv("SPLITTER_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SPLITTER_ADAPTER_PATH = os.getenv("SPLITTER_ADAPTER_PATH", "out-splitter-qwen7b/checkpoint-100")
SPLITTER_MAX_NEW = int(os.getenv("SPLITTER_MAX_NEW", "768"))

STORYTELLER_SCRIPT = os.getenv("STORYTELLER_SCRIPT", "/docker/argese/clean_dataset/storytelling/infer_from_splits.py")
STORYTELLER_ADAPTER = os.getenv("STORYTELLER_ADAPTER", "../qwen32b_storyteller_lora/final_best")

# Timeout (secondi)
TIMEOUT_SPLITTER = int(os.getenv("TIMEOUT_SPLITTER", "900"))
TIMEOUT_STORY = int(os.getenv("TIMEOUT_STORY", "1800"))

# =========================
# FastAPI
# =========================
app = FastAPI(title="Two-Stage Story Orchestrator", version="1.0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True, "stage": "split+story"}

# =========================
# GPU picker
# =========================
def pick_best_gpu(min_free_gb: float = 6.0) -> Optional[Dict[str, Any]]:
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        best = None
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            free_gb = mem.free / (1024**3)
            if free_gb >= min_free_gb:
                score = free_gb - (util / 100.0) * 2.0
                if best is None or score > best[0]:
                    best = (score, i, free_gb, util)
        if best:
            return {"index": best[1], "free_gb": round(best[2], 1), "util": int(best[3])}
    except Exception:
        pass

    # fallback su nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        )
        best = None
        for line in out.strip().splitlines():
            idx_s, free_s, util_s = [x.strip() for x in line.split(",")]
            idx, free_gb, util = int(idx_s), float(free_s) / 1024.0, int(util_s)
            if free_gb >= min_free_gb:
                score = free_gb - (util / 100.0) * 2.0
                if best is None or score > best[0]:
                    best = (score, idx, free_gb, util)
        if best:
            return {"index": best[1], "free_gb": round(best[2], 1), "util": best[3]}
    except Exception:
        pass

    return None

# =========================
# Schemi I/O
# =========================
class SplitterCfg(BaseModel):
    base_model: Optional[str] = None
    adapter_path: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9

class StoryCfg(BaseModel):
    adapter: Optional[str] = None
    preset: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    target_words: Optional[int] = None
    # === NEW: retrieval ===
    retriever: Optional[str] = None            # "auto" | "emb" | "tfidf"
    retriever_model: Optional[str] = None      # es: "sentence-transformers/all-MiniLM-L6-v2"
    k: Optional[int] = None                    # top-k paragrafi
    max_ctx_chars: Optional[int] = None        # limite contesto per sezione
    seg_words: Optional[int] = None            # lunghezza segmento (paragrafo) in parole
    overlap_words: Optional[int] = None        # overlap tra segmenti

class TwoStageRequest(BaseModel):
    persona: str = Field(..., description="Persona scelta dall'utente")
    paper_title: Optional[str] = Field(None, description="Titolo se disponibile")
    markdown: str = Field(..., description="Markdown pulito dal docparse locale")
    target_sections: int = 5
    splitter: Optional[SplitterCfg] = None
    storyteller: Optional[StoryCfg] = None

class TwoStageResponse(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    outline: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    meta: Dict[str, Any]
    title: Optional[str] = None

# =========================
# Helpers
# =========================
def _require_api_key(x_api_key: Optional[str]):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(401, "Unauthorized")

def _run(cmd: list, timeout: int, cwd: Optional[str] = None, env: Optional[dict] = None):
    try:
        proc = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
        )
        return proc.stdout, proc.stderr
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"Timeout: {' '.join(cmd)}")

def _read_first_jsonl(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise HTTPException(500, f"Missing output file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    return json.loads(s)
                except Exception:
                    continue
    raise HTTPException(500, f"No valid JSON line in: {path}")

# --- JSON unwrap helpers (globali, usate in entrambi gli endpoint) ---
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    return s.strip()

def _extract_first_balanced_json(s: str) -> str:
    s = _strip_code_fences(s)
    start = s.find("{")
    if start < 0:
        return s.strip()
    depth = 0; in_str = False; esc = False
    i = start
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
                if depth == 0:
                    return s[start:i+1]
        i += 1
    return s[start:].strip()

def _sanitize_title(s: str, max_words: int = 12) -> str:
    if not s: return ""
    t = str(s).strip()
    # tieni solo la prima riga
    t = t.splitlines()[0].strip()
    # rimuovi fence/virgolette/markdown
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    t = t.strip('"').strip("'").strip("`").strip()
    # rimuovi prefissi da chat
    t = re.sub(r'^(?:Human|User|Assistant|System)\s*:\s*', '', t, flags=re.I).strip()
    # tronca dove il modello “spiega”
    t = re.split(r'\b(?:Here it is again|It is|The title you provided)\b', t, maxsplit=1, flags=re.I)[0].strip()
    # compatta spazi
    t = re.sub(r'\s{2,}', ' ', t)
    # limita numero parole
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    return t

def _maybe_unwrap_json_text(txt: str, curr_title: str) -> str:
    s = (txt or "").strip()
    if not s:
        return s
    try:
        if (s.startswith('"') and s.endswith('"')) or s.startswith('\\"') or s.startswith("\\\""):
            s = json.loads(s)  # "\"{...}\"" -> "{...}"
    except Exception:
        s = s.replace('\\"', '"').replace("\\n", "\n")
    s = _strip_code_fences(s)
    if "{" in s and "}" in s:
        jraw = _extract_first_balanced_json(s)
        try:
            obj = json.loads(jraw)
            if isinstance(obj, dict):
                if isinstance(obj.get("text"), str):
                    return obj["text"].strip()
                if isinstance(obj.get("sections"), list):
                    for it in obj["sections"]:
                        if (isinstance(it, dict) and isinstance(it.get("title"), str)
                            and it["title"].strip() == curr_title and isinstance(it.get("text"), str)):
                            return it["text"].strip()
                    for it in obj["sections"]:
                        if isinstance(it, dict) and isinstance(it.get("text"), str):
                            return it["text"].strip()
        except Exception:
            pass
    if "}{" in s:
        parts = re.split(r"\}\s*\{", s)
        for p in parts:
            cand = p
            if not cand.startswith("{"): cand = "{" + cand
            if not cand.endswith("}"):  cand = cand + "}"
            try:
                o = json.loads(cand)
                if isinstance(o, dict):
                    if isinstance(o.get("text"), str):
                        return o["text"].strip()
                    if isinstance(o.get("sections"), list):
                        for it in o["sections"]:
                            if (isinstance(it, dict) and isinstance(it.get("title"), str)
                                and it["title"].strip() == curr_title and isinstance(it.get("text"), str)):
                                return it["text"].strip()
            except Exception:
                continue
    return s

# =========================
# Endpoint principale (split + story)
# =========================
@app.post("/api/two_stage_story", response_model=TwoStageResponse)
def two_stage_story(req: TwoStageRequest, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    # Defaults + override
    split_base    = (req.splitter.base_model if (req.splitter and req.splitter.base_model) else SPLITTER_BASE_MODEL)
    split_adapter = (req.splitter.adapter_path if (req.splitter and req.splitter.adapter_path) else SPLITTER_ADAPTER_PATH)
    split_max_new = (req.splitter.max_new_tokens if (req.splitter and req.splitter.max_new_tokens is not None) else SPLITTER_MAX_NEW)
    split_temp    = (float(req.splitter.temperature) if (req.splitter and req.splitter.temperature is not None) else 0.0)
    split_top_p   = (float(req.splitter.top_p)       if (req.splitter and req.splitter.top_p       is not None) else 0.9)

    persona      = (req.persona or "General Public").strip()
    paper_title  = (req.paper_title or "").strip()
    markdown     = req.markdown
    target_sections = max(1, int(req.target_sections or 5))

    story_adapter = (req.storyteller.adapter if (req.storyteller and req.storyteller.adapter) else STORYTELLER_ADAPTER)
    st_preset     = (req.storyteller.preset if (req.storyteller and req.storyteller.preset) else "medium").lower()
    st_temp       = float(req.storyteller.temperature) if (req.storyteller and req.storyteller.temperature is not None) else 0.0
    st_top_p      = float(req.storyteller.top_p) if (req.storyteller and req.storyteller.top_p is not None) else 0.9
    st_max_new    = int(req.storyteller.max_new_tokens) if (req.storyteller and req.storyteller.max_new_tokens is not None) else 1500
    st_min_new    = int(req.storyteller.min_new_tokens) if (req.storyteller and req.storyteller.min_new_tokens is not None) else 600
    st_target_w   = int(req.storyteller.target_words) if (req.storyteller and req.storyteller.target_words is not None) else None

    # === NEW: defaults retrieval storyteller ===
    st_retriever       = (req.storyteller.retriever       if (req.storyteller and req.storyteller.retriever       is not None) else "auto")
    st_retriever_model = (req.storyteller.retriever_model if (req.storyteller and req.storyteller.retriever_model is not None) else "sentence-transformers/all-MiniLM-L6-v2")
    st_k               = (int(req.storyteller.k)               if (req.storyteller and req.storyteller.k               is not None) else 6)
    st_max_ctx_chars   = (int(req.storyteller.max_ctx_chars)   if (req.storyteller and req.storyteller.max_ctx_chars   is not None) else 2500)
    st_seg_words       = (int(req.storyteller.seg_words)       if (req.storyteller and req.storyteller.seg_words       is not None) else 180)
    st_overlap_words   = (int(req.storyteller.overlap_words)   if (req.storyteller and req.storyteller.overlap_words   is not None) else 60)

    # Temp workspace
    rid = str(uuid.uuid4())
    workdir = tempfile.mkdtemp(prefix=f"twostage_{rid}_")
    in_split = os.path.join(workdir, "in_splitter.jsonl")
    out_split = os.path.join(workdir, "out_splitter.jsonl")
    out_story = os.path.join(workdir, "out_story.jsonl")

    timings = {}
    try:
        # 1) input per SPLITTER
        record = {"id": rid, "persona": persona, "md": markdown, "title": paper_title or None}
        with open(in_split, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # GPU preferita
        chosen = pick_best_gpu(min_free_gb=6.0)
        cvd = str(chosen["index"]) if chosen else None

        base_env = dict(os.environ)
        if cvd is not None:
            base_env["CUDA_VISIBLE_DEVICES"] = cvd
        base_env.setdefault("HF_HOME", "/docker/argese/clean_dataset/hf")
        base_env.setdefault("OFFLOAD_FOLDER", "/docker/argese/offload")
        base_env.setdefault("GPU_MEM_GB", "40")
        base_env.setdefault("CPU_MEM_GB", "128")

        # 2) SPLITTER
        t0 = time.time()
        cmd_split = [
            PY, SPLITTER_SCRIPT,
            "--base_model", split_base,
            "--adapter_path", split_adapter,
            "--input_path", in_split,
            "--output_jsonl", out_split,
            "--sections", str(target_sections),
            "--max_new_tokens", str(split_max_new),
            "--temperature", str(split_temp),
            "--top_p", str(split_top_p),
        ]
        _run(cmd_split, timeout=TIMEOUT_SPLITTER, cwd=None, env=base_env)
        timings["splitter_s"] = round(time.time() - t0, 3)

        # 3) STORYTELLER
        t1 = time.time()
        cmd_story = [
            PY, STORYTELLER_SCRIPT,
            "--in_jsonl", out_split,
            "--out_jsonl", out_story,
            "--adapter", story_adapter,
            "--preset", st_preset,
            "--temperature", str(st_temp),
            "--top_p", str(st_top_p),
            "--max_new_tokens", str(st_max_new),
            "--min_new_tokens", str(st_min_new),
            # === NEW: retrieval flags ===
            "--retriever", st_retriever,
            "--retriever_model", st_retriever_model,
            "--k", str(st_k),
            "--max_ctx_chars", str(st_max_ctx_chars),
            "--seg_words", str(st_seg_words),
            "--overlap_words", str(st_overlap_words),
        ]
        _run(cmd_story, timeout=TIMEOUT_STORY, cwd=None, env=base_env)
        timings["storyteller_s"] = round(time.time() - t1, 3)

        # 4) Leggi risultati
        split_obj = _read_first_jsonl(out_split)
        story_obj = _read_first_jsonl(out_story)

        outline = split_obj.get("sections", []) or []
        gen = story_obj.get("generation", {}) or {}
        sections = gen.get("sections", []) or gen.get("Sections", []) or []

        # Normalizza sezioni
        norm_sections = []
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                continue
            title = sec.get("title") or f"Section {i+1}"
            text  = sec.get("text") or sec.get("narrative") or ""
            text = _maybe_unwrap_json_text(text, title)
            if not str(text).strip():
                try:
                    desc = None
                    if isinstance(split_obj.get("sections"), list) and i < len(split_obj["sections"]):
                        desc = split_obj["sections"][i].get("description")
                    text = (desc or "").strip()
                except Exception:
                    text = ""
            norm_sections.append({"title": title, "text": text})
        sections = norm_sections

        # Titolo
        story_title = gen.get("title") if isinstance(gen, dict) else None
        story_title = _sanitize_title(story_title) if story_title else None
        if not story_title:
            if sections and isinstance(sections[0], dict):
                story_title = sections[0].get("title")
        if not story_title:
            base = split_obj.get("paper_title") or paper_title or "Untitled"
            story_title = f"{base} — {persona}"


        # Fallback duro se vuoto
        if not isinstance(sections, list) or len(sections) == 0:
            outline = split_obj.get("sections", []) or []
            sections = [{
                "title": s.get("title", f"Section {i+1}"),
                "text":  s.get("description", "")
            } for i, s in enumerate(outline[:5])]

        resp = {
            "persona": persona,
            "paper_title": split_obj.get("paper_title") or paper_title or None,
            "title": story_title,
            "outline": outline,
            "sections": sections,
            "meta": {
                "req_id": rid,
                "timings": {**timings, "total_s": round((time.time() - t0), 3)},
                "paths": {"in_splitter": in_split, "out_splitter": out_split, "out_story": out_story},
                "title_source": (
                    "storyteller.title" if gen.get("title") else
                    "sections[0].title" if (sections and sections[0].get('title')) else
                    "fallback(persona+paper_title)"
                ),
                "splitter_params": {
                    "base_model": split_base,
                    "adapter_path": split_adapter,
                    "max_new_tokens": split_max_new,
                    "temperature": split_temp,
                    "top_p": split_top_p,
                    "target_sections": target_sections,
                },
                "storyteller_params": {
                    "adapter": story_adapter,
                    "preset": st_preset,
                    "temperature": st_temp,
                    "top_p": st_top_p,
                    "max_new_tokens": st_max_new,
                    "min_new_tokens": st_min_new,
                    "target_words": st_target_w,
                    "retriever": st_retriever,
                    "retriever_model": st_retriever_model,
                    "k": st_k,
                    "max_ctx_chars": st_max_ctx_chars,
                    "seg_words": st_seg_words,
                    "overlap_words": st_overlap_words,
                }
            }
        }
        return resp

    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


# =========================
# Rigenerazione da outline (salta lo splitter)
# =========================
class TwoStageFromOutlineReq(BaseModel):
    persona: str
    paper_title: str | None = None
    cleaned_text: str                       # markdown pulito
    outline: List[Dict[str, Any]]           # [{title, description?}]
    storyteller: Optional[StoryCfg] = None

@app.post("/api/two_stage_story_from_outline")
def two_stage_story_from_outline(req: TwoStageFromOutlineReq, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    persona     = (req.persona or "General Public").strip()
    paper_title = (req.paper_title or "Paper").strip()
    cleaned     = req.cleaned_text or ""
    outline     = req.outline or []

    # storyteller params (stessi default dell’altro endpoint)
    st = req.storyteller or StoryCfg()
    story_adapter = st.adapter or STORYTELLER_ADAPTER
    st_preset   = (st.preset or "medium").lower()
    st_temp     = float(st.temperature) if st.temperature is not None else 0.0
    st_top_p    = float(st.top_p) if st.top_p is not None else 0.9
    st_max_new  = int(st.max_new_tokens) if st.max_new_tokens is not None else 1500
    st_min_new  = int(st.min_new_tokens) if st.min_new_tokens is not None else 600

    # retrieval knobs (default robusti)
    st_retriever       = (getattr(st, "retriever", None) or "auto")
    st_retriever_model = (getattr(st, "retriever_model", None) or "sentence-transformers/all-MiniLM-L6-v2")
    st_k               = int(getattr(st, "k", 6) or 6)
    st_max_ctx_chars   = int(getattr(st, "max_ctx_chars", 2500) or 2500)
    st_seg_words       = int(getattr(st, "seg_words", 180) or 180)
    st_overlap_words   = int(getattr(st, "overlap_words", 60) or 60)

    # workspace
    rid = str(uuid.uuid4())
    workdir = tempfile.mkdtemp(prefix=f"twostage2_{rid}_")
    in_story = os.path.join(workdir, "in_story.jsonl")
    out_story = os.path.join(workdir, "out_story.jsonl")
    
    def _filter_unsupported(text: str, source: str) -> str:
        src = source.lower()
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        kept = []
        for s in sents:
            toks = re.findall(r"\b[A-Z][a-zA-Z\-]{2,}\b", s)  # parole Capitalizzate (tipo Stanford)
            # se appare un nome proprio NON presente nella sorgente → scarta la frase
            if any(t.lower() not in src for t in toks):
                continue
            kept.append(s)
        return " ".join(kept).strip()


    try:
        # prepara JSONL per storyteller (outline fissato)
        record = {
            "id": rid,
            "persona": persona,
            "paper_title": paper_title,
            "cleaned_text": cleaned,
            "sections": outline,
        }
        with open(in_story, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # GPU
        chosen = pick_best_gpu(min_free_gb=6.0)
        base_env = dict(os.environ)
        if chosen:
            base_env["CUDA_VISIBLE_DEVICES"] = str(chosen["index"])
        base_env.setdefault("HF_HOME", "/docker/argese/clean_dataset/hf")
        base_env.setdefault("OFFLOAD_FOLDER", "/docker/argese/offload")

        # STORYTELLER (solo)
        cmd_story = [
            PY, STORYTELLER_SCRIPT,
            "--in_jsonl", in_story,
            "--out_jsonl", out_story,
            "--adapter", story_adapter,
            "--preset", st_preset,
            "--temperature", str(st_temp),
            "--top_p", str(st_top_p),
            "--max_new_tokens", str(st_max_new),
            "--min_new_tokens", str(st_min_new),
            "--retriever", st_retriever,
            "--retriever_model", st_retriever_model,
            "--k", str(st_k),
            "--max_ctx_chars", str(st_max_ctx_chars),
            "--seg_words", str(st_seg_words),
            "--overlap_words", str(st_overlap_words),
        ]
        _run(cmd_story, timeout=TIMEOUT_STORY, env=base_env)

        story_obj = _read_first_jsonl(out_story)
        gen = story_obj.get("generation", {}) or {}
        sections = gen.get("sections", []) or []

        # normalizza
        norm = []
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                continue
            title = sec.get("title") or (outline[i]["title"] if i < len(outline) else f"Section {i+1}")
            raw   = sec.get("text") or sec.get("narrative") or ""
            text = (
                _maybe_unwrap_json_text(raw, title)
                or (outline[i].get("description", "") if i < len(outline) else raw)
            )
            text = _maybe_unwrap_json_text(text, title)
            if cleaned:
                text = _filter_unsupported(text, cleaned)
            norm.append({"title": title, "text": text})

        ret_title = _sanitize_title(gen.get("title")) if isinstance(gen, dict) else None

        return {
            "persona": persona,
            "paper_title": paper_title,
            "title": paper_title,  # 🔒 blocca il titolo alla prima storia
            "outline": outline,
            "sections": norm,
            "meta": {
                "req_id": rid,
                "title_locked": True,
                "aiTitle": ret_title,                # opzionale: log titolo proposto dal modello
                "title_source": "paper_title_locked",
                "storyteller_params": {
                    "preset": st_preset, "temperature": st_temp, "top_p": st_top_p,
                    "max_new_tokens": st_max_new, "min_new_tokens": st_min_new,
                    "retriever": st_retriever, "retriever_model": st_retriever_model,
                    "k": st_k, "max_ctx_chars": st_max_ctx_chars,
                    "seg_words": st_seg_words, "overlap_words": st_overlap_words,
                },
            },
        }

    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
