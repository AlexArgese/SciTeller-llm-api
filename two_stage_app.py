# FILE: two_stage_app.py — Orchestratore SPLITTER → STORYTELLER (FastAPI su VM)
# run: uvicorn two_stage_app:app --host 127.0.0.1 --port 8000 --workers 1

import os, re, json, uuid, time, tempfile, subprocess, shutil, sys, fcntl
from typing import Optional, Dict, Any, List, Tuple, Literal
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from datetime import datetime
LOG_ROOT = os.getenv("SCI_LOG_ROOT", "/tmp/sci_logs")
os.makedirs(LOG_ROOT, exist_ok=True)

def _ts():
    return datetime.utcnow().isoformat(timespec="milliseconds")+"Z"

def _trace_open(req_id: str):
    path = os.path.join(LOG_ROOT, f"trace_{req_id}.ndjson")
    def _write(event: str, message: str = "", **data):
        rec = {"ts": _ts(), "req_id": req_id, "event": event, "message": message, **data}
        line = json.dumps(rec, ensure_ascii=False)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        # echo su console per visibilità immediata
        try:
            print(f"[trace] {event} {message} :: {json.dumps(data, ensure_ascii=False)}", file=sys.stderr, flush=True)
        except Exception:
            pass
    return path, _write


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
app = FastAPI(title="Two-Stage Story Orchestrator", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True, "stage": "split+story", "v": "1.1.0"}

# =========================
# GPU picker + locking
# =========================
def pick_best_gpu(min_free_gb: float = 6.0) -> Optional[Dict[str, Any]]:
    """Sceglie la GPU più libera (pynvml o nvidia-smi). Non prenota."""
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

class GPULock:
    """File-based lock per una GPU specifica: /tmp/gpu_{idx}.lock"""
    def __init__(self, idx: int, timeout_s: int = 10):
        self.idx = idx
        self.timeout_s = timeout_s
        self.fd = None
        self.path = f"/tmp/gpu_{idx}.lock"

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        start = time.time()
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(self.fd, 0)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except BlockingIOError:
                if time.time() - start > self.timeout_s:
                    raise HTTPException(503, f"GPU {self.idx} busy (lock timeout)")
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
        finally:
            self.fd = None

def build_env_with_gpu(min_free_gb: float = 6.0) -> Tuple[Dict[str, str], Optional[GPULock], Optional[int]]:
    """Sceglie una GPU, la blocca e costruisce env base. Ritorna (env, lock, idx)."""
    chosen = pick_best_gpu(min_free_gb=min_free_gb)
    lock = None
    env = dict(os.environ)
    if chosen:
        lock = GPULock(chosen["index"])
        lock.__enter__()  # acquisisci subito (uscirà nel finally del chiamante)
        env["CUDA_VISIBLE_DEVICES"] = str(chosen["index"])
    env.setdefault("HF_HOME", "/docker/argese/clean_dataset/hf")
    env.setdefault("OFFLOAD_FOLDER", "/docker/argese/offload")
    env.setdefault("GPU_MEM_GB", "40")
    env.setdefault("CPU_MEM_GB", "128")
    return env, lock, (chosen["index"] if chosen else None)

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
    length_preset: Optional[str] = Field(default=None, alias="preset")
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    # retrieval
    retriever: Optional[str] = None
    retriever_model: Optional[str] = None
    k: Optional[int] = None
    max_ctx_chars: Optional[int] = None
    seg_words: Optional[int] = None
    overlap_words: Optional[int] = None

class TwoStageRequest(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    markdown: str
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

class OneSectionReq(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    cleaned_text: str
    section: Dict[str, Any]
    storyteller: Optional[StoryCfg] = None

class OneSectionResp(BaseModel):
    title: str
    text: str
    paragraphs: List[str]

class RegenSectionsVMReq(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    cleaned_text: str
    outline: List[Dict[str, Any]]
    targets: List[int]
    storyteller: Optional[StoryCfg] = None

class RegenSectionsVMResp(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    sections: Dict[str, Dict[str, Any]]
    meta: Dict[str, Any]

# =========================
# Helpers comuni
# =========================
def _require_api_key(x_api_key: Optional[str]):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(401, "Unauthorized")

def _run(cmd: list, timeout: int, cwd: Optional[str] = None, env: Optional[dict] = None, trace=None, stage:str="proc"):
    if trace:
        trace("proc.start", f"Starting {stage} process…", stage=stage, cmd=" ".join(cmd), timeout_s=timeout, cwd=cwd)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
        dt = round(time.time() - t0, 3)
        if trace:
            trace("proc.ok", f"{stage} process finished.", stage=stage, elapsed_s=dt,
                  stdout_tail=(proc.stdout[-4000:] if proc.stdout else ""),
                  stderr_tail=(proc.stderr[-4000:] if proc.stderr else ""))
        return proc.stdout, proc.stderr
    except subprocess.CalledProcessError as e:
        dt = round(time.time() - t0, 3)
        if trace:
            trace("proc.fail", f"{stage} failed.", stage=stage, elapsed_s=dt, returncode=e.returncode,
                  stdout_tail=(e.stdout[-4000:] if e.stdout else ""), stderr_tail=(e.stderr[-4000:] if e.stderr else ""))
        raise HTTPException(500, f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    except subprocess.TimeoutExpired:
        if trace:
            trace("proc.timeout", f"{stage} timed out.", stage=stage, elapsed_s=round(time.time()-t0,3))
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
    t = t.splitlines()[0].strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    t = t.strip('"').strip("'").strip("`").strip()
    t = re.sub(r'^(?:Human|User|Assistant|System)\s*:\s*', '', t, flags=re.I).strip()
    t = re.split(r'\b(?:Here it is again|It is|The title you provided)\b', t, maxsplit=1, flags=re.I)[0].strip()
    t = re.sub(r'\s{2,}', ' ', t)
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
            s = json.loads(s)
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

def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not parts:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])', text)
    return [p.strip() for p in parts if p.strip()]

# ===== Persona guidance (per prompt paragrafo) =====
def _persona_guidance(persona: str) -> str:
    p = (persona or "").lower()
    if any(x in p for x in ["general", "public", "journalist"]):
        return ("Use accessible language and add 1–2 short, concrete examples or analogies "
                "to make abstract ideas tangible for non-experts.")
    if "student" in p:
        return ("Explain with short didactic sentences. Define key terms briefly and include 1 intuitive example.")
    if any(x in p for x in ["teacher", "clinician", "product manager", "investor"]):
        return ("Highlight implications and practical takeaways. Prefer clear, concrete examples over theory.")
    # researcher/engineer/reviewer
    return ("Keep technical precision. Examples are optional; focus on faithful paraphrase and clarity.")

# ===== Retrieval leggero (solo per regen_paragraph_vm) =====
def _tfidf_topk_fragments(cleaned_text: str, query: str, max_words=180, overlap=60, k=3, max_chars=1200) -> List[str]:
    """Segmenta il testo e recupera i top-k frammenti simili al paragrafo target. TF-IDF se disponibile, altrimenti fallback semplice."""
    words = (cleaned_text or "").split()
    if not words:
        return []
    segs, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        seg = " ".join(words[i:i+max_words]).strip()
        if seg:
            segs.append(seg)
        i += step
    if not segs:
        return []

    try:
        # TF-IDF cosine
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        vec = TfidfVectorizer(ngram_range=(1,2), max_features=80000, lowercase=True)
        X = vec.fit_transform(segs)
        q = vec.transform([query])
        sims = linear_kernel(q, X).ravel()
        idx = sims.argsort()[::-1][:max(1,k)]
    except Exception:
        # Fallback: rank by shared tokens count
        qtoks = set(re.findall(r"[a-zA-Z0-9]{3,}", query.lower()))
        scores = []
        for j, s in enumerate(segs):
            stoks = set(re.findall(r"[a-zA-Z0-9]{3,}", s.lower()))
            scores.append((j, len(qtoks & stoks)))
        idx = [j for j,_ in sorted(scores, key=lambda x: x[1], reverse=True)[:max(1,k)]]

    out = []
    budget = 0
    for j in idx:
        frag = segs[j].strip()
        if not frag:
            continue
        if budget + len(frag) + 2 > max_chars:
            break
        out.append(frag)
        budget += len(frag) + 2
    return out

# ===== Wrapper unificati per subprocess =====
def _build_story_env_and_lock(min_free_gb: float = 6.0) -> Tuple[Dict[str, str], Optional[GPULock], Optional[int]]:
    return build_env_with_gpu(min_free_gb=min_free_gb)

def _run_splitter(in_split: str, out_split: str, cfg: Dict[str, Any], timeout: int, env: Dict[str, str]):
    cmd = [
        PY, SPLITTER_SCRIPT,
        "--base_model", cfg["base_model"],
        "--adapter_path", cfg["adapter_path"],
        "--input_path", in_split,
        "--output_jsonl", out_split,
        "--sections", str(cfg["sections"]),
        "--max_new_tokens", str(cfg["max_new_tokens"]),
        "--temperature", str(cfg["temperature"]),
        "--top_p", str(cfg["top_p"]),
    ]
    _run(cmd, timeout=timeout, env=env)

def _run_storyteller(in_story: str, out_story: str, cfg: Dict[str, Any], timeout: int, env: Dict[str, str]):
    cmd = [
        PY, STORYTELLER_SCRIPT,
        "--in_jsonl", in_story,
        "--out_jsonl", out_story,
        "--adapter", cfg["adapter"],
        "--preset", cfg["preset"],
        "--temperature", str(cfg["temperature"]),
        "--top_p", str(cfg["top_p"]),
        "--max_new_tokens", str(cfg["max_new_tokens"]),
        "--retriever", cfg["retriever"],
        "--retriever_model", cfg["retriever_model"],
        "--k", str(cfg["k"]),
        "--max_ctx_chars", str(cfg["max_ctx_chars"]),
        "--seg_words", str(cfg["seg_words"]),
        "--overlap_words", str(cfg["overlap_words"]),
    ]
    _run(cmd, timeout=timeout, env=env)

# =========================
# Endpoint principale (split + story)
# =========================
@app.post("/api/two_stage_story", response_model=TwoStageResponse)
def two_stage_story(req: TwoStageRequest, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    # Defaults + override
    t_start = time.time()
    split_base    = (req.splitter.base_model if (req.splitter and req.splitter.base_model) else SPLITTER_BASE_MODEL)
    split_adapter = (req.splitter.adapter_path if (req.splitter and req.splitter.adapter_path) else SPLITTER_ADAPTER_PATH)
    split_max_new = (req.splitter.max_new_tokens if (req.splitter and req.splitter.max_new_tokens is not None) else SPLITTER_MAX_NEW)
    split_temp    = (float(req.splitter.temperature) if (req.splitter and req.splitter.temperature is not None) else 0.0)
    split_top_p   = (float(req.splitter.top_p)       if (req.splitter and req.splitter.top_p       is not None) else 0.9)

    persona      = (req.persona or "General Public").strip()
    paper_title  = (req.paper_title or "").strip()
    markdown     = req.markdown
    target_sections = max(1, int(req.target_sections or 5))

    st_adapter = (req.storyteller.adapter if (req.storyteller and req.storyteller.adapter) else STORYTELLER_ADAPTER)
    st_preset  = ((req.storyteller.length_preset if (req.storyteller and req.storyteller.length_preset) else "medium")).lower()
    st_temp    = float(req.storyteller.temperature) if (req.storyteller and req.storyteller.temperature is not None) else 0.0
    st_top_p   = float(req.storyteller.top_p) if (req.storyteller and req.storyteller.top_p is not None) else 0.9
    st_max_new = int(req.storyteller.max_new_tokens) if (req.storyteller and req.storyteller.max_new_tokens is not None) else 1500
    st_min_new = int(req.storyteller.min_new_tokens) if (req.storyteller and req.storyteller.min_new_tokens is not None) else 600

    # retrieval defaults
    st_retriever       = (req.storyteller.retriever       if (req.storyteller and req.storyteller.retriever) else "auto")
    st_retriever_model = (req.storyteller.retriever_model if (req.storyteller and req.storyteller.retriever_model) else "sentence-transformers/all-MiniLM-L6-v2")
    st_k               = (int(req.storyteller.k)               if (req.storyteller and req.storyteller.k               is not None) else 3)
    st_max_ctx_chars   = (int(req.storyteller.max_ctx_chars)   if (req.storyteller and req.storyteller.max_ctx_chars   is not None) else 1400)
    st_seg_words       = (int(req.storyteller.seg_words)       if (req.storyteller and req.storyteller.seg_words       is not None) else 180)
    st_overlap_words   = (int(req.storyteller.overlap_words)   if (req.storyteller and req.storyteller.overlap_words   is not None) else 60)

    # workspace
    rid = str(uuid.uuid4())
    log_path, trace = _trace_open(rid)
    trace("request.received", "Request received.",
          persona=persona, paper_title=paper_title, target_sections=target_sections)

    workdir = tempfile.mkdtemp(prefix=f"twostage_{rid}_")
    in_split = os.path.join(workdir, "in_splitter.jsonl")
    out_split = os.path.join(workdir, "out_splitter.jsonl")
    out_story = os.path.join(workdir, "out_story.jsonl")

    timings = {}
    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)
    env = dict(env or {})
    env["TRACE_LOG_FILE"] = log_path
    env["TRACE_REQ_ID"] = rid
    trace("gpu.pick", f"Picked GPU {gpu_idx}.", gpu_idx=gpu_idx)
    trace("io.workspace", "Workspace prepared.", workdir=workdir,
          in_split=in_split, out_split=out_split, out_story=out_story, log_file=log_path)


    try:
        # input per SPLITTER
        record = {"id": rid, "persona": persona, "md": markdown, "title": paper_title or None}
        with open(in_split, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        trace("splitter.input_ready", "Splitter input ready.", bytes=os.path.getsize(in_split))

        # SPLITTER
        t0 = time.time()
        trace("splitter.start", f"Splitting into {target_sections} sections…",
          cfg={"base_model": split_base, "adapter_path": split_adapter,
               "max_new_tokens": split_max_new, "temperature": split_temp,
               "top_p": split_top_p, "sections": target_sections})

        _run_splitter(
            in_split, out_split,
            cfg={
                "base_model": split_base,
                "adapter_path": split_adapter,
                "sections": target_sections,
                "max_new_tokens": split_max_new,
                "temperature": split_temp,
                "top_p": split_top_p,
            },
            timeout=TIMEOUT_SPLITTER, env=env
        )
        timings["splitter_s"] = round(time.time() - t0, 3)

        trace("splitter.done", "Splitter finished.", elapsed_s=timings["splitter_s"], out_size=os.path.getsize(out_split))

        # STORYTELLER
        t1 = time.time()

        trace("storyteller.start", "Generating narrative sections…",
          cfg={"adapter": st_adapter, "preset": st_preset, "temperature": st_temp,
               "top_p": st_top_p, "max_new_tokens": st_max_new,
               "retriever": st_retriever, "k": st_k})

        _run_storyteller(
            in_story=out_split, out_story=out_story,
            cfg={
                "adapter": st_adapter,
                "preset": st_preset,
                "temperature": st_temp, "top_p": st_top_p,
                "max_new_tokens": st_max_new, 
                "retriever": st_retriever, "retriever_model": st_retriever_model,
                "k": st_k, "max_ctx_chars": st_max_ctx_chars,
                "seg_words": st_seg_words, "overlap_words": st_overlap_words,
            },
            timeout=TIMEOUT_STORY, env=env
        )
        timings["storyteller_s"] = round(time.time() - t1, 3)

        trace("storyteller.done", "Storyteller finished.", elapsed_s=timings["storyteller_s"], out_size=os.path.getsize(out_story))


        # Leggi risultati
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
            norm_sections.append({
                "title": title,
                "text": text,
                "paragraphs": _split_paragraphs(text),
            })
        sections = norm_sections
        trace("postprocess.sections_parsed", f"Parsed {len(sections)} sections.",
          titles=[s.get("title") for s in sections])
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
                "timings": {**timings, "total_s": round((time.time() - t_start), 3)},
                "trace": {"id": rid, "log_file": log_path},
                "gpu": {"idx": gpu_idx},
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
                    "adapter": st_adapter,
                    "length_preset": st_preset,
                    "temperature": st_temp,
                    "top_p": st_top_p,
                    "max_new_tokens": st_max_new,
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
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)

# =========================
# Rigenerazione da outline (salta lo splitter) — invariato salvo env/lock
# =========================
class TwoStageFromOutlineReq(BaseModel):
    persona: str
    paper_title: str | None = None
    cleaned_text: str
    outline: List[Dict[str, Any]]
    storyteller: Optional[StoryCfg] = None

@app.post("/api/two_stage_story_from_outline")
def two_stage_story_from_outline(req: TwoStageFromOutlineReq, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    persona     = (req.persona or "General Public").strip()
    paper_title = (req.paper_title or "Paper").strip()
    cleaned     = req.cleaned_text or ""
    outline     = req.outline or []

    st = req.storyteller or StoryCfg()
    story_adapter = st.adapter or STORYTELLER_ADAPTER
    st_preset = ((st.length_preset or "medium")).lower()
    st_temp     = float(st.temperature) if st.temperature is not None else 0.0
    st_top_p    = float(st.top_p) if st.top_p is not None else 0.9
    st_max_new  = int(st.max_new_tokens) if st.max_new_tokens is not None else 1500
    st_min_new  = int(st.min_new_tokens) if st.min_new_tokens is not None else 600

    st_retriever       = (getattr(st, "retriever", None) or "auto")
    st_retriever_model = (getattr(st, "retriever_model", None) or "sentence-transformers/all-MiniLM-L6-v2")
    st_k               = int(getattr(st, "k", 3) or 3)
    st_max_ctx_chars   = int(getattr(st, "max_ctx_chars", 1400) or 1400)
    st_seg_words       = int(getattr(st, "seg_words", 180) or 180)
    st_overlap_words   = int(getattr(st, "overlap_words", 60) or 60)

    rid = str(uuid.uuid4())
    workdir = tempfile.mkdtemp(prefix=f"twostage2_{rid}_")
    in_story = os.path.join(workdir, "in_story.jsonl")
    out_story = os.path.join(workdir, "out_story.jsonl")

    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)

    def _filter_unsupported(text: str, source: str) -> str:
        src = source.lower()
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        kept = []
        for s in sents:
            toks = re.findall(r"\b[A-Z][a-zA-Z\-]{2,}\b", s)
            if any(t.lower() not in src for t in toks):
                continue
            kept.append(s)
        return " ".join(kept).strip()

    try:
        record = {
            "id": rid,
            "persona": persona,
            "paper_title": paper_title,
            "cleaned_text": cleaned,
            "sections": outline,
        }
        with open(in_story, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        _run_storyteller(
            in_story, out_story,
            cfg={
                "adapter": story_adapter, "preset": st_preset,
                "temperature": st_temp, "top_p": st_top_p,
                "max_new_tokens": st_max_new, 
                "retriever": st_retriever, "retriever_model": st_retriever_model,
                "k": st_k, "max_ctx_chars": st_max_ctx_chars,
                "seg_words": st_seg_words, "overlap_words": st_overlap_words,
            },
            timeout=TIMEOUT_STORY, env=env
        )

        story_obj = _read_first_jsonl(out_story)
        gen = story_obj.get("generation", {}) or {}
        sections = gen.get("sections", []) or []

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
            norm.append({
                "title": title,
                "text": text,
                "paragraphs": _split_paragraphs(text),
            })

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
                "aiTitle": ret_title,
                "title_source": "paper_title_locked",
                "gpu": {"idx": gpu_idx},
                "storyteller_params": {
                    "length_preset": st_preset, "temperature": st_temp, "top_p": st_top_p,
                    "max_new_tokens": st_max_new, 
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
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)

# =========================
# NEW: Rigenerazione parziale (solo target) — mappa sparsa
# =========================
@app.post("/api/regen_sections_vm", response_model=RegenSectionsVMResp)
def regen_sections_vm(req: RegenSectionsVMReq, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    persona      = (req.persona or "General Public").strip()
    paper_title  = (req.paper_title or "").strip() or None
    cleaned_text = (req.cleaned_text or "").strip()
    outline      = req.outline or []
    targets_in   = req.targets or []

    if not outline or not isinstance(outline, list):
        raise HTTPException(422, "outline mancante o non valido")
    if not cleaned_text:
        raise HTTPException(422, "cleaned_text mancante")
    if not targets_in:
        raise HTTPException(400, "no valid targets")

    st = req.storyteller or StoryCfg()
    story_adapter = st.adapter or STORYTELLER_ADAPTER
    st_preset = ((st.length_preset or "medium")).lower()
    st_temp     = float(st.temperature) if st.temperature is not None else 0.0
    st_top_p    = float(st.top_p) if st.top_p is not None else 0.9
    st_max_new  = int(st.max_new_tokens) if st.max_new_tokens is not None else 1500
    st_min_new  = int(st.min_new_tokens) if st.min_new_tokens is not None else 600
    st_retriever       = (getattr(st, "retriever", None) or "auto")
    st_retriever_model = (getattr(st, "retriever_model", None) or "sentence-transformers/all-MiniLM-L6-v2")
    st_k               = int(getattr(st, "k", 3) or 3)
    st_max_ctx_chars   = int(getattr(st, "max_ctx_chars", 1400) or 1400)
    st_seg_words       = int(getattr(st, "seg_words", 180) or 180)
    st_overlap_words   = int(getattr(st, "overlap_words", 60) or 60)

    uniq_targets = sorted({int(t) for t in targets_in if isinstance(t, (int,))})
    valid_targets: List[int] = []
    for t in uniq_targets:
        if 0 <= t < len(outline):
            valid_targets.append(t)
    if not valid_targets:
        raise HTTPException(400, "no valid targets in range")

    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)

    sparse_sections: Dict[str, Dict[str, Any]] = {}
    timings: Dict[str, Any] = {"per_section_s": {}}
    rid = str(uuid.uuid4())

    try:
        for t in valid_targets:
            start_t = time.time()
            workdir = tempfile.mkdtemp(prefix=f"regen_{rid}_{t}_")
            in_story = os.path.join(workdir, "in_story.jsonl")
            out_story = os.path.join(workdir, "out_story.jsonl")

            target_section = outline[t] or {}
            fixed_title = (target_section.get("title") or f"Section {t+1}").strip()
            section_record = {
                "id": f"{rid}_{t}",
                "persona": persona,
                "paper_title": paper_title or "",
                "cleaned_text": cleaned_text,
                "sections": [ {"title": fixed_title, "description": (target_section.get("description") or "")} ],
            }

            with open(in_story, "w", encoding="utf-8") as f:
                f.write(json.dumps(section_record, ensure_ascii=False) + "\n")

            _run_storyteller(
                in_story, out_story,
                cfg={
                    "adapter": story_adapter, "preset": st_preset,
                    "temperature": st_temp, "top_p": st_top_p,
                    "max_new_tokens": st_max_new, 
                    "retriever": st_retriever, "retriever_model": st_retriever_model,
                    "k": st_k, "max_ctx_chars": st_max_ctx_chars,
                    "seg_words": st_seg_words, "overlap_words": st_overlap_words,
                },
                timeout=TIMEOUT_STORY, env=env
            )

            story_obj = _read_first_jsonl(out_story)
            gen = story_obj.get("generation", {}) or {}
            out_sections = gen.get("sections", []) or []

            text = ""
            if out_sections and isinstance(out_sections[0], dict):
                raw = out_sections[0].get("text") or out_sections[0].get("narrative") or ""
                text = _maybe_unwrap_json_text(raw, fixed_title).strip()
            if not text:
                text = (target_section.get("description") or "").strip()

            sparse_sections[str(t)] = {
                "title": fixed_title,
                "text": text,
                "paragraphs": _split_paragraphs(text),
            }

            timings["per_section_s"][str(t)] = round(time.time() - start_t, 3)
            shutil.rmtree(workdir, ignore_errors=True)

        meta = {
            "req_id": rid,
            "timings": {**timings},
            "gpu": {"idx": gpu_idx},
            "storyteller_params": {
                "adapter": story_adapter,
                "length_preset": st_preset,
                "temperature": st_temp,
                "top_p": st_top_p,
                "max_new_tokens": st_max_new,
                "retriever": st_retriever,
                "retriever_model": st_retriever_model,
                "k": st_k,
                "max_ctx_chars": st_max_ctx_chars,
                "seg_words": st_seg_words,
                "overlap_words": st_overlap_words,
            },
            "targets": valid_targets,
        }

        return {
            "persona": persona,
            "paper_title": paper_title,
            "sections": sparse_sections,
            "meta": meta,
        }
    finally:
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)

# =========================
# NEW: Rigenerazione paragrafo singolo — persona-aware + retrieval leggero
# =========================
class ParagraphOps(BaseModel):
    paraphrase: bool
    simplify: bool
    length_op: Literal["keep", "shorten", "lengthen"]

class ParagraphOpsReq(BaseModel):
    persona: str
    paper_title: str
    cleaned_text: str
    section: Dict[str, Any]   
    paragraph_index: int
    ops: ParagraphOps
    temperature: float = 0.3
    top_p: float = 0.9
    n: int = 2
    k_ctx: int = 3                      # NEW: top-k frammenti di contesto
    max_ctx_chars: int = 1200           # NEW: limite caratteri contesto
    length_preset: Optional[Literal["short","medium","long"]] = None
    seed: Optional[int] = None

def _snap_creativity_to_step10(temp_in: float) -> float:
    """
    Converte l'input 'temperature' in base_temp:
    - se > 1.2 assumiamo percento (10..100) -> /100
    - clamp su [0.10, 1.20]
    - snap ai decimi (0.1, 0.2, …, 1.2) per coerenza UI a scatti di 10%
    """
    base = temp_in
    if base > 1.2:
        base = base / 100.0
    base = max(0.10, min(1.20, base))
    snapped = round(base, 1)
    if snapped < 0.10:
        snapped = 0.10
    return snapped

def _build_temperature_schedule(base_temp: float, n: int) -> List[float]:
    """
    Genera n temperature: base, base+0.04, base+0.08, … clamp a 1.20
    Esempio: base=0.30, n=3 -> [0.30, 0.34, 0.38]
    """
    out = []
    for i in range(max(1, int(n))):
        t = min(1.20, base_temp + 0.04 * i)
        out.append(round(t, 2))
    return out

def _normalize_text_for_dedup(s: str) -> str:
    # normalizzazione semplice compatibile con re standard (niente \p{L})
    return re.sub(r"[^\w\s]+", "", (s or "")).lower().strip()

@app.post("/api/regen_paragraph_vm")
def regen_paragraph_vm(req: ParagraphOpsReq, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    persona = req.persona or "General Public"
    paper_title = req.paper_title or "Paper"
    cleaned_text = req.cleaned_text or ""
    sec = req.section or {}
    paragraphs = sec.get("paragraphs") or []
    title = sec.get("title") or "Section"
    idx = int(req.paragraph_index)
    if not paragraphs or idx < 0 or idx >= len(paragraphs):
        raise HTTPException(422, "invalid paragraph_index or section.paragraphs")

    target_text = paragraphs[idx]
    # contesto immediato (vicini)
    context_local = "\n\n".join(paragraphs[max(0, idx - 1): idx + 2])

    # retrieval leggero su cleaned_text (top-k frammenti simili al paragrafo target)
    ctx_frags = _tfidf_topk_fragments(
        cleaned_text=cleaned_text,
        query=target_text,
        max_words=180, overlap=60,
        k=max(1, int(req.k_ctx)),
        max_chars=max(256, int(req.max_ctx_chars))
    )
    context_retrieved = "\n\n".join(ctx_frags).strip() if ctx_frags else ""

    ops = req.ops

    simplify_instruction = "Use simpler vocabulary and sentences." if ops.simplify else ""
    paraphrase_instruction = "Paraphrase faithfully without changing facts." if ops.paraphrase else "Keep most phrasing unchanged."
    persona_block = _persona_guidance(persona)

    # --- lunghezza ---
    length_op = getattr(req.ops, "length_op", "keep")
    length_preset = (req.length_preset or {
        "shorten": "short",
        "keep": "medium",
        "lengthen": "long",
    }.get(length_op, "medium")).lower()

    if length_preset == "short":
        length_instruction = "Write a shorter version, target 20–45 words (max ~60)."
        max_new_tokens = 100
        min_new_tokens = 20
    elif length_preset == "long":
        length_instruction = "Write a longer version, target 120–190 words (max ~200)."
        max_new_tokens = 300
        min_new_tokens = 160
    else:
        length_instruction = "Keep similar length, target 50–100 words (max ~110)."
        max_new_tokens = 200
        min_new_tokens = 90

    # ---- creatività: step 10% + schedule per alternative ----
    base_temp = _snap_creativity_to_step10(float(req.temperature))
    temps = _build_temperature_schedule(base_temp, req.n)
    top_p = float(req.top_p or 0.9)

    # 🧠 Prompt persona-aware (per singola alternativa)
    prompt = f"""
You are an AI Scientist Storyteller writing for a {persona}.
Paper title: {paper_title}
Section: {title}

Guidance for the persona:
- {persona_block}

Context (nearby paragraphs):
\"\"\"{context_local}\"\"\"

Retrieved paper fragments (factual grounding):
\"\"\"{context_retrieved if context_retrieved else "[NO CONTEXT FOUND]"}\"\"\"

Target paragraph to rewrite:
\"\"\"{target_text}\"\"\"

Rewrite this paragraph following these operations:
- {paraphrase_instruction}
- {simplify_instruction}
- {length_instruction}
- Target length must fit the range stated above; do not exceed it.
- Use English only. No lists; write fluent prose.
- Never invent facts, names, or numbers. If something is not in context, do not add it.
- If audience is non-expert, prefer a brief concrete example or analogy when helpful.

Return ONLY JSON with this schema:
{{ "text": "..." }}
""".strip()

    # Workspace base (env + lock GPU)
    rid = str(uuid.uuid4())
    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)

    outputs: List[Dict[str, Any]] = []
    seen = set()
    MAX_ATTEMPTS_PER_ALT = 5  # oltre al primo tentativo

    try:
        for i, t in enumerate(temps):
            # directory per ogni alternativa (evita collisioni file)
            workdir = tempfile.mkdtemp(prefix=f"parops_{rid}_{i}_")
            in_json = os.path.join(workdir, "in.jsonl")
            out_json = os.path.join(workdir, "out.jsonl")

            attempt = 0
            curr_temp = t
            while True:
                seed = (req.seed if req.seed is not None else None)
                if seed is None:
                    seed = int.from_bytes(os.urandom(4), "big")

                record = {
                    "id": f"{rid}_{i}_{attempt}",
                    "persona": persona,
                    "paper_title": paper_title,
                    "cleaned_text": cleaned_text,
                    "prompt": prompt
                }
                with open(in_json, "w", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                cmd = [
                    PY, STORYTELLER_SCRIPT,
                    "--in_jsonl", in_json,
                    "--out_jsonl", out_json,
                    "--adapter", STORYTELLER_ADAPTER,
                    "--preset", length_preset,
                    "--temperature", str(curr_temp),
                    "--top_p", str(top_p),
                    "--max_new_tokens", str(max_new_tokens),
                    "--seed", str(seed),
                ]

                _run(cmd, timeout=min(TIMEOUT_STORY, 900), env=env, stage=f"parregen_alt{i}_try{attempt}")

                story_obj = _read_first_jsonl(out_json)
                gen = story_obj.get("generation") or story_obj.get("output") or {}

                # parsing robusto: preferisci JSON {"text": "..."}; fallback su "text" stringa o blob
                candidate_text = None
                if isinstance(gen, dict) and isinstance(gen.get("text"), str):
                    try:
                        obj = json.loads(gen["text"])
                        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                            candidate_text = obj["text"].strip()
                        else:
                            candidate_text = gen["text"].strip()
                    except Exception:
                        candidate_text = gen["text"].strip()
                elif isinstance(gen, dict) and isinstance(gen.get("alternatives"), list):
                    for a in gen["alternatives"]:
                        if isinstance(a, dict) and isinstance(a.get("text"), str):
                            candidate_text = a["text"].strip()
                            break
                elif isinstance(gen, str) and gen.strip():
                    candidate_text = gen.strip()

                if not candidate_text:
                    candidate_text = target_text  # ultra fallback

                norm = _normalize_text_for_dedup(candidate_text)
                if norm not in seen:
                    seen.add(norm)
                    outputs.append({"text": candidate_text, "temperature": curr_temp, "seed": seed})
                    break

                # duplicato → bump temperatura e riprova
                if attempt >= MAX_ATTEMPTS_PER_ALT:
                    outputs.append({"text": candidate_text, "temperature": curr_temp, "seed": seed, "dup": True})
                    break

                curr_temp = round(min(1.20, curr_temp + 0.02), 2)
                attempt += 1

            # cleanup cartella alternativa
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass

    finally:
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)

    # eventuale taglio a n richieste
    alts = [{"text": o["text"]} for o in outputs[:max(1, int(req.n))]]

    return {
        "alternatives": alts,
        "meta": {
            "applied_ops": {
                "paraphrase": ops.paraphrase, "simplify": ops.simplify, "length_op": ops.length_op
            },
            "section_title": title,
            "paragraph_index": idx,
            "gpu": {"idx": gpu_idx},
            "ctx_used": bool(context_retrieved),
            "creativity": {
                "base_temperature": base_temp,
                "schedule": temps,
                "top_p": top_p,
            },
        },
    }
