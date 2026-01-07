# -*- coding: utf-8 -*-

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

# Optional deps (guarded)
try:
    from openai import OpenAI  # pip install openai
except Exception:
    OpenAI = None

# Transformers (pipeline and CausalLM)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # pip install transformers
    import torch
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

# FunASR for Paraformer
try:
    from funasr import AutoModel as FunASRAutoModel  # pip install funasr
except Exception:
    FunASRAutoModel = None

# YAML config
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


# ---------------- Logging ----------------
def get_logger(name="laos", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ---------------- Config ----------------
def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML or JSON config into a dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if ext in [".yaml", ".yml"]:
        if yaml is None:
            raise RuntimeError("pyyaml not installed; please `pip install pyyaml`.")
        return yaml.safe_load(text)
    elif ext == ".json":
        return json.loads(text)
    else:
        # try yaml first then json
        try:
            if yaml is None:
                raise RuntimeError
            return yaml.safe_load(text)
        except Exception:
            return json.loads(text)


# ---------------- Patient IO ----------------
def load_patient_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def patient_to_text(patient: Dict) -> str:
    """
    Convert structured patient JSON to a single text block.
    """
    parts = []
    for k, v in patient.items():
        if isinstance(v, (str, int, float)):
            parts.append(f"{k}: {v}")
        elif isinstance(v, list):
            parts.append(f"{k}: " + "; ".join(map(str, v)))
        elif isinstance(v, dict):
            sub = "; ".join(f"{sk}: {sv}" for sk, sv in v.items())
            parts.append(f"{k}: {sub}")
        else:
            parts.append(f"{k}: {str(v)}")
    return "\n".join(parts)


# ---------------- ASR Client ----------------
class ASRClient:
    """
    ASR front-end with multiple providers.
    - funasr + paraformer (preferred for Chinese medical notes)
    - whisper (optional)
    - speech_recognition (fallback)
    """

    def __init__(self, provider: str = "none", model: Optional[str] = None, device: str = "auto", logger: Optional[logging.Logger] = None, params: Optional[Dict[str, Any]] = None):
        self.provider = (provider or "none").lower()
        self.model = model
        self.device = device
        self.params = params or {}
        self.logger = logger or get_logger()
        self.funasr_model = None
        self._init_backend()

    def _init_backend(self):
        if self.provider == "funasr":
            if FunASRAutoModel is None:
                self.logger.error("FunASR is not installed. Please `pip install funasr`.")
                return
            name = self.model or "paraformer-zh"
            try:
                self.funasr_model = FunASRAutoModel(model=name, vad_model=None, punc_model=None)
                self.logger.info(f"Loaded FunASR model: {name}")
            except Exception as e:
                self.logger.error(f"Failed to load FunASR model {name}: {e}")
                self.funasr_model = None

    def transcribe(self, audio_path: Optional[str]) -> str:
        if not audio_path or not os.path.exists(audio_path):
            return ""
        if self.provider == "funasr" and self.funasr_model is not None:
            try:
                res = self.funasr_model.generate(
                    input=audio_path,
                    **self.params
                )
                # FunASR returns a list of segments; take concatenated text
                if isinstance(res, list) and res:
                    # common schema: [{"text": "xxx"}]
                    texts = []
                    for seg in res:
                        t = seg.get("text") or seg.get("sentence", "")
                        if t:
                            texts.append(t)
                    out = " ".join(texts).strip()
                    self.logger.info(f"FunASR transcript length: {len(out)}")
                    return out
                return ""
            except Exception as e:
                self.logger.error(f"FunASR transcribe failed: {e}")
                return ""
        # Whisper
        try:
            import whisper  # pip install -U openai-whisper
            model = whisper.load_model(self.model or "base")
            result = model.transcribe(audio_path, fp16=False)
            return (result.get("text") or "").strip()
        except Exception:
            pass
        # SpeechRecognition fallback
        try:
            import speech_recognition as sr  # pip install SpeechRecognition
            r = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = r.record(source)
            return r.recognize_google(audio, language="zh-CN")
        except Exception:
            return ""


# ---------------- JSON helpers ----------------
def extract_json_block(text: str) -> Optional[Dict]:
    """
    Extract the first valid JSON object from text.
    """
    # Direct try
    try:
        return json.loads(text)
    except Exception:
        pass
    # Bracket scanning
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in starts:
        for e in range(len(text) - 1, s, -1):
            if text[e] == "}":
                frag = text[s : e + 1]
                try:
                    return json.loads(frag)
                except Exception:
                    continue
    return None


def merge_with_skeleton(skeleton: Dict, data: Optional[Dict]) -> Dict:
    """
    Merge parsed data into a minimal skeleton, keeping only known top-level keys.
    """
    if data is None:
        return skeleton
    out = {}
    for k, v in skeleton.items():
        if k in data:
            dv = data[k]
            if isinstance(v, dict) and isinstance(dv, dict):
                out[k] = {**v, **dv}  # shallow merge
            else:
                out[k] = dv
        else:
            out[k] = v
    return out


# ---------------- Search client----------------
class SearchClient:
    """
    - If base_url is provided, POST JSON payload to that endpoint.
    - Otherwise, return empty results.
    Expected response: list[ { "id": str, "text": str, "score": float? } ] or { "docs": [...] }
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 15, logger: Optional[logging.Logger] = None):
        self.base_url = base_url or os.environ.get("LAOS_SEARCH_URL", None)
        self.api_key = api_key or os.environ.get("LAOS_SEARCH_API_KEY", None)
        self.timeout = timeout
        self.logger = logger or get_logger()

    def search(self, task: str, query: str, top_k: int = 5, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.base_url:
            self.logger.warning("Search base_url is not set; returning empty search results.")
            return []
        payload = {
            "task": task,
            "query": query,
            "top_k": top_k,
            "params": params or {},
        }
        try:
            import urllib.request
            req = urllib.request.Request(self.base_url, method="POST")
            req.add_header("Content-Type", "application/json")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            data = json.dumps(payload).encode("utf-8")
            with urllib.request.urlopen(req, data=data, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            docs = obj.get("docs", obj)
            results = []
            for i, d in enumerate(docs):
                if isinstance(d, dict):
                    results.append({
                        "id": d.get("id", f"doc_{i}"),
                        "text": d.get("text", ""),
                        "score": float(d.get("score", 0.0)) if "score" in d else None,
                    })
                else:
                    results.append({"id": f"doc_{i}", "text": str(d), "score": None})
            self.logger.info(f"Search returned {len(results)} docs.")
            return results
        except Exception as e:
            self.logger.error(f"Search request failed: {e}")
            return []


def build_context_from_docs(docs: List[Dict[str, Any]], top_k: int = 5) -> Tuple[str, str]:
    """
    Convert search results into common terms and an example snippet.
    """
    if not docs:
        return "[None]", "[None]"
    lines = []
    for d in docs[:top_k]:
        snippet = (d.get("text") or "")[:240].replace("\n", " ").strip()
        lines.append(f"- {snippet} ...")
    common_terms = "\n".join(lines)
    example = lines[0] if lines else "[None]"
    return common_terms, example


# ---------------- LLM client ----------------
@dataclass
class LLMConfig:
    provider: str = "dummy"  # dummy | openai | hf_textgen | hf_causal
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1024
    device: str = "auto"  # auto | cpu | cuda | cuda:0 ...
    dtype: str = "auto"   # auto | float16 | bfloat16 | float32
    chat_template: Optional[str] = None  # override if needed


class LLMClient:
    """
    Supports:
    - provider=openai: Chat Completions
    - provider=hf_textgen: transformers.pipeline("text-generation")
    - provider=hf_causal: AutoModelForCausalLM + AutoTokenizer with chat template (Qwen/LLaMA)
    """

    def __init__(self, cfg: LLMConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or get_logger()
        self.client = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._init_backend()

    def _select_device(self):
        if torch is None:
            return None
        if self.cfg.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.cfg.device

    def _select_dtype(self):
        if torch is None or self.cfg.dtype == "auto":
            return None
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.cfg.dtype.lower(), None)

    def _init_backend(self):
        provider = self.cfg.provider.lower()
        if provider == "openai" and OpenAI is not None:
            self.client = OpenAI()
            self.logger.info("LLM backend: OpenAI Chat Completions.")
        elif provider == "hf_textgen" and pipeline is not None:
            self.pipeline = pipeline(
                "text-generation",
                model=self.cfg.model,
                device_map="auto" if self.cfg.device == "auto" else None,
                torch_dtype=self._select_dtype(),
                max_new_tokens=self.cfg.max_tokens,
            )
            self.logger.info(f"LLM backend: HF pipeline text-generation | model={self.cfg.model}")
        elif provider == "hf_causal" and AutoModelForCausalLM is not None and AutoTokenizer is not None:
            device = self._select_device()
            dtype = self._select_dtype()
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model,
                device_map="auto" if self.cfg.device == "auto" else None,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            # Qwen/LLaMA: use chat template if available
            if not self.tokenizer.chat_template and self.cfg.chat_template:
                self.tokenizer.chat_template = self.cfg.chat_template
            self.logger.info(f"LLM backend: HF CausalLM | model={self.cfg.model} | device={device}")
        else:
            self.logger.warning("LLM backend: dummy (no real provider configured).")

    def _apply_chat_template(self, prompt: str) -> Any:
        """
        For chat models (Qwen/LLaMA-Instruct), format into messages and use tokenizer.apply_chat_template.
        """
        if self.tokenizer is None:
            return prompt
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        except Exception:
            # Fallback: just return raw prompt
            return prompt

    def generate(self, prompt: str) -> str:
        provider = self.cfg.provider.lower()
        self.logger.info(f"LLM generate | provider={self.cfg.provider} | model={self.cfg.model}")
        if provider == "openai" and self.client is not None:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_tokens,
            )
            out = resp.choices[0].message.content
            self.logger.info(f"LLM output length: {len(out)} chars")
            return out

        if provider == "hf_textgen" and self.pipeline is not None:
            out = self.pipeline(
                prompt,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_tokens,
            )
            text = out[0]["generated_text"] if isinstance(out, list) and out else str(out)
            self.logger.info(f"LLM output length: {len(text)} chars")
            return text

        if provider == "hf_causal" and self.model is not None and self.tokenizer is not None:
            text_prompt = self._apply_chat_template(prompt)
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            full = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            # Try to strip the prompt part
            if isinstance(text_prompt, str) and full.startswith(text_prompt):
                out = full[len(text_prompt):]
            else:
                out = full
            self.logger.info(f"LLM output length: {len(out)} chars")
            return out

        # dummy
        self.logger.warning("Using dummy LLM; returning placeholder JSON.")
        return '{"note": "dummy-llm: configure a real provider to get full outputs"}'