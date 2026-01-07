# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from typing import Optional, Dict, Any

from utils import (
    get_logger,
    load_config,
    load_patient_json,
    patient_to_text,
    extract_json_block,
    merge_with_skeleton,
    SearchClient,
    build_context_from_docs,
    LLMClient,
    LLMConfig,
    ASRClient,
)
from prompt import build_prompt, skeleton_for_task


def run_from_config(cfg: Dict[str, Any]):
    # ---------- Logging ----------
    log_level = (cfg.get("logging", {}) or {}).get("level", "INFO").upper()
    logger = get_logger(level=getattr(logging, log_level, logging.INFO))
    logger.info("Start LAOS inference (config-driven)")

    # ---------- Core task & IO ----------
    task = cfg.get("task")
    if task not in ["admission", "surgery", "discharge"]:
        raise ValueError("config.task must be one of: admission | surgery | discharge")
    io = cfg.get("io", {}) or {}
    patient_json = io.get("patient_json")
    out_dir = io.get("out_dir", "outputs")
    if not patient_json:
        raise ValueError("config.io.patient_json is required")

    # ---------- Input load ----------
    logger.info(f"Loading patient JSON: {patient_json}")
    patient = load_patient_json(patient_json)
    patient_text = patient_to_text(patient)
    logger.info(f"Patient text length: {len(patient_text)}")

    # ---------- ASR ----------
    asr_cfg = cfg.get("asr", {}) or {}
    audio_path = asr_cfg.get("audio_path")
    if audio_path:
        asr_client = ASRClient(
            provider=asr_cfg.get("provider", "funasr"),
            model=asr_cfg.get("model", "paraformer-zh"),
            device=asr_cfg.get("device", "auto"),
            params=asr_cfg.get("params", {}),
            logger=logger,
        )
        asr_text = asr_client.transcribe(audio_path)
        if asr_text:
            logger.info(f"ASR transcript length: {len(asr_text)}")
            patient_text += "\nASR: " + asr_text
        else:
            logger.warning("ASR transcript is empty or ASR unavailable.")
    else:
        logger.info("ASR skipped (no audio_path).")

    # ---------- Search ----------
    search_cfg = cfg.get("search", {}) or {}
    sclient = SearchClient(
        base_url=search_cfg.get("base_url"),
        api_key=search_cfg.get("api_key"),
        logger=logger,
        timeout=search_cfg.get("timeout", 15),
    )
    s_task = search_cfg.get("task", task)
    s_topk = int(search_cfg.get("top_k", 5))
    s_params = search_cfg.get("params", {}) or {}
    logger.info(f"Searching context | url={search_cfg.get('base_url')} | task={s_task} | top_k={s_topk}")
    docs = sclient.search(task=s_task, query=patient_text, top_k=s_topk, params=s_params)
    for i, d in enumerate(docs):
        preview = (d.get("text") or "")[:80].replace("\n", " ")
        logger.info(f"  Search#{i+1} id={d.get('id')} score={d.get('score')} | {preview}...")
    common_terms, examples = build_context_from_docs(docs, top_k=s_topk)

    # ---------- Doctor feedback ----------
    fb_cfg = cfg.get("doctor_feedback", {}) or {}
    doctor_feedback = fb_cfg.get("text") or ""
    fb_file = fb_cfg.get("file")
    if not doctor_feedback and fb_file and os.path.exists(fb_file):
        with open(fb_file, "r", encoding="utf-8") as f:
            doctor_feedback = f.read()
    if doctor_feedback:
        logger.info(f"Doctor feedback length: {len(doctor_feedback)}")
    else:
        logger.info("No doctor feedback provided.")

    # ---------- Prompt ----------
    prompt_text = build_prompt(
        task=task,
        patient_input=patient_text,
        common_terms=common_terms,
        examples=examples,
        doctor_feedback=doctor_feedback,
    )
    logger.info(f"Prompt built | length={len(prompt_text)}")

    # ---------- LLM ----------
    llm_cfg_raw = cfg.get("llm", {}) or {}
    llm_cfg = LLMConfig(
        provider=llm_cfg_raw.get("provider", "hf_causal"),
        model=llm_cfg_raw.get("model", "Qwen2-7B-Instruct"),
        temperature=float(llm_cfg_raw.get("temperature", 0.1)),
        top_p=float(llm_cfg_raw.get("top_p", 0.9)),
        top_k=int(llm_cfg_raw.get("top_k", 40)),
        max_tokens=int(llm_cfg_raw.get("max_tokens", 1024)),
        device=llm_cfg_raw.get("device", "auto"),
        dtype=llm_cfg_raw.get("dtype", "auto"),
        chat_template=llm_cfg_raw.get("chat_template"),
    )
    llm = LLMClient(llm_cfg, logger=logger)
    raw = llm.generate(prompt_text)

    # ---------- JSON post-process ----------
    parsed = extract_json_block(raw)
    if parsed is None:
        logger.warning("Failed to parse JSON from LLM output; using empty dict for merge.")
        parsed = {}
    skeleton = skeleton_for_task(task)
    final_obj = merge_with_skeleton(skeleton, parsed)
    logger.info(f"Final JSON keys: {list(final_obj.keys())}")

    # ---------- Save ----------
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(patient_json))[0]
    out_path = os.path.join(out_dir, f"{base}_{task}_pred.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_obj, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved prediction -> {out_path}")
    print(f"Saved prediction -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_from_config(cfg)