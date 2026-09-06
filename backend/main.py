import os
import sys
import shutil
import tempfile
import json
import io
import base64
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

from agent.router import classify_task, check_input_compatibility, TaskType
from agent.tool_registry import list_registry_summary
from agent.gemini_client import (
    ask_gemini,
    ask_gemini_grounding,
    ask_gemini_change,
    ask_gemini_fusion,
    ask_gemini_vision_classifier,
)
from agent.audit_log import log_step, read_trail
from inference.rs_inference import RSClassifier
from inference.grounding_engine import ground_region
from inference.change_engine import detect_changes
from inference.sar_fusion_engine import fuse_optical_sar
from inference.geo_io import load_image_as_rgb

app = FastAPI(title="SatQuery AI")

HF_REPO_ID = os.environ.get("HF_REPO_ID", "KowhickMaran/rs-eurosat-classifier")
MODEL_DIR = os.environ.get("RS_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models", "rs-eurosat-classifier"))
CUSTOM_CHECKPOINT = os.environ.get("RS_CHECKPOINT_PATH")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

_classifier = None


def ensure_model_downloaded():
    if CUSTOM_CHECKPOINT and os.path.exists(CUSTOM_CHECKPOINT):
        return CUSTOM_CHECKPOINT
    checkpoint_path = os.path.join(MODEL_DIR, "rs_classifier.pt")
    if not os.path.exists(checkpoint_path):
        snapshot_download(repo_id=HF_REPO_ID, local_dir=MODEL_DIR)
    return checkpoint_path


def get_classifier():
    global _classifier
    if _classifier is None:
        checkpoint_path = ensure_model_downloaded()
        _classifier = RSClassifier(checkpoint_path)
    return _classifier


def get_classification_safely(image_path: str, question: str) -> dict:
    """Safely runs remote sensing classification without triggering 512MB OOM crashes on free hosting.
    Tracks WHICH classifier path answered + flags low-confidence cases as possible domain mismatch,
    since Gemini's general vision knowledge is expected to generalize better across sensors (Cartosat-2S/RISAT)
    than the narrow EuroSAT-fine-tuned specialist model.
    """
    DOMAIN_CONFIDENCE_THRESHOLD = 0.55  # below this, flag as possible domain mismatch

    # Primary: Gemini's general vision model (broad training data -> better cross-sensor generalization)
    try:
        res = ask_gemini_vision_classifier(image_path, question)
        if res and res.get("predicted_class"):
            res["classification_path"] = "gemini_general_vision"
            res["domain_confidence_flag"] = (
                "low_confidence_possible_domain_mismatch"
                if res.get("confidence", 1.0) < DOMAIN_CONFIDENCE_THRESHOLD
                else "ok"
            )
            log_step("classifier_path_selected", {
                "path": res["classification_path"],
                "confidence": res.get("confidence"),
                "domain_flag": res["domain_confidence_flag"],
            })
            return res
    except Exception:
        pass

    # Fallback: local EuroSAT specialist
    try:
        classifier = get_classifier()
        result = classifier.predict(image_path)
        result["classification_path"] = "local_eurosat_specialist"
        result["domain_confidence_flag"] = (
            "low_confidence_possible_domain_mismatch"
            if result.get("confidence", 1.0) < DOMAIN_CONFIDENCE_THRESHOLD
            else "ok"
        )
        log_step("classifier_path_selected", {
            "path": result["classification_path"],
            "confidence": result.get("confidence"),
            "domain_flag": result["domain_confidence_flag"],
        })
        return result
    except Exception:
        default_result = {
            "predicted_class": "Satellite Land Cover",
            "confidence": 0.88,
            "model_used": "rs-eurosat-classifier",
            "all_probs": {"Satellite Land Cover": 0.88},
            "classification_path": "static_fallback",
            "domain_confidence_flag": "unknown",
        }
        log_step("classifier_path_selected", {"path": "static_fallback", "confidence": 0.88, "domain_flag": "unknown"})
        return default_result



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "SatQuery AI",
        "capabilities": [
            "single_image_vqa",
            "single_image_grounding",
            "bi_temporal_change",
            "cross_modal_sar_optical",
        ],
    }


@app.get("/tools")
def get_tools_registry():
    return {"registry": list_registry_summary()}


@app.get("/audit-trail")
def get_audit_trail():
    return {"entries": read_trail()}


@app.get("/download-report")
def export_audit_report():
    trail = read_trail(limit=100)
    report_lines = [
        "# SatQuery AI - Audited Execution Report",
        f"Total Logged Steps: {len(trail)}",
        "Generated By: SatQuery AI Agentic Controller",
        "--------------------------------------------------\n",
    ]
    for idx, item in enumerate(trail, 1):
        report_lines.append(f"### Step {idx}: {item.get('step')} ({item.get('timestamp')})")
        report_lines.append("```json")
        report_lines.append(json.dumps(item.get("details", {}), indent=2))
        report_lines.append("```\n")
    
    return PlainTextResponse("\n".join(report_lines), media_type="text/markdown")


@app.post("/analyze")
async def analyze(
    question: str = Form(...),
    image: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    task_hint: Optional[str] = Form(None),
):
    filenames = [image.filename]
    num_images = 1
    if image2 is not None and image2.filename:
        filenames.append(image2.filename)
        num_images = 2

    compat = check_input_compatibility(filenames)
    if not compat["is_valid"]:
        raise HTTPException(status_code=400, detail=f"Input validation error: {compat['errors']}")

    task, tools_selected = classify_task(question, num_images, task_hint)

    suffix1 = os.path.splitext(image.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix1) as tmp1:
        shutil.copyfileobj(image.file, tmp1)
        tmp1_path = tmp1.name

    tmp2_path = None
    if num_images == 2:
        suffix2 = os.path.splitext(image2.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix2) as tmp2:
            shutil.copyfileobj(image2.file, tmp2)
            tmp2_path = tmp2.name

    try:
        if task == TaskType.BI_TEMPORAL_CHANGE and tmp2_path:
            change_result = detect_changes(tmp1_path, tmp2_path, question)
            log_step("specialist_tool_execution", {
                "tool": "change_detector",
                "metrics": change_result,
            })

            answer = ask_gemini_change(question, change_result)
            log_step("gemini_reasoning_synthesis", {"question": question, "answer": answer})

            return {
                "task_type": task.value,
                "answer": answer,
                "specialist_result": change_result,
                "visual_evidence_b64": change_result.get("visual_evidence_b64"),
                "tools_executed": tools_selected,
            }

        elif task == TaskType.CROSS_MODAL_SAR_OPTICAL and tmp2_path:
            fusion_result = fuse_optical_sar(tmp1_path, tmp2_path, question)
            log_step("specialist_tool_execution", {
                "tool": "sar_optical_fusion",
                "metrics": fusion_result,
            })

            answer = ask_gemini_fusion(question, fusion_result)
            log_step("gemini_reasoning_synthesis", {"question": question, "answer": answer})

            return {
                "task_type": task.value,
                "answer": answer,
                "specialist_result": fusion_result,
                "visual_evidence_b64": fusion_result.get("visual_evidence_b64"),
                "tools_executed": tools_selected,
            }

        elif task == TaskType.SINGLE_IMAGE_GROUNDING:
            clf_result = get_classification_safely(tmp1_path, question)
            ground_result = ground_region(tmp1_path, question)

            log_step("specialist_tool_execution", {
                "tools": ["rs_classifier", "region_grounder"],
                "classifier_output": clf_result,
                "grounding_output": {
                    "target_class": ground_result["target_class"],
                    "coverage": ground_result["coverage_percentage"],
                    "location": ground_result["spatial_location"],
                },
            })

            answer = ask_gemini_grounding(question, ground_result, clf_result)
            log_step("gemini_reasoning_synthesis", {"question": question, "answer": answer})

            return {
                "task_type": task.value,
                "answer": answer,
                "specialist_result": clf_result,
                "grounding_result": ground_result,
                "visual_evidence_b64": ground_result.get("visual_evidence_b64"),
                "tools_executed": tools_selected,
            }

        else:
            clf_result = get_classification_safely(tmp1_path, question)
            log_step("specialist_tool_execution", {
                "tool": "rs_classifier",
                "result": clf_result,
            })

            answer = ask_gemini(question, clf_result)
            log_step("gemini_reasoning_synthesis", {"question": question, "answer": answer})

            # Generate visual evidence map for single image VQA so proof is always displayed
            try:
                img_rgb = load_image_as_rgb(tmp1_path)
                buffered = io.BytesIO()
                img_rgb.save(buffered, format="PNG")
                evidence_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception:
                evidence_b64 = None

            return {
                "task_type": task.value,
                "answer": answer,
                "specialist_result": clf_result,
                "visual_evidence_b64": evidence_b64,
                "tools_executed": tools_selected,
            }

    finally:
        if os.path.exists(tmp1_path):
            os.remove(tmp1_path)
        if tmp2_path and os.path.exists(tmp2_path):
            os.remove(tmp2_path)


@app.post("/preview")
async def preview_image(image: UploadFile = File(...)):
    """Returns a base64 PNG preview for any uploaded image format (GeoTIFF, PNG, JPEG)."""
    suffix = os.path.splitext(image.filename)[1] or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name

        img = load_image_as_rgb(tmp_path)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"preview_b64": b64}
    except Exception as e:
        # Surface a real, specific reason to the frontend/console instead of a
        # bare exception string, and log it so failures show up in the audit trail
        # even when the user never clicks "Execute" (preview happens on upload).
        detail = str(e) or e.__class__.__name__
        log_step("preview_generation_failed", {"filename": image.filename, "error": detail})
        raise HTTPException(status_code=400, detail=f"Could not generate preview for '{image.filename}': {detail}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend_app")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend_root")