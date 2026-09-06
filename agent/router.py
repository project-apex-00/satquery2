import os
from enum import Enum
from typing import List, Dict, Any, Tuple
from agent.audit_log import log_step
from agent.tool_registry import REGISTRY


class TaskType(str, Enum):
    SINGLE_IMAGE_VQA = "single_image_vqa"
    SINGLE_IMAGE_GROUNDING = "single_image_grounding"
    BI_TEMPORAL_CHANGE = "bi_temporal_change"
    CROSS_MODAL_SAR_OPTICAL = "cross_modal_sar_optical"
    UNKNOWN = "unknown"


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def check_input_compatibility(
    filenames: List[str], 
    dimensions: List[Tuple[int, int]] = None
) -> Dict[str, Any]:
    validation_errors = []
    
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            validation_errors.append(f"Format '{ext}' not supported. Allowed: {list(SUPPORTED_EXTENSIONS)}")

    dim_match = True
    if dimensions and len(dimensions) == 2:
        if dimensions[0] != dimensions[1]:
            dim_match = False

    result = {
        "is_valid": len(validation_errors) == 0,
        "format_verified": len(validation_errors) == 0,
        "pair_dimension_matched": dim_match,
        "input_count": len(filenames),
        "errors": validation_errors,
    }

    log_step("input_compatibility_check", {
        "filenames": filenames,
        "dimensions": [f"{w}x{h}" for w, h in (dimensions or [])],
        "validation_result": result,
    })

    return result


def classify_task(
    question: str, 
    num_images: int, 
    task_hint: str = None
) -> Tuple[TaskType, List[str]]:
    q = question.lower()

    if task_hint == "bi_temporal_change" and num_images >= 2:
        task = TaskType.BI_TEMPORAL_CHANGE
    elif task_hint == "cross_modal_sar_optical" and num_images >= 2:
        task = TaskType.CROSS_MODAL_SAR_OPTICAL
    elif task_hint == "single_image_grounding":
        task = TaskType.SINGLE_IMAGE_GROUNDING
    elif num_images >= 2 and any(w in q for w in ["change", "differ", "between", "before", "after", "temporal", "increase", "decrease"]):
        task = TaskType.BI_TEMPORAL_CHANGE
    elif num_images >= 2 and any(w in q for w in ["sar", "radar", "fusion", "cross-modal", "optical and sar", "penetrat", "backscatter"]):
        task = TaskType.CROSS_MODAL_SAR_OPTICAL
    elif any(w in q for w in ["highlight", "where is", "locate", "grounding", "bounding box", "find the", "outline"]):
        task = TaskType.SINGLE_IMAGE_GROUNDING
    elif num_images == 1:
        task = TaskType.SINGLE_IMAGE_VQA
    elif num_images >= 2:
        task = TaskType.BI_TEMPORAL_CHANGE
    else:
        task = TaskType.UNKNOWN

    tools_selected = []
    if task == TaskType.SINGLE_IMAGE_VQA:
        tools_selected = ["rs_classifier", "gemini_reasoner"]
    elif task == TaskType.SINGLE_IMAGE_GROUNDING:
        tools_selected = ["rs_classifier", "region_grounder", "gemini_reasoner"]
    elif task == TaskType.BI_TEMPORAL_CHANGE:
        tools_selected = ["change_detector", "rs_classifier", "gemini_reasoner"]
    elif task == TaskType.CROSS_MODAL_SAR_OPTICAL:
        tools_selected = ["sar_optical_fusion", "gemini_reasoner"]

    log_step("router_orchestration_decision", {
        "question": question,
        "num_images": num_images,
        "chosen_task": task.value,
        "tools_sequenced": tools_selected,
        "tools_metadata": [
            {
                "tool_id": t_id,
                "version": REGISTRY[t_id].version,
                "output_type": REGISTRY[t_id].output_type,
            }
            for t_id in tools_selected if t_id in REGISTRY
        ],
    })

    return task, tools_selected
