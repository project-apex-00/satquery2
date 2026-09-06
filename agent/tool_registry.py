from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class SpecialistTool:
    tool_id: str
    name: str
    version: str
    description: str
    supported_modalities: List[str]
    supported_formats: List[str]
    permitted_parameters: Dict[str, Any]
    output_type: str


REGISTRY: Dict[str, SpecialistTool] = {
    "rs_classifier": SpecialistTool(
        tool_id="rs_classifier",
        name="RS-EuroSAT Specialist Classifier",
        version="1.0-clip-finetuned",
        description="Fine-tuned remote-sensing CLIP vision classifier for 10 Sentinel-2 land cover categories.",
        supported_modalities=["optical", "multispectral"],
        supported_formats=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        permitted_parameters={"device": "cpu", "return_all_probs": True},
        output_type="classification",
    ),
    "region_grounder": SpecialistTool(
        tool_id="region_grounder",
        name="Text-Guided Remote Sensing Region Grounder",
        version="2.0-spatial-mask",
        description="Extracts spatial bounding boxes and pixel highlight masks for queried land-cover features.",
        supported_modalities=["optical", "multispectral", "sar"],
        supported_formats=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        permitted_parameters={"confidence_threshold": 0.5, "return_overlay": True},
        output_type="grounding_bbox",
    ),
    "change_detector": SpecialistTool(
        tool_id="change_detector",
        name="Bi-Temporal Change Vector Analyzer",
        version="2.1-cva-spectral",
        description="Analyzes co-registered bi-temporal image pairs (T1, T2) to quantify land-cover shifts and render change heatmaps.",
        supported_modalities=["bitemporal_optical", "bitemporal_pair"],
        supported_formats=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        permitted_parameters={"statistical_threshold": 0.65, "generate_legend": True},
        output_type="change_map",
    ),
    "sar_optical_fusion": SpecialistTool(
        tool_id="sar_optical_fusion",
        name="Optical-SAR Cross-Modal Fusion Engine",
        version="1.5-microwave-optical",
        description="Jointly analyzes optical reflectance and SAR microwave backscatter to discriminate structures and water through cloud cover.",
        supported_modalities=["cross_modal_pair", "optical_plus_sar"],
        supported_formats=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        permitted_parameters={"calibrate_backscatter": True, "generate_composite": True},
        output_type="fusion_map",
    ),
    "gemini_reasoner": SpecialistTool(
        tool_id="gemini_reasoner",
        name="Domain-Adapted Gemini Reasoning Agent",
        version="1.5-flash",
        description="Synthesizes structured specialist outputs into natural-language answers grounded strictly in specialist evidence.",
        supported_modalities=["text_and_structured_evidence"],
        supported_formats=[".json"],
        permitted_parameters={"temperature": 0.2, "grounding_strictness": "high"},
        output_type="natural_language_answer",
    ),
}


def get_tool(tool_id: str) -> SpecialistTool:
    return REGISTRY.get(tool_id)


def list_registry_summary() -> List[Dict[str, Any]]:
    return [
        {
            "tool_id": t.tool_id,
            "name": t.name,
            "version": t.version,
            "modalities": t.supported_modalities,
            "output_type": t.output_type,
        }
        for t in REGISTRY.values()
    ]
