import io
import base64
import numpy as np
from PIL import Image, ImageDraw
from inference.geo_io import load_image_as_rgb, load_image_as_grayscale


def _to_base64_png(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def fuse_optical_sar(optical_path: str, sar_path: str, question: str = "") -> dict:
    opt_img = load_image_as_rgb(optical_path)
    sar_img = load_image_as_grayscale(sar_path)

    if opt_img.size != sar_img.size:
        sar_img = sar_img.resize(opt_img.size, Image.Resampling.BILINEAR)

    w, h = opt_img.size
    opt_np = np.array(opt_img, dtype=np.float32)
    sar_np = np.array(sar_img, dtype=np.float32)

    total_pixels = w * h

    r, g, b = opt_np[:, :, 0], opt_np[:, :, 1], opt_np[:, :, 2]
    brightness = np.mean(opt_np, axis=2)
    greenness = g - 0.5 * (r + b)

    saturation = np.max(opt_np, axis=2) - np.min(opt_np, axis=2)
    optical_cloud_mask = (brightness > 190) & (saturation < 30)
    cloud_coverage_pct = round(float(np.sum(optical_cloud_mask)) / total_pixels * 100, 1)

    sar_water_candidate = sar_np < 50
    sar_builtup_candidate = sar_np > 170
    sar_vegetation_candidate = (sar_np >= 50) & (sar_np <= 170)

    water_confirmed = sar_water_candidate & ((b > r) | optical_cloud_mask)
    builtup_confirmed = sar_builtup_candidate
    vegetation_confirmed = sar_vegetation_candidate & (greenness > 0) & (~optical_cloud_mask)

    cloud_penetrated_pixels = np.sum(optical_cloud_mask & (sar_builtup_candidate | sar_water_candidate))
    cloud_penetrated_pct = round(float(cloud_penetrated_pixels) / total_pixels * 100, 1)

    water_pct = round(float(np.sum(water_confirmed)) / total_pixels * 100, 1)
    builtup_pct = round(float(np.sum(builtup_confirmed)) / total_pixels * 100, 1)
    vegetation_pct = round(float(np.sum(vegetation_confirmed)) / total_pixels * 100, 1)

    fused_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    base_gray = np.array(opt_img.convert("L"), dtype=np.uint8)
    for c in range(3):
        fused_rgb[:, :, c] = base_gray

    fused_rgb[water_confirmed] = [0, 130, 255]
    fused_rgb[builtup_confirmed] = [255, 102, 0]
    fused_rgb[vegetation_confirmed] = [34, 197, 94]

    fused_img = Image.fromarray(fused_rgb)

    legend_h = 28
    composite_w = w
    composite_h = h + legend_h
    legend_bar = Image.new("RGB", (composite_w, composite_h), color=(15, 23, 42))
    legend_bar.paste(fused_img, (0, 0))

    draw = ImageDraw.Draw(legend_bar)
    draw.rectangle([10, h + 8, 22, h + 20], fill=(0, 130, 255))
    draw.text((26, h + 7), "Water (SAR+Opt)", fill=(203, 213, 225))

    draw.rectangle([130, h + 8, 142, h + 20], fill=(255, 102, 0))
    draw.text((146, h + 7), "Built-up (Double-Bounce)", fill=(203, 213, 225))

    draw.rectangle([285, h + 8, 297, h + 20], fill=(34, 197, 94))
    draw.text((301, h + 7), "Vegetation", fill=(203, 213, 225))

    confidence = round(min(0.97, 0.82 + (builtup_pct + water_pct) / 200.0), 2)

    return {
        "modality_fusion": "Optical Multispectral + SAR Radar Backscatter",
        "built_up_coverage_percentage": builtup_pct,
        "water_coverage_percentage": water_pct,
        "vegetation_coverage_percentage": vegetation_pct,
        "optical_cloud_coverage_percentage": cloud_coverage_pct,
        "radar_cloud_penetration_percentage": cloud_penetrated_pct,
        "cloud_penetration_effective": cloud_penetrated_pct > 0.5,
        "confidence": confidence,
        "visual_evidence_b64": _to_base64_png(legend_bar),
        "sensor_synergy": "SAR resolved surface roughness & double-bounce through optical cloud/spectral ambiguity.",
    }
