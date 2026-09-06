import io
import base64
import numpy as np
from PIL import Image, ImageDraw
from inference.geo_io import load_image_as_rgb


def _to_base64_png(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def detect_changes(path_t1: str, path_t2: str, question: str = "") -> dict:
    img_t1 = load_image_as_rgb(path_t1)
    img_t2 = load_image_as_rgb(path_t2)

    if img_t1.size != img_t2.size:
        img_t2 = img_t2.resize(img_t1.size, Image.Resampling.BILINEAR)

    w, h = img_t1.size
    t1_np = np.array(img_t1, dtype=np.float32)
    t2_np = np.array(img_t2, dtype=np.float32)

    diff_vector = np.sqrt(np.sum((t2_np - t1_np) ** 2, axis=2))
    
    mean_diff = np.mean(diff_vector)
    std_diff = np.std(diff_vector)
    change_threshold = max(30.0, mean_diff + 0.65 * std_diff)
    change_mask = diff_vector > change_threshold

    greenness_t1 = t1_np[:, :, 1] - 0.5 * (t1_np[:, :, 0] + t1_np[:, :, 2])
    greenness_t2 = t2_np[:, :, 1] - 0.5 * (t2_np[:, :, 0] + t2_np[:, :, 2])
    delta_green = greenness_t2 - greenness_t1

    brightness_t1 = np.mean(t1_np, axis=2)
    brightness_t2 = np.mean(t2_np, axis=2)
    delta_brightness = brightness_t2 - brightness_t1

    veg_loss_mask = change_mask & (delta_green < -15)
    veg_gain_mask = change_mask & (delta_green > 15)
    builtup_gain_mask = change_mask & (delta_brightness > 20) & (delta_green <= 5)
    other_change_mask = change_mask & (~veg_loss_mask) & (~veg_gain_mask) & (~builtup_gain_mask)

    total_pixels = w * h
    total_change_pct = round(float(np.sum(change_mask)) / total_pixels * 100, 2)
    veg_loss_pct = round(float(np.sum(veg_loss_mask)) / total_pixels * 100, 2)
    veg_gain_pct = round(float(np.sum(veg_gain_mask)) / total_pixels * 100, 2)
    builtup_gain_pct = round(float(np.sum(builtup_gain_mask)) / total_pixels * 100, 2)
    unchanged_pct = round(100.0 - total_change_pct, 2)

    gray_bg = img_t2.convert("L").convert("RGB")
    heatmap_np = np.array(gray_bg, dtype=np.uint8)

    heatmap_np[veg_loss_mask] = [239, 68, 68]
    heatmap_np[builtup_gain_mask] = [245, 158, 11]
    heatmap_np[veg_gain_mask] = [34, 197, 94]
    heatmap_np[other_change_mask] = [168, 85, 247]

    heatmap_img = Image.fromarray(heatmap_np)

    legend_h = 28
    composite_w = w
    composite_h = h + legend_h
    legend_bar = Image.new("RGB", (composite_w, composite_h), color=(15, 23, 42))
    legend_bar.paste(heatmap_img, (0, 0))

    draw = ImageDraw.Draw(legend_bar)
    draw.rectangle([10, h + 8, 22, h + 20], fill=(239, 68, 68))
    draw.text((26, h + 7), "Veg Loss", fill=(203, 213, 225))

    draw.rectangle([95, h + 8, 107, h + 20], fill=(245, 158, 11))
    draw.text((111, h + 7), "Built-up Gain", fill=(203, 213, 225))

    draw.rectangle([200, h + 8, 212, h + 20], fill=(34, 197, 94))
    draw.text((216, h + 7), "Veg Growth", fill=(203, 213, 225))

    if builtup_gain_pct > veg_loss_pct and builtup_gain_pct > 2.0:
        dominant_trend = "Urban / Built-up Expansion"
    elif veg_loss_pct > 2.0:
        dominant_trend = "Vegetation Reduction / Clearing"
    elif veg_gain_pct > 2.0:
        dominant_trend = "Vegetation Growth / Reforestation"
    elif total_change_pct > 5.0:
        dominant_trend = "General Surface Land-Cover Shift"
    else:
        dominant_trend = "Stable / Unchanged Surface"

    confidence = round(min(0.98, 0.78 + (total_change_pct / 150.0)), 2)

    return {
        "dominant_trend": dominant_trend,
        "total_change_percentage": total_change_pct,
        "unchanged_percentage": unchanged_pct,
        "vegetation_loss_percentage": veg_loss_pct,
        "vegetation_gain_percentage": veg_gain_pct,
        "built_up_gain_percentage": builtup_gain_pct,
        "confidence": confidence,
        "status": "Significant Alteration" if total_change_pct > 5.0 else "Minimal Change",
        "visual_evidence_b64": _to_base64_png(legend_bar),
        "dimensions": f"{w}x{h} px",
    }
