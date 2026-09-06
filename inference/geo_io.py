"""
geo_io.py

Shared GeoTIFF-safe image loader for SatQuery AI.
Supports multi-band GeoTIFF, float/16-bit reflectance, and standard PNG/JPEG images.
Uses rasterio with 2-98th percentile contrast stretching when available, with PIL.Image fallback.
"""

import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _normalize_percentile(arr: np.ndarray, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
    """Normalize array using percentile contrast stretching to 0-255 uint8."""
    if arr.size == 0:
        return arr.astype(np.uint8)

    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return np.zeros_like(arr, dtype=np.uint8)

    valid_vals = arr[valid_mask]
    low, high = np.percentile(valid_vals, (p_min, p_max))
    if high <= low:
        high = low + 1e-5

    normalized = np.clip((arr - low) / (high - low) * 255.0, 0, 255)
    return normalized.astype(np.uint8)


def load_image_as_rgb(image_path: str) -> Image.Image:
    """Load an image file (GeoTIFF/PNG/JPEG) and return a 3-channel RGB PIL Image."""
    rasterio_error = None
    if HAS_RASTERIO:
        try:
            with rasterio.open(image_path) as src:
                band_count = src.count
                nodata = src.nodata

                if band_count >= 3:
                    arr = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)
                    arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
                elif band_count == 2:
                    # Not enough bands for true RGB -- duplicate band 1 into the
                    # missing channels rather than dropping data silently.
                    b1 = src.read(1).astype(np.float32)
                    b2 = src.read(2).astype(np.float32)
                    arr = np.stack([b1, b2, b1], axis=-1)
                else:
                    b1 = src.read(1).astype(np.float32)
                    arr = np.stack([b1] * 3, axis=-1)

                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)

                if arr.dtype != np.uint8:
                    arr = _normalize_percentile(arr)

                return Image.fromarray(arr, mode="RGB")
        except Exception as e:
            # Keep the reason instead of silently discarding it -- if the
            # PIL fallback below also fails, the caller needs to know *why*
            # the GeoTIFF couldn't be decoded rather than getting a bare
            # "cannot identify image file" from PIL.
            rasterio_error = e

    try:
        img = Image.open(image_path)
        return img.convert("RGB")
    except Exception as pil_error:
        if rasterio_error is not None:
            raise ValueError(
                f"Could not decode this GeoTIFF. rasterio failed with: {rasterio_error}; "
                f"PIL fallback also failed with: {pil_error}. This usually means the file "
                f"uses a band layout, compression, or dtype that isn't supported here."
            ) from pil_error
        raise


def load_image_as_grayscale(image_path: str) -> Image.Image:
    """Load an image file (GeoTIFF/PNG/JPEG) and return a 1-channel Grayscale PIL Image."""
    if HAS_RASTERIO:
        try:
            with rasterio.open(image_path) as src:
                arr = src.read(1)
                if arr.dtype != np.uint8:
                    arr = _normalize_percentile(arr)

                return Image.fromarray(arr, mode="L")
        except Exception:
            pass

    img = Image.open(image_path)
    return img.convert("L")