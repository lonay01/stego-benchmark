"""CPU-only attack worker; its environment is independent of every decoder."""

import io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .spec import seed


def pixels(array):
    return np.rint(np.clip(array, 0, 255)).astype(np.uint8)


def jpeg(array, quality):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "JPEG", quality=int(quality), subsampling=2,
                               optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.array(image.convert("RGB"))


def resize(array, scale):
    image = Image.fromarray(array)
    w, h = image.size
    return np.array(image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC))


def rotate(array, angle):
    return np.array(Image.fromarray(array).rotate(angle, Image.Resampling.BICUBIC,
                                                  expand=False, fillcolor=(0, 0, 0)))


def apply(array, attack, image_id, replicate):
    """Return RGB8 pixels and actual parameters, not just nominal severity."""
    array = np.asarray(array)
    h, w, channels = array.shape
    if array.dtype != np.uint8 or channels != 3 or max(w, h) > 512:
        raise ValueError("Expected native RGB8 carrier of at most 512x512")
    rng_seed = seed(image_id, replicate, attack["id"])
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    family, value = attack["family"], attack.get("value")
    params = {**attack, "seed": rng_seed, "width": w, "height": h}

    def noise(image, sigma):
        field = rng.standard_normal((512, 512, 3))
        top, left = (512 - h) // 2, (512 - w) // 2
        return pixels(image.astype(np.float64) + 255 * sigma * field[top:top+h, left:left+w])

    if family == "clean":
        out = array.copy()
    elif family == "jpeg":
        out = jpeg(array, value)
    elif family == "noise":
        out = noise(array, value)
    elif family == "blur":
        out = pixels(gaussian_filter(array.astype(np.float64), (value, value, 0),
                                    mode="reflect", truncate=3))
    elif family == "brightness":
        out = pixels(array.astype(np.float64) * value)
    elif family == "resize":
        out = resize(array, value)
    elif family == "rotation":
        out = rotate(array, value)
    elif family == "crop":
        cw, ch = max(1, round(w * value**.5)), max(1, round(h * value**.5))
        u, v = rng.random(2)
        left, top = min(w-cw, int(u*(w-cw+1))), min(h-ch, int(v*(h-ch+1)))
        params.update(u=float(u), v=float(v), box=[left, top, left+cw, top+ch],
                      actual_area=cw*ch/(w*h))
        out = np.array(Image.fromarray(array).crop(params["box"]).resize((w, h), Image.Resampling.BICUBIC))
    elif family == "compound":
        low, high, degrees, scale, sigma, quality = [
            (.9, 1.1, 2, .9, .005, 90), (.85, 1.15, 5, .75, .01, 70),
            (.75, 1.25, 10, .5, .02, 50), (.6, 1.4, 15, .25, .04, 30),
        ][value-1]
        brightness, angle = float(rng.uniform(low, high)), float(rng.uniform(-degrees, degrees))
        params.update(brightness=brightness, angle=angle, scale=scale, sigma=sigma, jpeg=quality)
        out = jpeg(noise(resize(rotate(pixels(array.astype(float)*brightness), angle), scale), sigma), quality)
    else:
        raise ValueError(f"Unknown attack: {family}")
    return out, params


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "PNG", compress_level=3)
    return buffer.getvalue()
