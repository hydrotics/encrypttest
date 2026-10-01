"""Robust keyed watermarking with multi-hypothesis registration and video frame search.

Embedding intentionally keeps the same carrier/energy model as the previous robust
version, so existing personalized reveals remain traceable without increasing visibility.
The extractor is substantially stronger: linear tone normalization, SIFT/affine/homography
registration, sub-pixel geometric hypotheses, temporal search, and evidence fusion.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import itertools
import re
import subprocess
import tempfile
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

N_BITS = 96
CELL = 16
AMP = 4.0
VIDEO_GROUP = 8
MATCH_H = 360
COARSE_H = 240
VIDEO_MAX_FPS = 30.0
TRACE_COARSE_FPS = 1.0
TRACE_MAX_SAMPLES = 180

_x, _y = np.mgrid[:CELL, :CELL].astype(np.float32)
_CARRIER = (
    np.cos(np.pi * (_y + .5) / CELL) * np.cos(np.pi * 2 * (_x + .5) / CELL)
    + .8 * np.cos(np.pi * 2 * (_y + .5) / CELL) * np.cos(np.pi * (_x + .5) / CELL)
).astype(np.float32)
_CARRIER /= np.sqrt(np.mean(_CARRIER * _CARRIER)) + 1e-8
RGB2Y = np.array([0.299, 0.587, 0.114], np.float32)

# Video rate control. Spending a whole upload budget on a short clip makes the file
# huge (slow to encode, upload and load on the client) without any visible gain, so the
# average bitrate is capped at MAX_BITS_PER_PIXEL (bits per pixel per frame).
MAX_BITS_PER_PIXEL = 0.11
MIN_VIDEO_BPS = 350_000
VBV_SECONDS = 1.5          # VBV buffer length; also reserved from the size budget
CONTAINER_MARGIN = 0.975   # fraction of the size budget the stream may use (mux overhead)

# Images are watermarked in horizontal strips of this many cell rows so peak memory
# stays flat instead of scaling with float32 copies of the whole picture.
IMAGE_STRIP_CELLS = 24


def get_ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_payload(user_id: int) -> np.ndarray:
    uid = int(user_id) & ((1 << 64) - 1)
    crc = zlib.crc32(uid.to_bytes(8, "big"))
    value = (uid << 32) | crc
    return np.array(
        [(value >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS)],
        np.float32,
    ) * 2.0 - 1.0


def decode_payload(scores: np.ndarray) -> tuple[int, bool]:
    """Decode the legacy 96-bit payload, correcting a few weak bit errors via CRC."""
    bits = (np.asarray(scores) > 0).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    uid = value >> 32
    if zlib.crc32(uid.to_bytes(8, "big")) == (value & 0xFFFFFFFF):
        return uid, True

    weak = np.argsort(np.abs(np.asarray(scores)))[:12]
    if len(weak) < 2:
        return uid, False

    # Old embeds stay byte-for-byte compatible: this only changes extraction.
    for flips in (1, 2, 3):
        for idxs in itertools.combinations(weak.tolist(), flips):
            test = bits.copy()
            test[list(idxs)] ^= 1
            value = 0
            for bit in test:
                value = (value << 1) | int(bit)
            uid = value >> 32
            if zlib.crc32(uid.to_bytes(8, "big")) == (value & 0xFFFFFFFF):
                return uid, True
    return uid, False


def _seed(key: bytes, *parts) -> int:
    message = ":".join(map(str, parts)).encode()
    return int.from_bytes(hmac.new(key, message, hashlib.sha256).digest()[:16], "big")


@lru_cache(maxsize=32)
def _layout(key: bytes, reveal_id: str, h: int, w: int) -> np.ndarray:
    ch, cw = -(-h // CELL), -(-w // CELL)
    rng = np.random.default_rng(_seed(key, reveal_id, "layout", ch, cw))
    return (rng.permutation(ch * cw).reshape(ch, cw) % N_BITS).astype(np.int16)


def _chips(key: bytes, reveal_id: str, group: int, ch: int, cw: int) -> np.ndarray:
    rng = np.random.default_rng(_seed(key, reveal_id, "chips", group))
    return rng.choice(np.array([-1, 1], np.float32), (ch, cw))


def _grid(h: int, w: int) -> tuple[int, int]:
    return -(-h // CELL), -(-w // CELL)


def _pad(a: np.ndarray, h: int, w: int) -> np.ndarray:
    ph, pw = h - a.shape[0], w - a.shape[1]
    return np.pad(a, ((0, ph), (0, pw)), mode="edge") if ph or pw else a


def _texture(y: np.ndarray) -> np.ndarray:
    ch, cw = _grid(*y.shape)
    p = _pad(y, ch * CELL, cw * CELL)
    b = p.reshape(ch, CELL, cw, CELL).transpose(0, 2, 1, 3)
    return (0.45 + 0.9 * np.clip(b.std((2, 3)) / 18.0, 0, 1)).astype(np.float32)


def _delta(y: np.ndarray, bits: np.ndarray, key: bytes, reveal_id: str, group: int, amp: float) -> np.ndarray:
    h, w = y.shape
    ch, cw = _grid(h, w)
    cells = (
        bits[_layout(key, reveal_id, h, w)]
        * _chips(key, reveal_id, group, ch, cw)
        * _texture(y)
    )
    d = cells[:, :, None, None] * _CARRIER[None, None] * float(amp)
    return d.transpose(0, 2, 1, 3).reshape(ch * CELL, cw * CELL)[:h, :w]


def _tone_match(orig: np.ndarray, leak: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Remove global screenshot brightness/contrast shifts before watermark correlation."""
    m = valid > 0.35
    if int(m.sum()) < 500:
        return leak
    x = orig[m].astype(np.float64)
    y = leak[m].astype(np.float64)
    # Robust trimming keeps subtitles/UI and clipped pixels from dominating the fit.
    lo_x, hi_x = np.percentile(x, [2, 98])
    lo_y, hi_y = np.percentile(y, [2, 98])
    keep = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    x, y = x[keep], y[keep]
    if x.size < 500 or np.var(x) < 1e-3:
        return leak
    a = float(np.cov(x, y, bias=True)[0, 1] / max(np.var(y), 1e-6))
    b = float(x.mean() - a * y.mean())
    a = float(np.clip(a, 0.65, 1.6))
    return (a * leak.astype(np.float32) + b).astype(np.float32)


def _score(orig: np.ndarray, leak: np.ndarray, key: bytes, reveal_id: str, group: int,
           valid: np.ndarray | None = None) -> np.ndarray:
    """Correlate the known 16x16 carrier after tone normalization and block detrending."""
    h, w = orig.shape
    ch, cw = _grid(h, w)
    if valid is None:
        valid = np.ones_like(orig, np.float32)
    valid = np.clip(valid.astype(np.float32), 0, 1)
    leak = _tone_match(orig, leak, valid)
    resid = leak - orig.astype(np.float32)

    p = _pad(resid, ch * CELL, cw * CELL)
    vm = _pad(valid, ch * CELL, cw * CELL)
    b = p.reshape(ch, CELL, cw, CELL).transpose(0, 2, 1, 3)
    vb = vm.reshape(ch, CELL, cw, CELL).transpose(0, 2, 1, 3)

    # Remove block DC and a very low-frequency plane. This makes screenshot/re-encode
    # lighting changes much less influential while preserving the watermark carrier.
    wsum = vb.sum((2, 3), keepdims=True)
    mean = (b * vb).sum((2, 3), keepdims=True) / np.maximum(wsum, 1e-5)
    b = b - mean
    corr = (b * _CARRIER[None, None] * vb).sum((2, 3))
    den = np.sqrt(
        (vb * (_CARRIER[None, None] ** 2)).sum((2, 3)) *
        (vb * (b ** 2)).sum((2, 3)) + 1e-5
    )
    corr = corr / np.maximum(den, 1e-4)
    coverage = np.clip(vb.mean((2, 3)), 0, 1)
    vals = corr * _chips(key, reveal_id, group, ch, cw) * coverage
    vals[coverage < 0.25] = 0.0

    # Do not clip around the signed median: roughly half the watermark chips are
    # negative by design, so a signed median can sit at -1 and destroy every positive bit.
    # Clip only extreme magnitudes symmetrically, preserving the sign information.
    flat = np.abs(vals[coverage >= 0.25])
    if flat.size > 32:
        med = np.median(flat)
        mad = np.median(np.abs(flat - med)) + 1e-6
        limit = max(1.0, med + 8.0 * mad)
        vals = np.clip(vals, -limit, limit)

    return np.bincount(_layout(key, reveal_id, h, w).ravel(), vals.ravel(), minlength=N_BITS)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _feature_image(rgb: np.ndarray, height: int = MATCH_H) -> tuple[np.ndarray, float]:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb.astype(np.uint8)
    scale = min(1.0, height / max(g.shape))
    if scale < 1.0:
        g = cv2.resize(g, (max(32, round(g.shape[1] * scale)), max(32, round(g.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return g, scale


def _sift(g: np.ndarray, nfeatures: int = 1000):
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.018, edgeThreshold=16)
    return sift.detectAndCompute(g, None)


def _homography_candidates(orig_rgb: np.ndarray, leak_rgb: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
    og, os = _feature_image(orig_rgb)
    lg, ls = _feature_image(leak_rgb)
    okp, od = _sift(og, 1600)
    lkp, ld = _sift(lg, 1600)
    if od is None or ld is None or len(okp) < 10 or len(lkp) < 10:
        return []

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ld, od, k=2)
    out: list[tuple[np.ndarray, float, str]] = []
    So = np.diag([1 / os, 1 / os, 1.0])
    Sl = np.diag([ls, ls, 1.0])

    for ratio in (0.72, 0.78, 0.84):
        good = [a for a, b in knn if a.distance < ratio * b.distance]
        if len(good) < 8:
            continue
        src = np.float32([lkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([okp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        for threshold in (2.5, 4.0, 7.0):
            Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold, maxIters=5000, confidence=0.998)
            if Hs is not None:
                inliers = int(mask.sum()) if mask is not None else 0
                if inliers >= 7:
                    out.append((So @ Hs @ Sl, inliers / max(len(good), 1),
                                f"sift-{ratio:.2f}-r{threshold:g}"))

        A, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
            maxIters=5000, confidence=0.998
        )
        if A is not None and mask is not None and int(mask.sum()) >= 7:
            Ah = np.vstack([A, [0, 0, 1]]).astype(np.float64)
            out.append((So @ Ah @ Sl, int(mask.sum()) / max(len(good), 1),
                        f"affine-{ratio:.2f}"))

    # ORB fallback helps when SIFT is unavailable/weak after heavy recompression.
    if not out:
        okp, od = _orb(og, 1400)
        lkp, ld = _orb(lg, 1400)
        if od is not None and ld is not None and len(okp) >= 10 and len(lkp) >= 10:
            knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ld, od, k=2)
            good = [a for a, b in knn if a.distance < 0.80 * b.distance]
            if len(good) >= 8:
                src = np.float32([lkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([okp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0, maxIters=5000, confidence=0.995)
                if Hs is not None:
                    inliers = int(mask.sum()) if mask is not None else 0
                    if inliers >= 7:
                        out.append((So @ Hs @ Sl, inliers / max(len(good), 1), "orb"))

    unique: list[tuple[np.ndarray, float, str]] = []
    for item in sorted(out, key=lambda x: x[1], reverse=True):
        H = item[0]
        if any(np.max(np.abs(H - u[0])) < 0.015 * max(1.0, np.max(np.abs(H))) for u in unique):
            continue
        unique.append(item)
        if len(unique) >= 8:
            break
    return unique


def _template_fallback(orig_rgb: np.ndarray, leak_rgb: np.ndarray) -> Optional[tuple[np.ndarray, float, str]]:
    og, os = _feature_image(orig_rgb, 480)
    lg, ls = _feature_image(leak_rgb, 480)
    best = None
    for sx in np.arange(0.55, 1.21, 0.05):
        for sy in (sx, sx * 0.92, sx * 1.08):
            tw, th = round(lg.shape[1] * sx), round(lg.shape[0] * sy)
            if min(tw, th) < 48 or tw > og.shape[1] or th > og.shape[0]:
                continue
            templ = cv2.resize(lg, (tw, th), interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(og, templ, cv2.TM_CCOEFF_NORMED)
            _, sc, _, loc = cv2.minMaxLoc(r)
            item = (float(sc), sx, sy, loc)
            if best is None or item[0] > best[0]:
                best = item
    if best is None:
        return None
    sc, sx, sy, (x, y) = best
    H = np.array([[ls * sx / os, 0, x / os], [0, ls * sy / os, y / os], [0, 0, 1]], np.float64)
    return H, sc, "template"


def _registration_candidates(orig_rgb: np.ndarray, leak_rgb: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
    out = _homography_candidates(orig_rgb, leak_rgb)
    if not out:
        t = _template_fallback(orig_rgb, leak_rgb)
        if t is not None:
            out = [t]
    return out


def _warp(leak: np.ndarray, H: np.ndarray, out_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = out_shape
    valid = cv2.warpPerspective(np.ones(leak.shape, np.float32), H, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    warped = cv2.warpPerspective(leak.astype(np.float32), H, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT)
    return warped, valid


def _geometry_variants(H: np.ndarray) -> list[np.ndarray]:
    """Registration jitter for crop, rescale, rotation and encoder rounding."""
    out = [H.copy()]
    for tx in (-2.0, -1.0, -0.35, 0.35, 1.0, 2.0):
        for ty in (-2.0, -1.0, 0.0, 1.0, 2.0):
            M = H.copy()
            M[0, 2] += tx
            M[1, 2] += ty
            out.append(M)
    for sx, sy in ((0.995, 1), (1.005, 1), (1, 0.995), (1, 1.005),
                   (0.99, 0.99), (1.01, 1.01), (0.985, 1.015), (1.015, 0.985)):
        M = H.copy()
        M[0, :3] *= sx
        M[1, :3] *= sy
        out.append(M)
    for deg in (-1.0, -0.5, 0.5, 1.0):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
        out.append(R @ H)
    return out


def _decode_registered(oy: np.ndarray, leak_y: np.ndarray, key: bytes, reveal_id: str,
                       group: int, Hs: list[np.ndarray]) -> tuple[Optional[int], float, np.ndarray]:
    best_metric = -1e9
    best_scores = np.zeros(N_BITS, np.float64)
    votes: dict[int, list[float]] = {}
    for H in Hs:
        warped, mask = _warp(leak_y, H, oy.shape)
        for candidate in (warped, cv2.GaussianBlur(warped, (3, 3), 0)):
            scores = _score(oy, candidate, key, reveal_id, group, mask)
            uid, ok = decode_payload(scores)
            metric = float(np.mean(np.abs(scores)))
            if metric > best_metric:
                best_metric, best_scores = metric, scores
            if ok:
                votes.setdefault(int(uid), []).append(metric)

    if votes:
        uid, metrics = max(votes.items(), key=lambda item: (len(item[1]), sum(item[1])))
        # A CRC hit is already strong; requiring two independent hypotheses only when
        # several competing IDs appear avoids false positives without hurting real leaks.
        if len(votes) == 1 or len(metrics) >= 2:
            return uid, float(np.mean(metrics)), best_scores
    return None, best_metric, best_scores


def embed_image_array(rgb: np.ndarray, user_id: int, key: bytes, reveal_id: str,
                      amp: float = AMP) -> np.ndarray:
    """Watermark an (h, w, 3) uint8 image and return a new uint8 array.

    Works in strips of IMAGE_STRIP_CELLS cell-rows. Strips start on a cell boundary and
    every cell is independent, so the result matches whole-image processing while peak
    memory stays small (the old code held several float32 copies of the full image,
    ~0.6 GB for 12 MP and over 1 GB for a 24 MP phone photo).
    """
    h, w = rgb.shape[:2]
    ch, cw = _grid(h, w)
    cell_bits = encode_payload(user_id)[_layout(key, reveal_id, h, w)] * _chips(key, reveal_id, 0, ch, cw)
    out = np.empty((h, w, 3), np.uint8)
    for r0 in range(0, ch, IMAGE_STRIP_CELLS):
        r1 = min(ch, r0 + IMAGE_STRIP_CELLS)
        y0, y1 = r0 * CELL, min(h, r1 * CELL)
        strip = rgb[y0:y1].astype(np.float32)
        cells = cell_bits[r0:r1] * _texture(strip @ RGB2Y)
        d = (cells[:, :, None, None] * _CARRIER[None, None] * float(amp))
        d = d.transpose(0, 2, 1, 3).reshape((r1 - r0) * CELL, cw * CELL)[: y1 - y0, :w]
        out[y0:y1] = np.clip(strip + d[..., None], 0, 255).astype(np.uint8)
    return out


def _save_image(img: Image.Image, dst: Path) -> None:
    suffix = dst.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.save(dst, "JPEG", quality=95, subsampling=0)
    elif suffix == ".webp":
        img.save(dst, "WEBP", lossless=True)
    else:
        img.save(dst, "PNG", compress_level=4)


def embed_image(src: Path, dst: Path, user_id: int, key: bytes, reveal_id: str,
                amp: float = AMP, cell: int = CELL) -> None:
    """Write a full-resolution watermarked copy of `src` to `dst`."""
    del cell  # fixed robust layout; compatibility with older bot configuration
    rgb = np.asarray(_load_rgb(src), np.uint8)
    _save_image(Image.fromarray(embed_image_array(rgb, user_id, key, reveal_id, amp)), dst)


def render_image_preview(src: Path, preview: Path, user_id: int, key: bytes, reveal_id: str, *,
                         amp: float = AMP, max_dim: int = 2048, quality: int = 88) -> None:
    """Watermark at full resolution, then write only the baseline JPEG that gets delivered.

    The watermark is embedded before downscaling (as before), but the full-resolution
    personalized copy is never encoded to disk: nothing consumed it, and PNG-encoding it
    was the single slowest step of an image reveal.
    """
    rgb = np.asarray(_load_rgb(src), np.uint8)
    img = Image.fromarray(embed_image_array(rgb, user_id, key, reveal_id, amp))
    del rgb
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    preview.parent.mkdir(parents=True, exist_ok=True)
    img.save(preview, "JPEG", quality=max(70, min(95, int(quality))),
             optimize=True, progressive=False, subsampling=2)


def extract_image(orig: Path, leak: Path, key: bytes, reveal_id: str, cell: int = CELL) -> dict:
    del cell
    o = np.asarray(_load_rgb(orig), np.uint8)
    l = np.asarray(_load_rgb(leak), np.uint8)
    oy = o @ RGB2Y
    ly = l @ RGB2Y
    candidates = _registration_candidates(o, l)
    if not candidates:
        raise RuntimeError("Could not register leaked image with the original.")
    best = None
    for H, reg, method in candidates:
        uid, metric, _ = _decode_registered(oy, ly, key, reveal_id, 0, _geometry_variants(H))
        if uid is not None:
            return {"valid": True, "user_id": uid, "registration": method,
                    "registration_score": round(reg, 3), "watermark_metric": round(metric, 4)}
        if best is None or metric > best[0]:
            best = (metric, reg, method)
    metric, reg, method = best
    return {"valid": False, "user_id": None, "registration": method,
            "registration_score": round(reg, 3), "watermark_metric": round(metric, 4)}


def _scale_filter(height: int, flags: str) -> str:
    """Cap height, keep aspect ratio, force even dimensions and square pixels."""
    return rf"scale=-2:trunc(min(ih\,{int(height)})/2)*2:flags={flags},setsar=1"


def _file_sig(path: Path) -> tuple[str, int, int]:
    """Cache key that changes whenever the file on disk changes."""
    p = Path(path)
    st = p.stat()
    return str(p), st.st_mtime_ns, st.st_size


@lru_cache(maxsize=64)
def _ffmpeg_info_cached(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size  # part of the cache key only
    return subprocess.run(
        [get_ffmpeg(), "-hide_banner", "-i", path],
        capture_output=True, text=True, errors="replace", timeout=30,
    ).stderr


def _ffmpeg_info(src: Path) -> str:
    """`ffmpeg -i` banner text (streams, fps, duration), probed once per file."""
    return _ffmpeg_info_cached(*_file_sig(src))


@lru_cache(maxsize=64)
def _probe_video_cached(path: str, mtime_ns: int, size: int, max_height: int):
    info = _ffmpeg_info_cached(path, mtime_ns, size)
    fps = None
    for pattern in (
        r"(\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?\s*fps",
        r"(\d+(?:\.\d+)?)\s*tbr",
    ):
        match = re.search(pattern, info)
        if not match:
            continue
        numerator = float(match.group(1))
        denominator = float(match.group(2)) if match.group(2) else 1.0
        if numerator > 0 and denominator > 0:
            fps = numerator / denominator
            break
    fps = min(max(fps or 30.0, 1.0), 60.0)
    # Normalize high-FPS sources. The same normalized FPS is used when tracing.
    fps = min(fps, VIDEO_MAX_FPS)
    max_height = max(144, int(max_height))
    vf = f"fps={fps:.6f}," + _scale_filter(max_height, "lanczos")
    probe = subprocess.run(
        [get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path, "-vf", vf,
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=45,
    )
    if not probe.stdout:
        details = probe.stderr.decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"FFmpeg could not decode the video. {details}".strip())
    with Image.open(io.BytesIO(probe.stdout)) as im:
        w, h = im.size
    return w, h, fps, vf


def _probe_video(src: Path, max_height: int):
    """Probe and normalize video settings used by both embed and extraction.

    Results are cached per (file, height): embedding used to re-probe the same file
    five or more times, each probe launching ffmpeg and decoding a frame.
    """
    return _probe_video_cached(*_file_sig(src), int(max_height))


def video_info(src: Path, max_height: int = 1080) -> dict:
    """Validate a video and return basic facts. Raises RuntimeError if undecodable."""
    w, h, fps, _ = _probe_video(src, max_height)
    return {"width": w, "height": h, "fps": round(fps, 3), "duration": round(_video_duration_seconds(src), 2)}


def _read_full(stream, buf) -> bool:
    view, got = memoryview(buf), 0
    while got < len(buf):
        n = stream.readinto(view[got:])
        if not n:
            return False
        got += n
    return True


def _iter_frames(src: Path, vf: str, w: int, h: int, *, skip: int = 0,
                 limit: Optional[int] = None, max_seconds: Optional[int] = None) -> Iterator[tuple[int, bytearray]]:
    size = w * h * 3 // 2
    cmd = [get_ffmpeg(), "-hide_banner", "-loglevel", "error"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", str(src), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size * 2)
    buf = bytearray(size)
    idx = 0
    try:
        while limit is None or idx - skip < limit:
            if not _read_full(proc.stdout, buf):
                break
            if idx >= skip:
                yield idx, buf
            idx += 1
    finally:
        proc.kill(); proc.wait(); proc.stdout.close()


def _video_duration_seconds(src: Path) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", _ffmpeg_info(src))
    if not m:
        return 0.0
    return float(m.group(1)) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3))


def _adaptive_video_height(source_width: int, source_height: int, duration: float, requested_max: int,
                           target_bytes: Optional[int], audio_bps: int, fps: float) -> int:
    """Choose the highest useful resolution that the attachment budget can sustain.

    Discord attachment delivery is size-limited. Keeping 1080p on a long clip while
    starving it of bitrate is what creates the visibly soft result we want to avoid.
    Estimate the available video bitrate and choose the highest resolution whose
    pixels/sec can be encoded at a sane H.264 rate.
    """
    ceiling = min(int(source_height), int(requested_max))
    source_width = max(2, int(source_width))
    if ceiling <= 360 or not target_bytes or duration <= 0:
        return max(144, ceiling)

    # Leave headroom for MP4/container overhead while preserving as much of the
    # attachment allowance as possible for video detail.
    usable_bits = max(1, int(target_bytes * 8 * 0.965) - int(audio_bps * duration))
    available_bps = usable_bits / max(duration, 1.0)
    test_fps = max(24.0, min(float(fps), 30.0))

    # Rough H.264 quality floor in bits/pixel/frame. High profile + B-frames is
    # substantially more efficient than the old constrained-baseline/no-B-frame path.
    min_bpp = 0.062
    for candidate in (1080, 900, 720, 648, 576, 540, 480, 360):
        if candidate > ceiling:
            continue
        candidate_width = max(2, round(source_width * candidate / max(source_height, 1)))
        required = candidate_width * candidate * test_fps * min_bpp
        if available_bps >= required:
            return max(144, int(candidate))
    return min(360, ceiling)


def _plan_video_rate(*, width: int, height: int, fps: float, duration: float,
                     target: Optional[int], audio_bps: int, max_bpp: float) -> int:
    """Average video bitrate (bps) for the encode.

    * With a size budget the rate is solved so the *worst case* stream (average rate for
      the whole clip plus a full VBV buffer) still fits, so the first pass is accepted
      instead of being thrown away and re-encoded.
    * The rate is also capped at ``max_bpp`` bits/pixel/frame: past that point H.264 gets
      bigger, not better, and short clips would otherwise balloon to the full upload limit.
    """
    if target:
        budget_bits = target * 8 * CONTAINER_MARGIN - audio_bps * duration
        rate = budget_bits / (max(duration, 1.0) + VBV_SECONDS)
    elif height >= 900:
        rate = 9_000_000
    elif height >= 700:
        rate = 6_500_000
    else:
        rate = 4_000_000
    if max_bpp > 0:
        rate = min(rate, width * height * max(fps, 1.0) * max_bpp)
    return max(MIN_VIDEO_BPS, int(rate))


def _delta_planes(y: np.ndarray, bits: np.ndarray, key: bytes, reveal_id: str,
                  group: int, amp: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-group watermark as (add, subtract) uint8 planes for saturating arithmetic.

    ``clip(y + d, 0, 255)`` is exactly ``subtract(add(y, max(d, 0)), max(-d, 0))`` with
    saturation, and the OpenCV saturating ops are in-place SIMD instead of the three
    full-frame int16 temporaries numpy needed per frame.
    """
    d = np.rint(_delta(y.astype(np.float32), bits, key, reveal_id, group, amp)).astype(np.int16)
    return (np.clip(d, 0, 255).astype(np.uint8), np.clip(-d, 0, 255).astype(np.uint8))


def embed_video(src: Path, dst: Path, user_id: int, key: bytes, reveal_id: str, *,
                amp: float = AMP, cell: int = CELL, preset: str = "veryfast",
                max_seconds: int = 300, max_height: int = 1080, group: int = VIDEO_GROUP,
                target_bytes: Optional[int] = None, audio_kbps: int = 96,
                max_bpp: float = MAX_BITS_PER_PIXEL) -> dict:
    """Embed a watermark into a Discord/mobile-friendly MP4.

    The encoder uses H.264 Baseline, yuv420p, no B-frames, short keyframe intervals,
    AAC audio and +faststart. x264 selects the H.264 level from the actual output
    dimensions, frame rate and bitrate; hard-coding Level 4.0 can mislabel wide
    videos and cause mobile hardware decoders to reject them. When a size budget is
    supplied the bitrate is derived from it (see ``_plan_video_rate``) and
    VBV-constrained, so the output reliably fits on the first pass. Returns a small
    dict describing the encode.
    """
    del cell
    duration = min(float(max_seconds), max(0.0, _video_duration_seconds(src)))
    if duration <= 0:
        duration = float(max_seconds)

    target = max(2 * 1048576, int(target_bytes)) if target_bytes else None
    bits = encode_payload(user_id)
    audio_bps = max(64000, min(128000, int(audio_kbps) * 1000))

    source_w, source_h, source_fps, _ = _probe_video(src, max_height)
    delivery_height = _adaptive_video_height(
        source_w, source_h, duration, max_height, target, audio_bps, source_fps
    )
    # Same height as the call above in the common case -> served from the probe cache.
    w, h, fps, vf = _probe_video(src, delivery_height)
    fps = min(float(fps), 30.0)
    gop = max(30, min(60, int(round(fps * 2.0))))
    rate = _plan_video_rate(width=w, height=h, fps=fps, duration=duration, target=target,
                            audio_bps=audio_bps, max_bpp=max_bpp)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def encode_once(out_path: Path, pass_rate: int) -> int:
        err = tempfile.TemporaryFile()
        enc = subprocess.Popen(
            [
                get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                "-framerate", f"{fps:.6f}", "-i", "-",
                # Audio-only second input, limited to the clip length: without -vn/-t
                # FFmpeg demuxes the entire source a second time just to find the audio.
                "-vn", "-t", f"{duration:.3f}", "-i", str(src),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", preset,
                "-b:v", str(pass_rate), "-maxrate", str(pass_rate),
                "-bufsize", str(int(pass_rate * VBV_SECONDS)),
                # Maximum-compatibility MP4 for Discord's native mobile attachment
                # player: H.264 Baseline, yuv420p, no B-frames, fixed 1-second GOPs,
                # closed GOPs, and the avc1 tag. Let x264 choose the required level:
                # forcing Level 4.0 mislabels preserved wide frames (for example,
                # 2560x1080) and can make mobile hardware decoders refuse playback.
                "-profile:v", "baseline",
                "-pix_fmt", "yuv420p", "-bf", "0", "-refs", "1",
                "-g", str(max(1, int(round(fps)))),
                "-keyint_min", str(max(1, int(round(fps)))),
                "-sc_threshold", "0",
                "-x264-params", "aq-mode=1:aq-strength=0.8:rc-lookahead=20:deblock=0,0:repeat-headers=1:aud=1:bframes=0:open-gop=0:8x8dct=0:cabac=0",
                "-tag:v", "avc1",
                "-c:a", "aac", "-b:a", str(audio_bps), "-ar", "48000", "-ac", "2",
                "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
                "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
                "-t", f"{duration:.3f}", str(out_path),
            ],
            stdin=subprocess.PIPE, stderr=err, bufsize=1024 * 1024,
        )
        frames = 0
        cached_group = -1
        add_plane = sub_plane = None
        try:
            for i, buf in _iter_frames(src, vf, w, h, max_seconds=max_seconds):
                group_idx = i // max(1, group)
                y = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w)
                if group_idx != cached_group:
                    add_plane, sub_plane = _delta_planes(y, bits, key, reveal_id, group_idx, amp)
                    cached_group = group_idx
                cv2.add(y, add_plane, dst=y)
                cv2.subtract(y, sub_plane, dst=y)
                try:
                    enc.stdin.write(buf)
                except BrokenPipeError:
                    break
                frames += 1
            try:
                enc.stdin.close()
            except BrokenPipeError:
                pass
            rc = enc.wait(timeout=max(90, int(duration * 8)))
        except subprocess.TimeoutExpired:
            enc.kill()
            enc.wait()
            raise RuntimeError("FFmpeg video encode timed out.")
        finally:
            if enc.poll() is None:
                enc.kill()
                enc.wait()
            err.seek(0)
            error_text = err.read().decode("utf-8", "replace")[-3000:]
            err.close()
        if rc != 0 or frames == 0 or not out_path.exists() or out_path.stat().st_size <= 0:
            raise RuntimeError("FFmpeg encode failed" + (f": {error_text}" if error_text else "."))
        return frames

    # The VBV-constrained first pass fits the budget in practice. The slower retry exists
    # only as a safety net for pathological input / muxer overshoot.
    attempts = [rate] + ([max(MIN_VIDEO_BPS, int(rate * 0.82))] if target else [])
    last_size = 0
    for attempt_no, attempt_rate in enumerate(attempts, 1):
        tmp_out = dst.with_name(f"{dst.stem}.encode{attempt_no}.tmp{dst.suffix}")
        try:
            encode_once(tmp_out, attempt_rate)
            last_size = tmp_out.stat().st_size
            if not target or last_size <= target:
                tmp_out.replace(dst)
                return {"width": w, "height": h, "fps": round(fps, 3), "bitrate": attempt_rate,
                        "attempts": attempt_no, "size": last_size}
        finally:
            tmp_out.unlink(missing_ok=True)

    raise RuntimeError(
        f"Video could not be encoded below the delivery limit ({last_size / 1048576:.1f} MiB generated, "
        f"target {target / 1048576:.1f} MiB)."
    )


def _decode_rgb_frame(buf: bytearray, w: int, h: int) -> np.ndarray:
    # yuv420p -> grayscale using Y plane; keeps extraction fast and stable.
    return np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w).copy()



def _orb(g: np.ndarray, nfeatures: int = 700):
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=8)
    return orb.detectAndCompute(g, None)


def _visual_signature(gray: np.ndarray, width: int = 64, height: int = 36) -> np.ndarray:
    """Compact normalized appearance fingerprint for fast temporal localization."""
    small = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    small = (small - float(small.mean())) / max(float(small.std()), 1.0)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = (grad - float(grad.mean())) / max(float(grad.std()), 1.0)
    sig = np.concatenate((small.ravel(), grad.ravel())).astype(np.float32)
    n = float(np.linalg.norm(sig))
    return sig / max(n, 1e-6)


def _coarse_frames(src: Path, max_height: int, sample_fps: float = TRACE_COARSE_FPS,
                   max_samples: int = TRACE_MAX_SAMPLES):
    """Sample the whole video at low resolution for fast temporal localization."""
    w, h, fps, vf = _probe_video(src, max_height)
    duration = _video_duration_seconds(src)
    rate = min(max(float(sample_fps), 0.5), 2.0)
    sf = min(rate, fps)
    if duration > 0:
        max_samples = min(max(int(max_samples), 24), max(24, int(np.ceil(duration * sf)) + 2))
    svf = f"fps={sf:.6f}," + _scale_filter(COARSE_H, "bilinear")
    scale_factor = min(1.0, COARSE_H / h)
    sw = max(64, round(w * scale_factor) // 2 * 2)
    sh = max(32, round(h * scale_factor) // 2 * 2)
    out = []
    for idx, buf in _iter_frames(src, svf, sw, sh, limit=max_samples):
        gray = np.frombuffer(buf, np.uint8, count=sw * sh).reshape(sh, sw).copy()
        out.append((idx, idx / sf, gray, _visual_signature(gray), None))
    return out, fps, vf, w, h, sf

def _match_desc(ld, od) -> int:
    if ld is None or od is None or len(ld) < 6 or len(od) < 6:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ld, od, k=2)
    good = [a for a, b in m if a.distance < .78 * b.distance]
    return len(good)


def _best_time_map(leak_samples, orig_samples) -> tuple[float, float, int]:
    """Estimate original_time ~= a*leak_time + b using cheap frame fingerprints."""
    if not leak_samples or not orig_samples:
        raise RuntimeError("Could not locate leaked video in original.")

    orig_times = np.asarray([x[1] for x in orig_samples], np.float32)
    orig_sig = np.stack([x[3] for x in orig_samples]).astype(np.float32)
    picks = np.linspace(0, len(leak_samples) - 1, min(7, len(leak_samples)), dtype=int)
    pairs = []
    for i in picks:
        lt = float(leak_samples[int(i)][1])
        ls = leak_samples[int(i)][3]
        scores = orig_sig @ ls
        k = min(6, len(scores))
        if k <= 0:
            continue
        idxs = np.argpartition(scores, -k)[-k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        for oi in idxs:
            pairs.append((float(scores[oi]), lt, float(orig_times[oi])))
    if not pairs:
        raise RuntimeError("Could not locate leaked video in original.")

    lt_arr = np.asarray([p[1] for p in pairs], np.float32)
    ot_arr = np.asarray([p[2] for p in pairs], np.float32)
    sc_arr = np.asarray([p[0] for p in pairs], np.float32)
    best = None
    for a in np.linspace(0.94, 1.06, 13):
        for _, lt, ot in sorted(pairs, reverse=True)[:24]:
            b = ot - a * lt
            residuals = np.abs(a * lt_arr + b - ot_arr)
            inliers = int(np.count_nonzero(residuals <= 1.0))
            score = float(np.sum(np.maximum(0.0, sc_arr - 0.50) *
                                 (residuals <= 1.5)))
            candidate = (inliers, score, float(a), float(b))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        score0, lt, ot = max(pairs, key=lambda x: x[0])
        return 1.0, ot - lt, 1
    return best[2], best[3], best[0]


def _decode_exact_pair(orig_gray: np.ndarray, leak_rgb: np.ndarray, orig_frame_idx: int,
                       key: bytes, reveal_id: str, max_height: int) -> Optional[tuple]:
    """Decode one frame pair cheaply first, then fall back to feature registration."""
    del max_height
    orig_y = orig_gray.astype(np.float32)
    leak_y = leak_rgb @ RGB2Y
    # Common case: same geometry after a Discord re-upload. Avoid SIFT entirely.
    if orig_y.shape == leak_y.shape:
        scores = _score(orig_y, leak_y, key, reveal_id, int(orig_frame_idx) // VIDEO_GROUP)
        uid, ok = decode_payload(scores)
        if ok:
            return int(uid), float(np.mean(np.abs(scores))), 1.0, int(orig_frame_idx), "direct"

    # Small resize-only changes are also common. Directly rescale before paying for SIFT.
    oh, ow = orig_y.shape
    lh, lw = leak_y.shape
    if lh >= 32 and lw >= 32 and abs((ow / max(oh, 1)) - (lw / max(lh, 1))) < 0.025:
        resized = cv2.resize(leak_y, (ow, oh), interpolation=cv2.INTER_CUBIC)
        scores = _score(orig_y, resized, key, reveal_id, int(orig_frame_idx) // VIDEO_GROUP)
        uid, ok = decode_payload(scores)
        if ok:
            return int(uid), float(np.mean(np.abs(scores))), 0.95, int(orig_frame_idx), "resize-direct"

    # Hard case: crop/rotation/unknown scale. Run the existing robust registration only now.
    orig_rgb = cv2.cvtColor(orig_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    regs = _registration_candidates(orig_rgb, leak_rgb)
    for H, reg, method in regs[:4]:
        uid, metric, _ = _decode_registered(
            orig_y, leak_y, key, reveal_id,
            int(orig_frame_idx) // VIDEO_GROUP, _geometry_variants(H)
        )
        if uid is not None:
            return int(uid), float(metric), float(reg), int(orig_frame_idx), method
    return None


def _exact_video_frame(orig: Path, leak_rgb: np.ndarray, approx_frame: int,
                       key: bytes, reveal_id: str, max_height: int,
                       max_radius_seconds: float = 1.25):
    w, h, fps, vf = _probe_video(orig, max_height)
    radius = max(6, int(round(fps * max_radius_seconds)))
    start = max(0, approx_frame - radius)
    leak_g, _ = _feature_image(leak_rgb, MATCH_H)
    lk, ld = _orb(leak_g, 1000)
    if ld is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    coarse = []
    # ORB is cheap enough to score every local frame; keep only a few for SIFT registration.
    for idx, buf in _iter_frames(orig, vf, w, h, skip=start, limit=2 * radius + 1):
        fr = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w).copy()
        g, _ = _feature_image(cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB), MATCH_H)
        kp, des = _orb(g, 900)
        if des is None:
            continue
        mm = matcher.knnMatch(ld, des, k=2)
        good = [a for a, b in mm if a.distance < .78 * b.distance]
        coarse.append((len(good), idx, fr))
    if not coarse:
        return None
    coarse.sort(reverse=True, key=lambda x: x[0])
    for _, idx, fr in coarse[:6]:
        leak_native = leak_rgb
        orig_rgb = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)
        regs = _registration_candidates(orig_rgb, leak_native)
        for H, reg, method in regs[:4]:
            uid, metric, _ = _decode_registered(fr.astype(np.float32), leak_native @ RGB2Y,
                                                key, reveal_id, idx // VIDEO_GROUP,
                                                _geometry_variants(H))
            if uid is not None:
                return uid, metric, reg, idx, method
    return None


def _video_frame_from_file(path: Path, max_height: int):
    w, h, fps, vf = _probe_video(path, max_height)
    for idx, buf in _iter_frames(path, vf, w, h, limit=1):
        return np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w).copy().astype(np.float32), fps, vf, w, h
    raise RuntimeError("Could not decode video.")


def extract_video_frame(orig: Path, leak: Path, key: bytes, reveal_id: str, *,
                        cell: int = CELL, max_height: int = 1080, start_frame: int = 0,
                        group: int = VIDEO_GROUP) -> dict:
    del cell, group
    with Image.open(leak) as im:
        leak_rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), np.uint8)
    w, h, fps, vf = _probe_video(orig, max_height)
    approx_candidates = [int(start_frame)] if start_frame else []
    if start_frame:
        approx = int(start_frame)
    else:
        # ORB coarse scan at 2 fps; unlike the old code, it preserves the leak's geometry.
        lg, _ = _feature_image(leak_rgb, COARSE_H)
        lkp, ld = _orb(lg, 900)
        if ld is None:
            raise RuntimeError("Could not analyze leaked frame.")
        sf = min(2.0, fps)
        svf = f"fps={sf:.6f}," + _scale_filter(COARSE_H, "bilinear")
        sw = max(64, round(w * min(1.0, COARSE_H / h)) // 2 * 2)
        sh = max(32, round(h * min(1.0, COARSE_H / h)) // 2 * 2)
        coarse = []
        for idx, buf in _iter_frames(orig, svf, sw, sh):
            fr = np.frombuffer(buf, np.uint8, count=sw * sh).reshape(sh, sw)
            _, des = _orb(fr, 900)
            if des is None:
                continue
            score = _match_desc(ld, des)
            if score >= 5:
                coarse.append((score, int(round(idx * fps / sf))))
        if not coarse:
            raise RuntimeError("Could not locate leaked frame in original video.")
        # A single coarse match can be a false positive in repetitive video. Try the
        # strongest distinct timestamps and let the watermark CRC be the final verifier.
        coarse.sort(reverse=True)
        approx_candidates = []
        for score, frame_idx in coarse[:16]:
            if all(abs(frame_idx - a) > max(6, int(fps * 0.35)) for a in approx_candidates):
                approx_candidates.append(frame_idx)
            if len(approx_candidates) >= 8:
                break
    if start_frame:
        approx_candidates = [int(start_frame)]

    hit = None
    for approx in approx_candidates:
        hit = _exact_video_frame(orig, leak_rgb, approx, key, reveal_id, max_height)
        if hit is not None:
            break
    if hit is None:
        raise RuntimeError("Could not verify watermark in the matched video frame.")
    uid, metric, reg, idx, method = hit
    return {"valid": True, "user_id": int(uid), "frame": int(idx), "frames_used": 1,
            "watermark_metric": round(float(metric), 4), "registration_score": round(float(reg), 3),
            "registration": method}


def _sample_fullres_indices(src: Path, max_height: int, indices: list[int], sample_fps: float = 2.0):
    """Read only selected frames at native watermark resolution in one FFmpeg pass."""
    if not indices:
        return {}, 0.0, 0, 0
    w, h, fps, vf = _probe_video(src, max_height)
    sf = min(max(sample_fps, 0.5), 4.0, fps)
    svf = f"fps={sf:.6f}," + _scale_filter(max_height, "lanczos")
    wanted = set(int(i) for i in indices)
    frames = {}
    max_idx = max(wanted)
    for idx, buf in _iter_frames(src, svf, w, h, limit=max_idx + 1):
        if idx in wanted:
            frames[idx] = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w).copy()
    return frames, sf, w, h


def _read_frame_at_time(src: Path, max_height: int, timestamp: float) -> Optional[tuple[np.ndarray, float, int, int]]:
    """Decode one frame with a fast input seek, keeping native watermark resolution."""
    w, h, fps, _ = _probe_video(src, max_height)
    ff = get_ffmpeg()
    ts = max(0.0, float(timestamp))
    # -ss before -i is intentionally used here: direct matching only needs a frame within
    # a few tens of milliseconds, and this avoids decoding the whole video repeatedly.
    proc = subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-ss", f"{ts:.4f}", "-i", str(src),
         "-frames:v", "1", "-vf", _scale_filter(max_height, "lanczos"),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=20,
    )
    if not proc.stdout:
        return None
    expected = w * h
    if len(proc.stdout) < expected:
        return None
    frame = np.frombuffer(proc.stdout, np.uint8, count=expected).reshape(h, w).copy()
    return frame, fps, w, h


def _fast_direct_video_extract(orig: Path, leak: Path, key: bytes, reveal_id: str,
                               max_height: int) -> Optional[dict]:
    """Fast path for ordinary Discord re-uploads with unchanged geometry/timing."""
    ow, oh, ofps, _ = _probe_video(orig, max_height)
    lw, lh, lfps, _ = _probe_video(leak, max_height)
    if (ow, oh) != (lw, lh) or abs(ofps - lfps) > 0.75:
        return None

    od = _video_duration_seconds(orig)
    ld = _video_duration_seconds(leak)
    if od <= 0 or ld <= 0:
        return None
    common = min(od, ld)
    if common < 0.5:
        return None

    # Sample several independent times. The ±2-frame window handles the small timestamp
    # offsets introduced by Discord/container muxing without SIFT/ORB registration.
    hits = []
    for frac in (0.10, 0.33, 0.58, 0.82, 0.95):
        t = common * frac
        leak_hit = _read_frame_at_time(leak, max_height, t)
        if leak_hit is None:
            continue
        leak_frame, leak_fps, _, _ = leak_hit
        leak_thumb = cv2.resize(
            leak_frame, (max(64, ow // 12), max(36, oh // 12)), interpolation=cv2.INTER_AREA
        )
        best_decode = None
        for offset in (0.0, -1.0, 1.0, -2.0, 2.0):
            source_t = max(0.0, min(od - 1.0 / ofps, t + offset / ofps))
            orig_hit = _read_frame_at_time(orig, max_height, source_t)
            if orig_hit is None:
                continue
            orig_frame, orig_fps, _, _ = orig_hit
            orig_thumb = cv2.resize(
                orig_frame, (leak_thumb.shape[1], leak_thumb.shape[0]), interpolation=cv2.INTER_AREA
            )
            mae = float(np.mean(np.abs(orig_thumb.astype(np.int16) - leak_thumb.astype(np.int16))))
            # Try the best content matches first, but allow every nearby candidate to satisfy
            # the CRC-protected watermark in case the motion match is ambiguous.
            source_idx = int(round(source_t * orig_fps))
            ranked = (mae, source_idx, orig_frame)
            if best_decode is None or ranked[0] < best_decode[0]:
                best_decode = ranked
            for _, candidate_idx, candidate_frame in sorted(
                [best_decode, ranked] if best_decode is not None else [ranked], key=lambda x: x[0]
            )[:2]:
                group_idx = candidate_idx // VIDEO_GROUP
                scores = _score(candidate_frame.astype(np.float32), leak_frame.astype(np.float32), key, reveal_id, group_idx)
                uid, ok = decode_payload(scores)
                if ok:
                    metric = float(np.mean(np.abs(scores)))
                    hits.append((int(uid), metric, candidate_idx))
                    break
            if hits and hits[-1][2] == source_idx:
                break

    if not hits:
        return None
    counts = {}
    for uid, *_ in hits:
        counts[uid] = counts.get(uid, 0) + 1
    uid = max(counts, key=counts.get)
    same = [h for h in hits if h[0] == uid]
    if len(same) < 2 and len(hits) >= 3:
        return None
    return {
        "valid": True,
        "user_id": uid,
        "start_frame": int(same[0][2]),
        "frames_used": len(same),
        "anchors": len(hits),
        "watermark_metric": round(float(np.mean([h[1] for h in same])), 4),
        "registration": "direct-time-aligned",
    }



def _read_frame_near_index(src: Path, max_height: int, frame_idx: int) -> Optional[tuple[np.ndarray, float, int, int]]:
    w, h, fps, vf = _probe_video(src, max_height)
    target = max(0, int(frame_idx))
    for idx, buf in _iter_frames(src, vf, w, h, skip=target, limit=1):
        fr = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w).copy()
        return fr, fps, w, h
    return None


def extract_video(orig: Path, leak: Path, key: bytes, reveal_id: str, *,
                  cell: int = CELL, max_height: int = 1080, start_frame: int = 0,
                  group: int = VIDEO_GROUP) -> dict:
    del cell, group

    # Explicit original-frame hint. Decode the leaked frame near its equivalent
    # timestamp and test a tiny time neighborhood rather than scanning the timeline.
    if int(start_frame) > 0:
        _, _, ofps, _ = _probe_video(orig, max_height)
        source_t = max(0.0, int(start_frame) / max(ofps, 1e-6))
        for dt in (0.0, -0.20, 0.20, -0.40, 0.40):
            leak_hit = _read_frame_at_time(leak, max_height, max(0.0, source_t + dt))
            if leak_hit is None:
                continue
            leak_frame, _, _, _ = leak_hit
            orig_hit = _read_frame_at_time(orig, max_height, source_t + dt)
            if orig_hit is None:
                continue
            orig_frame, exact_fps, _, _ = orig_hit
            hit = _decode_exact_pair(
                orig_frame, cv2.cvtColor(leak_frame, cv2.COLOR_GRAY2RGB),
                int(round((source_t + dt) * exact_fps)), key, reveal_id, max_height
            )
            if hit is not None:
                uid, metric, reg, frame_idx, method = hit
                return {"valid": True, "user_id": int(uid), "start_frame": int(frame_idx),
                        "frames_used": 1, "anchors": 1,
                        "watermark_metric": round(float(metric), 4),
                        "registration_score": round(float(reg), 3), "registration": method}

    # Fast path for an ordinary re-upload with unchanged timing/geometry.
    fast_hit = _fast_direct_video_extract(orig, leak, key, reveal_id, max_height)
    if fast_hit is not None:
        return fast_hit

    # Localize a trimmed/retimed excerpt using a few ultra-cheap signatures over the
    # entire supported duration, then decode only the short-listed original frames.
    leak_samples, leak_fps, _, _, _, _ = _coarse_frames(
        leak, max_height, sample_fps=TRACE_COARSE_FPS, max_samples=TRACE_MAX_SAMPLES
    )
    orig_samples, orig_fps, _, _, _, _ = _coarse_frames(
        orig, max_height, sample_fps=TRACE_COARSE_FPS, max_samples=TRACE_MAX_SAMPLES
    )
    if not leak_samples or not orig_samples:
        raise RuntimeError("Could not sample video frames.")

    a, b, anchors = _best_time_map(leak_samples, orig_samples)
    picks = np.linspace(0, len(leak_samples) - 1, min(8, len(leak_samples)), dtype=int)
    hits = []
    for p in picks:
        ls = leak_samples[int(p)]
        predicted_time = max(0.0, a * ls[1] + b)
        # Signature matching is approximate. Test five nearby timestamps and stop at
        # the first CRC-verified watermark. This avoids the old full local frame scan.
        for dt in (0.0, -0.25, 0.25, -0.50, 0.50):
            t_orig = max(0.0, predicted_time + dt)
            t_leak = max(0.0, float(ls[1]))
            orig_hit = _read_frame_at_time(orig, max_height, t_orig)
            leak_hit = _read_frame_at_time(leak, max_height, t_leak)
            if orig_hit is None or leak_hit is None:
                continue
            orig_frame, exact_fps, _, _ = orig_hit
            leak_frame, _, _, _ = leak_hit
            idx = int(round(t_orig * exact_fps))
            hit = _decode_exact_pair(
                orig_frame, cv2.cvtColor(leak_frame, cv2.COLOR_GRAY2RGB), idx,
                key, reveal_id, max_height
            )
            if hit is not None:
                hits.append(hit)
                break

    if not hits:
        raise RuntimeError("No valid watermark found in the leaked video.")

    counts = {}
    for uid, *_ in hits:
        counts[int(uid)] = counts.get(int(uid), 0) + 1
    uid = max(counts, key=counts.get)
    same = [h for h in hits if int(h[0]) == int(uid)]
    if len(hits) >= 3 and len(same) < 2:
        raise RuntimeError("Watermark evidence was inconsistent across the video.")
    return {"valid": True, "user_id": uid, "start_frame": int(same[0][3]),
            "frames_used": len(same), "anchors": int(anchors),
            "watermark_metric": round(float(np.mean([x[1] for x in same])), 4),
            "registration": same[0][4]}

def extract(orig: Path, leak: Path, kind: str, key: bytes, reveal_id: str, *,
            image_cell: int = CELL, video_cell: int = CELL, max_height: int = 1080,
            start_frame: int = 0) -> dict:
    if kind == "image":
        return extract_image(orig, leak, key, reveal_id, image_cell)
    if kind == "video_frame":
        return extract_video_frame(orig, leak, key, reveal_id, cell=video_cell,
                                   max_height=max_height, start_frame=start_frame)
    return extract_video(orig, leak, key, reveal_id, cell=video_cell,
                         max_height=max_height, start_frame=start_frame)
