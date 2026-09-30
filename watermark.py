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
VIDEO_GROUP = 4
MATCH_H = 360
COARSE_H = 240
VIDEO_MAX_FPS = 30.0

_x, _y = np.mgrid[:CELL, :CELL].astype(np.float32)
_CARRIER = (
    np.cos(np.pi * (_y + .5) / CELL) * np.cos(np.pi * 2 * (_x + .5) / CELL)
    + .8 * np.cos(np.pi * 2 * (_y + .5) / CELL) * np.cos(np.pi * (_x + .5) / CELL)
).astype(np.float32)
_CARRIER /= np.sqrt(np.mean(_CARRIER * _CARRIER)) + 1e-8
RGB2Y = np.array([0.299, 0.587, 0.114], np.float32)


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
    value = 0
    for score in scores:
        value = (value << 1) | int(score > 0)
    uid = value >> 32
    return uid, zlib.crc32(uid.to_bytes(8, "big")) == (value & 0xFFFFFFFF)


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

    flat = vals[coverage >= 0.25]
    if flat.size > 32:
        med = np.median(flat)
        mad = np.median(np.abs(flat - med)) + 1e-6
        vals = np.clip(vals, med - 4.0 * mad, med + 4.0 * mad)

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
    okp, od = _sift(og)
    lkp, ld = _sift(lg)
    if od is None or ld is None or len(okp) < 10 or len(lkp) < 10:
        return []

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ld, od, k=2)
    good = [a for a, b in knn if a.distance < 0.78 * b.distance]
    if len(good) < 8:
        return []

    src = np.float32([lkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([okp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    out: list[tuple[np.ndarray, float, str]] = []

    for threshold in (2.5, 4.0, 6.0, 9.0):
        Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold, maxIters=4000, confidence=0.995)
        if Hs is not None:
            inliers = int(mask.sum()) if mask is not None else 0
            if inliers >= 7:
                So = np.diag([1 / os, 1 / os, 1.0])
                Sl = np.diag([ls, ls, 1.0])
                H = So @ Hs @ Sl
                out.append((H, inliers / max(len(good), 1), f"sift-r{threshold:g}"))

    # Affine fallback is often more stable than an overfit homography on video frames.
    A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                          maxIters=4000, confidence=0.995)
    if A is not None and mask is not None and int(mask.sum()) >= 7:
        Ah = np.vstack([A, [0, 0, 1]]).astype(np.float64)
        So = np.diag([1 / os, 1 / os, 1.0])
        Sl = np.diag([ls, ls, 1.0])
        H = So @ Ah @ Sl
        out.append((H, int(mask.sum()) / max(len(good), 1), "affine"))

    # Deduplicate near-identical transforms.
    unique: list[tuple[np.ndarray, float, str]] = []
    for item in sorted(out, key=lambda x: x[1], reverse=True):
        H = item[0]
        if any(np.max(np.abs(H - u[0])) < 0.02 * max(1.0, np.max(np.abs(H))) for u in unique):
            continue
        unique.append(item)
        if len(unique) >= 6:
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
    """Small sub-pixel/registration perturbations for screenshots and social encoders."""
    out = [H.copy()]
    for tx in (-2.0, -1.0, -0.35, 0.35, 1.0, 2.0):
        for ty in (-2.0, -1.0, 0.0, 1.0, 2.0):
            M = H.copy()
            M[0, 2] += tx
            M[1, 2] += ty
            out.append(M)
    for sx, sy in ((0.995, 1), (1.005, 1), (1, 0.995), (1, 1.005),
                   (0.99, 0.99), (1.01, 1.01)):
        M = H.copy()
        M[0, :3] *= sx
        M[1, :3] *= sy
        out.append(M)
    return out


def _decode_registered(oy: np.ndarray, leak_y: np.ndarray, key: bytes, reveal_id: str,
                       group: int, Hs: list[np.ndarray]) -> tuple[Optional[int], float, np.ndarray]:
    best_metric = -1e9
    best_scores = np.zeros(N_BITS, np.float64)
    for H in Hs:
        warped, mask = _warp(leak_y, H, oy.shape)
        scores = _score(oy, warped, key, reveal_id, group, mask)
        uid, ok = decode_payload(scores)
        metric = float(np.mean(np.abs(scores)))
        if ok:
            return uid, metric, scores
        if metric > best_metric:
            best_metric = metric
            best_scores = scores
    return None, best_metric, best_scores


def embed_image(src: Path, dst: Path, user_id: int, key: bytes, reveal_id: str,
                amp: float = AMP, cell: int = CELL) -> None:
    del cell  # fixed robust layout; compatibility with older bot configuration
    rgb = np.asarray(_load_rgb(src), np.float32)
    y = rgb @ RGB2Y
    delta = _delta(y, encode_payload(user_id), key, reveal_id, 0, amp)
    out = Image.fromarray(np.clip(rgb + delta[..., None], 0, 255).astype(np.uint8), "RGB")
    suffix = dst.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        out.save(dst, "JPEG", quality=95, subsampling=0)
    elif suffix == ".webp":
        out.save(dst, "WEBP", lossless=True)
    else:
        out.save(dst, "PNG", compress_level=4)


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


def _probe_video(src: Path, max_height: int):
    """Probe and normalize video settings used by both embed and extraction."""
    ff = get_ffmpeg()
    info = subprocess.run(
        [ff, "-hide_banner", "-i", str(src)],
        capture_output=True, text=True, timeout=30,
    ).stderr
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
    vf = rf"fps={fps:.6f},scale=-2:trunc(min(ih\,{max_height})/2)*2:flags=lanczos"
    probe = subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", vf,
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=45,
    )
    if not probe.stdout:
        details = probe.stderr.decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"FFmpeg could not decode the video. {details}".strip())
    with Image.open(io.BytesIO(probe.stdout)) as im:
        w, h = im.size
    return w, h, fps, vf

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


def embed_video(src: Path, dst: Path, user_id: int, key: bytes, reveal_id: str, *,
                amp: float = AMP, cell: int = CELL, crf: int = 24, preset: str = "veryfast",
                max_seconds: int = 300, max_height: int = 720, group: int = VIDEO_GROUP) -> None:
    """Embed watermark and output a mobile-compatible, fast-start MP4."""
    del cell
    w, h, fps, vf = _probe_video(src, max_height)
    bits = encode_payload(user_id)
    ff = get_ffmpeg()
    gop = max(15, int(round(fps * 2.0)))
    maxrate = 2_500_000 if h <= 720 else 5_000_000
    bufsize = maxrate * 2
    duration = max(1, int(max_seconds))
    dst.parent.mkdir(parents=True, exist_ok=True)

    err = tempfile.TemporaryFile()
    enc = subprocess.Popen(
        [
            ff, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
            "-framerate", str(fps), "-i", "-",
            "-i", str(src),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-profile:v", "baseline", "-level", "3.1" if h <= 720 else "4.0",
            "-pix_fmt", "yuv420p", "-bf", "0",
            "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "40",
            "-maxrate", str(maxrate), "-bufsize", str(bufsize),
            "-tag:v", "avc1",
            "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "2",
            "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
            "-movflags", "+faststart", "-t", str(duration), "-shortest", str(dst),
        ],
        stdin=subprocess.PIPE, stderr=err, bufsize=1024 * 1024,
    )

    frames = 0
    rc = 1
    try:
        for i, buf in _iter_frames(src, vf, w, h, max_seconds=duration):
            y = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w)
            delta = np.rint(_delta(y.astype(np.float32), bits, key, reveal_id, i // group, amp))
            y[:] = np.clip(y.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            try:
                enc.stdin.write(buf)
            except BrokenPipeError:
                break
            frames += 1
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
        rc = enc.wait(timeout=max(60, duration * 10))
    except subprocess.TimeoutExpired:
        enc.kill()
        enc.wait()
        rc = 1
        raise RuntimeError("FFmpeg video encode timed out.")
    finally:
        if enc.poll() is None:
            enc.kill()
            enc.wait()
        err.seek(0)
        error_text = err.read().decode("utf-8", "replace")[-3000:]
        err.close()

    if rc != 0 or frames == 0 or not dst.exists() or dst.stat().st_size <= 0:
        raise RuntimeError("FFmpeg encode failed" + (f": {error_text}" if error_text else "."))

def _decode_rgb_frame(buf: bytearray, w: int, h: int) -> np.ndarray:
    # yuv420p -> grayscale using Y plane; keeps extraction fast and stable.
    return np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w).copy()



def _orb(g: np.ndarray, nfeatures: int = 700):
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=8)
    return orb.detectAndCompute(g, None)


def _coarse_frames(src: Path, max_height: int, sample_fps: float = 2.0,
                   max_samples: int = 360):
    w, h, fps, vf = _probe_video(src, max_height)
    rate = min(max(sample_fps, .5), 4.0)
    sf = min(rate, fps)
    svf = rf"fps={sf:.6f},scale=-2:trunc(min(ih\,{COARSE_H})/2)*2:flags=bilinear"
    scale_factor = min(1.0, COARSE_H / h)
    sw = max(64, round(w * scale_factor) // 2 * 2)
    sh = max(32, round(h * scale_factor) // 2 * 2)
    out = []
    for idx, buf in _iter_frames(src, svf, sw, sh, limit=max_samples):
        gray = np.frombuffer(buf, np.uint8, count=sw * sh).reshape(sh, sw).copy()
        kp, des = _orb(gray, 700)
        out.append((idx, idx / sf, gray, kp, des))
    return out, fps, vf, w, h, sf

def _match_desc(ld, od) -> int:
    if ld is None or od is None or len(ld) < 6 or len(od) < 6:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ld, od, k=2)
    good = [a for a, b in m if a.distance < .78 * b.distance]
    return len(good)


def _coarse_candidates(leak_sample, orig_samples, n=10):
    _, ltime, _, _, ld = leak_sample
    scored = []
    for oi, otime, _, _, od in orig_samples:
        score = _match_desc(ld, od)
        if score:
            scored.append((score, oi, otime, ltime))
    scored.sort(reverse=True)
    return scored[:n]


def _best_time_map(leak_samples, orig_samples) -> tuple[float, float, int]:
    """Fit original_time ~= a*leak_time + b using several cheap ORB anchors."""
    picks = np.linspace(0, len(leak_samples) - 1, min(5, len(leak_samples)), dtype=int)
    pairs = []
    for i in picks:
        cand = _coarse_candidates(leak_samples[int(i)], orig_samples, 8)
        if not cand:
            continue
        # Keep several possibilities; later RANSAC-style scoring selects the consistent map.
        pairs.extend((c[0], c[3], c[2]) for c in cand)
    if not pairs:
        raise RuntimeError("Could not locate leaked video in original.")

    best = None
    # Most social-media processing preserves timing, so keep slope around 1 but permit modest retiming.
    slopes = np.linspace(.85, 1.18, 18)
    for a in slopes:
        for _, lt, ot in pairs:
            b = ot - a * lt
            residuals = sorted(abs((a * p[1] + b) - p[2]) for p in pairs)
            if not residuals:
                continue
            inliers = sum(r <= .65 for r in residuals)
            score = inliers * 10 - np.median(residuals)
            candidate = (score, a, b, inliers)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None or best[3] < 2:
        # Single-anchor fallback.
        _, lt, ot = max(pairs, key=lambda x: x[0])
        return 1.0, ot - lt, 1
    return best[1], best[2], best[3]


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
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        sf = min(2.0, fps)
        svf = f"fps={sf:.6f},scale=-2:trunc(min(ih\\,{COARSE_H})/2)*2:flags=bilinear"
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


def extract_video(orig: Path, leak: Path, key: bytes, reveal_id: str, *,
                  cell: int = CELL, max_height: int = 1080, start_frame: int = 0,
                  group: int = VIDEO_GROUP) -> dict:
    del cell, start_frame, group
    leak_samples, leak_fps, _, _, _, _ = _coarse_frames(leak, max_height, sample_fps=2.0, max_samples=80)
    orig_samples, orig_fps, _, _, _, _ = _coarse_frames(orig, max_height, sample_fps=2.0, max_samples=360)
    if not leak_samples or not orig_samples:
        raise RuntimeError("Could not sample video frames.")

    a, b, anchors = _best_time_map(leak_samples, orig_samples)
    picks = np.linspace(0, len(leak_samples) - 1, min(6, len(leak_samples)), dtype=int)
    hits = []
    for p in picks:
        ls = leak_samples[int(p)]
        predicted_time = a * ls[1] + b
        approx = max(0, int(round(predicted_time * orig_fps)))
        # Reconstruct RGB only for SIFT registration; grayscale content is sufficient for ORB,
        # but the leak's actual geometry must be preserved.
        # The coarse frame is grayscale; convert to RGB for the registration stage.
        leak_rgb = cv2.cvtColor(ls[2], cv2.COLOR_GRAY2RGB)
        hit = _exact_video_frame(orig, leak_rgb, approx, key, reveal_id, max_height, max_radius_seconds=.9)
        if hit is not None:
            uid, metric, reg, frame_idx, method = hit
            hits.append((uid, metric, reg, frame_idx, method))

    if not hits:
        raise RuntimeError("No valid watermark found in the leaked video.")

    counts = {}
    for uid, *_ in hits:
        counts[int(uid)] = counts.get(int(uid), 0) + 1
    uid = max(counts, key=counts.get)
    same = [h for h in hits if int(h[0]) == int(uid)]
    return {"valid": True, "user_id": int(uid), "start_frame": int(same[0][3]),
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
