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

MAX_BITS_PER_PIXEL = 0.11
MIN_VIDEO_BPS = 128_000
VBV_SECONDS = 1.5
CONTAINER_MARGIN = 0.975

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


def _delta(
    y: np.ndarray,
    bits: np.ndarray,
    key: bytes,
    reveal_id: str,
    group: int,
    amp: float,
) -> np.ndarray:
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
    m = valid > 0.35
    if int(m.sum()) < 500:
        return leak

    x = orig[m].astype(np.float64)
    y = leak[m].astype(np.float64)

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


def _score(
    orig: np.ndarray,
    leak: np.ndarray,
    key: bytes,
    reveal_id: str,
    group: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
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

    wsum = vb.sum((2, 3), keepdims=True)
    mean = (b * vb).sum((2, 3), keepdims=True) / np.maximum(wsum, 1e-5)
    b = b - mean

    corr = (b * _CARRIER[None, None] * vb).sum((2, 3))
    den = np.sqrt(
        (vb * (_CARRIER[None, None] ** 2)).sum((2, 3))
        * (vb * (b ** 2)).sum((2, 3))
        + 1e-5
    )
    corr = corr / np.maximum(den, 1e-4)

    coverage = np.clip(vb.mean((2, 3)), 0, 1)
    vals = corr * _chips(key, reveal_id, group, ch, cw) * coverage
    vals[coverage < 0.25] = 0.0

    flat = np.abs(vals[coverage >= 0.25])
    if flat.size > 32:
        med = np.median(flat)
        mad = np.median(np.abs(flat - med)) + 1e-6
        limit = max(1.0, med + 8.0 * mad)
        vals = np.clip(vals, -limit, limit)

    return np.bincount(
        _layout(key, reveal_id, h, w).ravel(),
        vals.ravel(),
        minlength=N_BITS,
    )


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _feature_image(
    rgb: np.ndarray,
    height: int = MATCH_H,
) -> tuple[np.ndarray, float]:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb.astype(np.uint8)
    scale = min(1.0, height / max(g.shape))
    if scale < 1.0:
        g = cv2.resize(
            g,
            (
                max(32, round(g.shape[1] * scale)),
                max(32, round(g.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return g, scale


def _registration_candidates(
    orig: np.ndarray,
    leak: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    og, os = _feature_image(orig)
    lg, ls = _feature_image(leak)

    candidates: list[tuple[np.ndarray, np.ndarray]] = [(leak, np.ones(leak.shape, np.float32))]

    try:
        sift = cv2.SIFT_create(nfeatures=900)
        ok, od = sift.detectAndCompute(og, None)
        lk, ld = sift.detectAndCompute(lg, None)

        if od is not None and ld is not None and len(ok) >= 8 and len(lk) >= 8:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            matches = matcher.knnMatch(ld, od, k=2)
            good = [m for m, n in matches if m.distance < 0.72 * n.distance]

            if len(good) >= 6:
                src = np.float32([lk[m.queryIdx].pt for m in good])
                dst = np.float32([ok[m.trainIdx].pt for m in good])

                hmat, mask = cv2.findHomography(
                    src,
                    dst,
                    cv2.RANSAC,
                    5.0,
                )

                if hmat is not None and mask is not None and int(mask.sum()) >= 5:
                    inv = np.linalg.inv(hmat)
                    warped = cv2.warpPerspective(
                        leak,
                        inv,
                        (orig.shape[1], orig.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    valid = cv2.warpPerspective(
                        np.ones(leak.shape, np.float32),
                        inv,
                        (orig.shape[1], orig.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    candidates.append((warped, valid))
    except cv2.error:
        pass

    if orig.shape != leak.shape:
        resized = cv2.resize(
            leak,
            (orig.shape[1], orig.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        candidates.append((resized, np.ones(orig.shape, np.float32)))

    return candidates


def _geometry_variants(h: int) -> list[tuple[float, float]]:
    jitter = min(0.015, 4.0 / max(h, 1))
    return [
        (0.0, 0.0),
        (-jitter, 0.0),
        (jitter, 0.0),
        (0.0, -jitter),
        (0.0, jitter),
        (-jitter, -jitter),
        (jitter, jitter),
    ]


def _warp_y(
    y: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    h, w = y.shape
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, 0.0, 1.0)
    matrix[0, 0] *= 1.0 + scale_x
    matrix[1, 1] *= 1.0 + scale_y
    return cv2.warpAffine(
        y,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _decode_registered(
    oy: np.ndarray,
    leak_y: np.ndarray,
    key: bytes,
    reveal_id: str,
    group: int,
    variants: list[tuple[float, float]],
) -> tuple[int, float, bool]:
    best_uid = 0
    best_metric = float("-inf")
    best_ok = False

    for sx, sy in variants:
        warped = _warp_y(leak_y, sx, sy)

        for candidate in (warped, cv2.GaussianBlur(warped, (3, 3), 0)):
            scores = _score(
                oy,
                candidate,
                key,
                reveal_id,
                group,
            )
            uid, ok = decode_payload(scores)

            metric = float(np.mean(np.abs(scores)))
            if ok and metric > best_metric:
                best_uid = uid
                best_metric = metric
                best_ok = True

    return best_uid, best_metric, best_ok


def embed_image_array(
    rgb: np.ndarray,
    user_id: int,
    key: bytes,
    reveal_id: str,
    amp: float = AMP,
) -> np.ndarray:
    h, w, _ = rgb.shape
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    d = _delta(y, encode_payload(user_id), key, reveal_id, 0, amp)

    y2 = np.clip(y + d, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(
        cv2.merge((y2, cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 1],
                   cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 2])),
        cv2.COLOR_YCrCb2RGB,
    )
    return out


def _save_image(img: Image.Image, dst: Path) -> None:
    suffix = dst.suffix.lower()

    if suffix in (".jpg", ".jpeg"):
        img.save(dst, "JPEG", quality=95, subsampling=0)
    elif suffix == ".webp":
        img.save(dst, "WEBP", lossless=True)
    else:
        img.save(dst, "PNG", compress_level=4)


def embed_image(
    src: Path,
    dst: Path,
    user_id: int,
    key: bytes,
    reveal_id: str,
    amp: float = AMP,
    cell: int = CELL,
) -> None:
    del cell
    rgb = np.asarray(_load_rgb(src), np.uint8)
    _save_image(
        Image.fromarray(
            embed_image_array(rgb, user_id, key, reveal_id, amp)
        ),
        dst,
    )


def render_image_preview(
    src: Path,
    preview: Path,
    user_id: int,
    key: bytes,
    reveal_id: str,
    *,
    amp: float = AMP,
    max_dim: int = 2048,
    quality: int = 88,
) -> None:
    rgb = np.asarray(_load_rgb(src), np.uint8)
    img = Image.fromarray(
        embed_image_array(rgb, user_id, key, reveal_id, amp)
    )
    del rgb

    img.thumbnail(
        (max_dim, max_dim),
        Image.Resampling.LANCZOS,
    )

    preview.parent.mkdir(parents=True, exist_ok=True)

    img.save(
        preview,
        "JPEG",
        quality=max(70, min(95, int(quality))),
        optimize=True,
        progressive=False,
        subsampling=2,
    )


def extract_image(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    cell: int = CELL,
) -> dict:
    del cell

    o = np.asarray(_load_rgb(orig), np.uint8)
    l = np.asarray(_load_rgb(leak), np.uint8)

    oy = o @ RGB2Y
    ly = l @ RGB2Y

    candidates = _registration_candidates(o, l)

    best = None

    for candidate, valid in candidates:
        uid, metric, ok = _decode_registered(
            oy,
            ly if candidate is leak else candidate @ RGB2Y,
            key,
            reveal_id,
            0,
            _geometry_variants(oy.shape[0]),
        )

        if ok and (best is None or metric > best["metric"]):
            best = {
                "user_id": int(uid),
                "metric": metric,
                "ok": True,
            }

    if best:
        return best

    return {
        "user_id": 0,
        "metric": 0.0,
        "ok": False,
    }


def _scale_filter(height: int, flags: str) -> str:
    return rf"scale=-2:trunc(min(ih\,{int(height)})/2)*2:flags={flags},setsar=1"


def _file_sig(path: Path) -> tuple[str, int, int]:
    p = Path(path)
    st = p.stat()
    return str(p), st.st_mtime_ns, st.st_size


@lru_cache(maxsize=64)
def _ffmpeg_info_cached(
    path: str,
    mtime_ns: int,
    size: int,
) -> str:
    del mtime_ns, size

    return subprocess.run(
        [get_ffmpeg(), "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    ).stderr


def _ffmpeg_info(src: Path) -> str:
    return _ffmpeg_info_cached(*_file_sig(src))


@lru_cache(maxsize=64)
def _probe_video_cached(
    path: str,
    mtime_ns: int,
    size: int,
    max_height: int,
):
    info = _ffmpeg_info_cached(path, mtime_ns, size)

    fps = None

    for pattern in (
        r"(\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+))?)?\s*fps",
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
    fps = min(fps, VIDEO_MAX_FPS)

    max_height = max(144, int(max_height))
    vf = f"fps={fps:.6f}," + _scale_filter(max_height, "lanczos")

    probe = subprocess.run(
        [
            get_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        capture_output=True,
        timeout=45,
    )

    if not probe.stdout:
        details = probe.stderr.decode("utf-8", "replace")[-1500:]
        raise RuntimeError(
            f"FFmpeg could not decode the video. {details}".strip()
        )

    with Image.open(io.BytesIO(probe.stdout)) as im:
        w, h = im.size

    return w, h, fps, vf


def _probe_video(src: Path, max_height: int):
    return _probe_video_cached(
        *_file_sig(src),
        int(max_height),
    )


def video_info(
    src: Path,
    max_height: int = 1080,
) -> dict:
    w, h, fps, _ = _probe_video(src, max_height)

    return {
        "width": w,
        "height": h,
        "fps": round(fps, 3),
        "duration": round(_video_duration_seconds(src), 2),
    }


def _read_full(stream, buf) -> bool:
    view = memoryview(buf)
    got = 0

    while got < len(buf):
        n = stream.readinto(view[got:])

        if not n:
            return False

        got += n

    return True


def _iter_frames(
    src: Path,
    vf: str,
    w: int,
    h: int,
    *,
    skip: int = 0,
    limit: Optional[int] = None,
    max_seconds: Optional[int] = None,
) -> Iterator[tuple[int, bytearray]]:
    size = w * h * 3 // 2

    cmd = [
        get_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if max_seconds:
        cmd += ["-t", str(max_seconds)]

    cmd += [
        "-i",
        str(src),
        "-an",
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=size * 2,
    )

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
        proc.kill()
        proc.wait()
        proc.stdout.close()


def _video_duration_seconds(src: Path) -> float:
    m = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        _ffmpeg_info(src),
    )

    if not m:
        return 0.0

    return (
        float(m.group(1)) * 3600.0
        + float(m.group(2)) * 60.0
        + float(m.group(3))
    )


def _adaptive_video_height(
    source_width: int,
    source_height: int,
    duration: float,
    requested_max: int,
    target_bytes: Optional[int],
    audio_bps: int,
    fps: float,
) -> int:
    ceiling = min(int(source_height), int(requested_max))
    source_width = max(2, int(source_width))

    if ceiling <= 360 or not target_bytes or duration <= 0:
        return max(144, ceiling)

    usable_bits = max(
        1,
        int(target_bytes * 8 * 0.965)
        - int(audio_bps * duration),
    )

    available_bps = usable_bits / max(duration, 1.0)
    test_fps = max(24.0, min(float(fps), 30.0))
    min_bpp = 0.062

    for candidate in (
        1080,
        900,
        720,
        648,
        576,
        540,
        480,
        360,
    ):
        if candidate > ceiling:
            continue

        candidate_width = max(
            2,
            round(
                source_width
                * candidate
                / max(source_height, 1)
            ),
        )

        required = (
            candidate_width
            * candidate
            * test_fps
            * min_bpp
        )

        if available_bps >= required:
            return max(144, int(candidate))

    return min(360, ceiling)


def _plan_video_rate(
    *,
    width: int,
    height: int,
    fps: float,
    duration: float,
    target: Optional[int],
    audio_bps: int,
    max_bpp: float,
) -> int:
    if target:
        budget_bits = (
            target * 8 * CONTAINER_MARGIN
            - audio_bps * duration
        )

        rate = budget_bits / (
            max(duration, 1.0) + VBV_SECONDS
        )
    elif height >= 900:
        rate = 9_000_000
    elif height >= 700:
        rate = 6_500_000
    else:
        rate = 4_000_000

    if max_bpp > 0:
        rate = min(
            rate,
            width
            * height
            * max(fps, 1.0)
            * max_bpp,
        )

    return max(MIN_VIDEO_BPS, int(rate))


def _delta_planes(
    y: np.ndarray,
    bits: np.ndarray,
    key: bytes,
    reveal_id: str,
    group: int,
    amp: float,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.rint(
        _delta(
            y.astype(np.float32),
            bits,
            key,
            reveal_id,
            group,
            amp,
        )
    ).astype(np.int16)

    return (
        np.clip(d, 0, 255).astype(np.uint8),
        np.clip(-d, 0, 255).astype(np.uint8),
    )


def embed_video(
    src: Path,
    dst: Path,
    user_id: int,
    key: bytes,
    reveal_id: str,
    *,
    amp: float = AMP,
    cell: int = CELL,
    preset: str = "fast",
    max_seconds: int = 300,
    max_height: int = 1080,
    group: int = VIDEO_GROUP,
    target_bytes: Optional[int] = None,
    audio_kbps: int = 96,
    max_bpp: float = MAX_BITS_PER_PIXEL,
) -> dict:
    del cell

    duration = min(
        float(max_seconds),
        max(0.0, _video_duration_seconds(src)),
    )

    if duration <= 0:
        duration = float(max_seconds)

    target = max(1, int(target_bytes)) if target_bytes else None
    bits = encode_payload(user_id)

    audio_bps = max(
        64000,
        min(128000, int(audio_kbps) * 1000),
    )

    source_w, source_h, source_fps, _ = _probe_video(
        src,
        max_height,
    )

    delivery_height = _adaptive_video_height(
        source_w,
        source_h,
        duration,
        max_height,
        target,
        audio_bps,
        source_fps,
    )

    w, h, fps, vf = _probe_video(
        src,
        delivery_height,
    )

    fps = min(float(fps), 30.0)

    gop = max(
        1,
        min(120, int(round(fps * 2.0))),
    )

    rate = _plan_video_rate(
        width=w,
        height=h,
        fps=fps,
        duration=duration,
        target=target,
        audio_bps=audio_bps,
        max_bpp=max_bpp,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)

    def encode_once(
        out_path: Path,
        pass_rate: int,
    ) -> int:
        err = tempfile.TemporaryFile()

        enc = subprocess.Popen(
            [
                get_ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s",
                f"{w}x{h}",
                "-framerate",
                f"{fps:.6f}",
                "-i",
                "-",

                "-vn",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(src),

                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",

                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-b:v",
                str(pass_rate),
                "-maxrate",
                str(pass_rate),
                "-bufsize",
                str(max(64_000, int(pass_rate * VBV_SECONDS))),

                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-bf",
                "3",
                "-refs",
                "3",
                "-g",
                str(gop),
                "-keyint_min",
                str(max(1, int(round(fps)))),
                "-sc_threshold",
                "0",
                "-x264-params",
                "aq-mode=1:"
                "aq-strength=0.8:"
                "rc-lookahead=30:"
                "deblock=0,0:"
                "repeat-headers=1:"
                "aud=1:"
                "bframes=3:"
                "open-gop=0:"
                "8x8dct=1:"
                "cabac=1",

                "-tag:v",
                "avc1",

                "-c:a",
                "aac",
                "-b:a",
                str(audio_bps),
                "-ar",
                "48000",
                "-ac",
                "2",

                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-sn",
                "-dn",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                "-t",
                f"{duration:.3f}",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
            stderr=err,
            bufsize=1024 * 1024,
        )

        frames = 0
        cached_group = -1
        add_plane = None
        sub_plane = None

        try:
            for i, buf in _iter_frames(
                src,
                vf,
                w,
                h,
                max_seconds=max_seconds,
            ):
                group_idx = i // max(1, group)

                # IMPORTANT:
                # _iter_frames reuses the same bytearray for every frame.
                # Never let OpenCV or numpy operate on a view that can be
                # unexpectedly aliased by FFmpeg's pipe buffer.
                #
                # Make a dedicated writable frame buffer before applying
                # the watermark. This also prevents OpenCV from receiving
                # a non-contiguous/read-only view on some numpy/OpenCV builds.
                frame = bytearray(buf)

                y = np.frombuffer(
                    frame,
                    dtype=np.uint8,
                    count=w * h,
                ).reshape(h, w)

                if group_idx != cached_group:
                    add_plane, sub_plane = _delta_planes(
                        y,
                        bits,
                        key,
                        reveal_id,
                        group_idx,
                        amp,
                    )
                    cached_group = group_idx

                # Do not use dst=y. Some OpenCV builds reject a numpy view
                # backed by a Python bytearray for an in-place operation.
                # Compute the result into a normal contiguous ndarray and
                # copy it back into the Y plane.
                added = cv2.add(
                    y,
                    add_plane,
                )
                watermarked = cv2.subtract(
                    added,
                    sub_plane,
                )

                y[:, :] = watermarked

                try:
                    enc.stdin.write(frame)
                except BrokenPipeError:
                    break

                frames += 1

            try:
                enc.stdin.close()
            except BrokenPipeError:
                pass

            rc = enc.wait(
                timeout=max(
                    90,
                    int(duration * 8),
                )
            )

        except subprocess.TimeoutExpired:
            enc.kill()
            enc.wait()
            raise RuntimeError(
                "FFmpeg video encode timed out."
            )

        finally:
            if enc.poll() is None:
                enc.kill()
                enc.wait()

            err.seek(0)
            error_text = err.read().decode(
                "utf-8",
                "replace",
            )[-3000:]
            err.close()

        if (
            rc != 0
            or frames == 0
            or not out_path.exists()
            or out_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                "FFmpeg encode failed"
                + (
                    f": {error_text}"
                    if error_text
                    else "."
                )
            )

        return frames

    attempts = [rate]

    if target:
        retry_rate = max(
            MIN_VIDEO_BPS,
            int(rate * 0.82),
        )

        if retry_rate < rate:
            attempts.append(retry_rate)

    last_size = 0

    for attempt_no, attempt_rate in enumerate(
        attempts,
        1,
    ):
        tmp_out = dst.with_name(
            f"{dst.stem}.encode{attempt_no}.tmp{dst.suffix}"
        )

        try:
            encode_once(
                tmp_out,
                attempt_rate,
            )

            last_size = tmp_out.stat().st_size

            if not target or last_size <= target:
                tmp_out.replace(dst)

                return {
                    "width": w,
                    "height": h,
                    "fps": round(fps, 3),
                    "bitrate": attempt_rate,
                    "attempts": attempt_no,
                    "size": last_size,
                }

        finally:
            tmp_out.unlink(missing_ok=True)

    raise RuntimeError(
        "Video could not be encoded below the delivery limit "
        f"({last_size / 1048576:.1f} MiB generated, "
        f"target {target / 1048576:.1f} MiB)."
    )


def _decode_rgb_frame(
    buf: bytearray,
    w: int,
    h: int,
) -> np.ndarray:
    return np.frombuffer(
        buf,
        np.uint8,
        count=w * h,
    ).reshape(h, w).copy()


def _orb(
    g: np.ndarray,
    nfeatures: int = 700,
):
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        fastThreshold=8,
    )
    return orb.detectAndCompute(g, None)


def _visual_signature(
    gray: np.ndarray,
    width: int = 64,
    height: int = 36,
) -> np.ndarray:
    small = cv2.resize(
        gray,
        (width, height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    small -= small.mean()
    norm = np.linalg.norm(small)

    if norm > 1e-6:
        small /= norm

    return small


def _frame_difference(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    aa = _visual_signature(a)
    bb = _visual_signature(b)
    return float(np.mean(np.abs(aa - bb)))


def _read_video_frames(
    src: Path,
    max_height: int,
    *,
    max_seconds: int = 300,
) -> tuple[list[np.ndarray], float, str, int, int]:
    w, h, fps, vf = _probe_video(
        src,
        max_height,
    )

    frames: list[np.ndarray] = []

    for _, buf in _iter_frames(
        src,
        vf,
        w,
        h,
        max_seconds=max_seconds,
    ):
        frames.append(
            np.frombuffer(
                buf,
                np.uint8,
                count=w * h,
            )
            .reshape(h, w)
            .copy()
        )

    return frames, fps, vf, w, h


def _exact_video_frame(
    orig: Path,
    leak_rgb: np.ndarray,
    approx_frame: int,
    key: bytes,
    reveal_id: str,
    max_height: int,
) -> Optional[tuple]:
    w, h, fps, vf = _probe_video(
        orig,
        max_height,
    )

    total = max(
        1,
        int(round(
            _video_duration_seconds(orig)
            * fps
        )),
    )

    candidates = sorted(
        set(
            max(
                0,
                min(
                    total - 1,
                    approx_frame + d,
                ),
            )
            for d in range(-8, 9)
        )
    )

    target = cv2.cvtColor(
        leak_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    for frame_no in candidates:
        for idx, buf in _iter_frames(
            orig,
            vf,
            w,
            h,
            skip=frame_no,
            limit=1,
        ):
            del idx

            fr = np.frombuffer(
                buf,
                np.uint8,
                count=w * h,
            ).reshape(h, w).copy()

            uid, metric, ok = _decode_registered(
                fr.astype(np.float32),
                target.astype(np.float32),
                key,
                reveal_id,
                frame_no // VIDEO_GROUP,
                _geometry_variants(h),
            )

            if ok:
                return uid, metric, frame_no

    return None


def _fast_direct_video_extract(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    max_height: int,
) -> Optional[tuple]:
    w, h, fps, vf = _probe_video(
        orig,
        max_height,
    )

    leak_rgb = np.asarray(
        _load_rgb(leak),
        np.uint8,
    )

    leak_gray = cv2.cvtColor(
        leak_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    best = None

    for idx, buf in _iter_frames(
        orig,
        vf,
        w,
        h,
        max_seconds=300,
    ):
        fr = np.frombuffer(
            buf,
            np.uint8,
            count=w * h,
        ).reshape(h, w).copy()

        if fr.shape != leak_gray.shape:
            candidate = cv2.resize(
                fr,
                (leak_gray.shape[1], leak_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            candidate = fr

        diff = _frame_difference(
            candidate,
            leak_gray,
        )

        if best is None or diff < best[0]:
            best = (diff, idx, fr)

    if best is None:
        return None

    _, idx, fr = best

    uid, metric, ok = _decode_registered(
        fr.astype(np.float32),
        leak_gray.astype(np.float32),
        key,
        reveal_id,
        idx // VIDEO_GROUP,
        _geometry_variants(h),
    )

    if ok:
        return uid, metric, idx

    return None


def extract_video_frame(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    *,
    cell: int = CELL,
    max_height: int = 1080,
) -> dict:
    del cell

    hit = _fast_direct_video_extract(
        orig,
        leak,
        key,
        reveal_id,
        max_height,
    )

    if hit:
        uid, metric, frame = hit
        return {
            "user_id": int(uid),
            "metric": float(metric),
            "frame": int(frame),
            "ok": True,
        }

    return {
        "user_id": 0,
        "metric": 0.0,
        "frame": 0,
        "ok": False,
    }


def extract_video(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    *,
    max_height: int = 1080,
) -> dict:
    leak_rgb = np.asarray(
        _load_rgb(leak),
        np.uint8,
    )

    hit = _fast_direct_video_extract(
        orig,
        leak,
        key,
        reveal_id,
        max_height,
    )

    if hit:
        uid, metric, frame = hit
        return {
            "user_id": int(uid),
            "metric": float(metric),
            "frame": int(frame),
            "time": float(frame / max(
                1.0,
                _probe_video(orig, max_height)[2],
            )),
            "ok": True,
        }

    frames, fps, vf, w, h = _read_video_frames(
        orig,
        max_height,
        max_seconds=300,
    )

    if not frames:
        return {
            "user_id": 0,
            "metric": 0.0,
            "frame": 0,
            "time": 0.0,
            "ok": False,
        }

    leak_gray = cv2.cvtColor(
        leak_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    best = None

    for idx, fr in enumerate(frames):
        if fr.shape != leak_gray.shape:
            candidate = cv2.resize(
                fr,
                (leak_gray.shape[1], leak_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            candidate = fr

        diff = _frame_difference(
            candidate,
            leak_gray,
        )

        if best is None or diff < best[0]:
            best = (diff, idx, fr)

    if best is None:
        return {
            "user_id": 0,
            "metric": 0.0,
            "frame": 0,
            "time": 0.0,
            "ok": False,
        }

    _, idx, fr = best

    uid, metric, ok = _decode_registered(
        fr.astype(np.float32),
        leak_gray.astype(np.float32),
        key,
        reveal_id,
        idx // VIDEO_GROUP,
        _geometry_variants(h),
    )

    return {
        "user_id": int(uid) if ok else 0,
        "metric": float(metric),
        "frame": int(idx),
        "time": float(idx / max(fps, 1.0)),
        "ok": bool(ok),
    }


def extract(
    orig: Path,
    leak: Path,
    kind: str,
    key: bytes,
    reveal_id: str,
    *,
    image_cell: int = CELL,
    video_cell: int = CELL,
    max_height: int = 1080,
) -> dict:
    if kind == "image":
        return extract_image(
            orig,
            leak,
            key,
            reveal_id,
            image_cell,
        )

    if kind == "video_frame":
        return extract_video_frame(
            orig,
            leak,
            key,
            reveal_id,
            cell=video_cell,
            max_height=max_height,
        )

    if kind == "video":
        return extract_video(
            orig,
            leak,
            key,
            reveal_id,
            max_height=max_height,
        )

    raise ValueError(
        f"Unsupported extraction kind: {kind}"
    )
