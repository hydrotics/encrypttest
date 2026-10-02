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
MATCH_H = 480
COARSE_H = 320
VIDEO_MAX_FPS = 30.0
TRACE_COARSE_FPS = 2.0
TRACE_MAX_SAMPLES = 12
TRACE_FRAME_HINTS = 8
TRACE_FRAME_OFFSETS_SEC = (0.0, -0.25, 0.25, -0.55, 0.55, -0.95, 0.95)




VIDEO_FPS_HYPOTHESES = (30.0, 27.0, 24.0, 20.0, 18.0, 15.0, 12.0)

_x, _y = np.mgrid[:CELL, :CELL].astype(np.float32)
_CARRIER = (
    np.cos(np.pi * (_y + .5) / CELL) * np.cos(np.pi * 2 * (_x + .5) / CELL)
    + .8 * np.cos(np.pi * 2 * (_y + .5) / CELL) * np.cos(np.pi * (_x + .5) / CELL)
).astype(np.float32)
_CARRIER /= np.sqrt(np.mean(_CARRIER * _CARRIER)) + 1e-8
RGB2Y = np.array([0.299, 0.587, 0.114], np.float32)

MAX_BITS_PER_PIXEL = 0.11
MIN_VIDEO_BPS = 128_000
CONTAINER_MARGIN = 0.975



def get_ffmpeg() -> str:
    getter = getattr(imageio_ffmpeg, "get_ffmpeg_exe", None)
    if getter is None:
        getter = getattr(imageio_ffmpeg, "get_exe", None)
    if getter is None:
        raise RuntimeError(
            "This imageio-ffmpeg installation does not expose an FFmpeg executable getter."
        )
    exe = getter()
    if not exe:
        raise RuntimeError("No usable FFmpeg executable was found.")
    return str(exe)


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
    p = _pad(leak - orig.astype(np.float32), ch * CELL, cw * CELL)
    vm = _pad(valid, ch * CELL, cw * CELL)
    b = p.reshape(ch, CELL, cw, CELL).transpose(0, 2, 1, 3)
    vb = vm.reshape(ch, CELL, cw, CELL).transpose(0, 2, 1, 3)
    weight = vb.sum((2, 3))
    b -= (b * vb).sum((2, 3), keepdims=True) / np.maximum(weight[:, :, None, None], 1e-5)
    smooth = cv2.GaussianBlur(
        b.reshape(ch * cw, CELL, CELL),
        (0, 0),
        1.15,
    ).reshape(ch, cw, CELL, CELL)
    b -= 0.30 * smooth
    carrier = _CARRIER[None, None]
    corr = (b * carrier * vb).sum((2, 3))
    den = np.sqrt(
        (vb * (carrier ** 2)).sum((2, 3))
        * (vb * (b ** 2)).sum((2, 3))
        + 1e-5
    )
    corr /= np.maximum(den, 1e-4)
    coverage = np.clip(vb.mean((2, 3)), 0, 1)
    vals = corr * _chips(key, reveal_id, group, ch, cw) * coverage
    vals[coverage < 0.25] = 0.0
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


def _homography_is_reasonable(
    hmat: np.ndarray,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> bool:
    if hmat is None or not np.all(np.isfinite(hmat)):
        return False
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    pts = np.float32([
        [0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]
    ]).reshape(-1, 1, 2)
    try:
        q = cv2.perspectiveTransform(pts, hmat).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(q)):
        return False
    area = cv2.contourArea(q.astype(np.float32).reshape(-1, 1, 2))
    dst_area = float(max(dst_w * dst_h, 1))
    ratio = area / dst_area
    return 0.005 <= ratio <= 4.0 and np.max(np.abs(q)) < 10.0 * max(dst_w, dst_h)


def _affine_candidate(
    leak: np.ndarray,
    orig_shape: tuple[int, int],
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    partial: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    if partial:
        mat, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0,
            maxIters=4000, confidence=0.995, refineIters=20,
        )
    else:
        mat, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0,
            maxIters=4000, confidence=0.995, refineIters=20,
        )
    if mat is None or mask is None or int(mask.sum()) < 6:
        return None
    warped = cv2.warpAffine(
        leak, mat, (orig_shape[1], orig_shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    valid = cv2.warpAffine(
        np.ones(leak.shape[:2], np.float32), mat, (orig_shape[1], orig_shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    return warped, valid


def _ecc_refine(
    orig_rgb: np.ndarray,
    warped_rgb: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        scale = min(1.0, MATCH_H / max(orig_rgb.shape[:2]))
        orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY) if orig_rgb.ndim == 3 else orig_rgb
        warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY) if warped_rgb.ndim == 3 else warped_rgb
        if scale < 1.0:
            size = (max(32, round(orig_rgb.shape[1] * scale)), max(32, round(orig_rgb.shape[0] * scale)))
            template = cv2.resize(orig_gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            moving = cv2.resize(warped_gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            mask_small = cv2.resize(valid, size, interpolation=cv2.INTER_AREA).astype(np.uint8)
        else:
            template = orig_gray.astype(np.float32) / 255.0
            moving = warped_gray.astype(np.float32) / 255.0
            mask_small = (valid > 0.25).astype(np.uint8)

        template = cv2.GaussianBlur(template, (0, 0), 1.0)
        moving = cv2.GaussianBlur(moving, (0, 0), 1.0)
        matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        cc, matrix = cv2.findTransformECC(
            template, moving, matrix, cv2.MOTION_AFFINE, criteria,
            inputMask=mask_small if int(mask_small.sum()) > 100 else None,
            gaussFiltSize=5,
        )
        if not np.isfinite(cc):
            return warped_rgb, valid
        refined = cv2.warpAffine(
            warped_rgb, matrix, (orig_rgb.shape[1], orig_rgb.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        refined_valid = cv2.warpAffine(
            valid, matrix, (orig_rgb.shape[1], orig_rgb.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return refined, refined_valid
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return warped_rgb, valid


def _registration_candidates(
    orig: np.ndarray,
    leak: np.ndarray,
    *,
    refine_ecc: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    og, os = _feature_image(orig)
    lg, ls = _feature_image(leak)

    candidates: list[tuple[np.ndarray, np.ndarray]] = [
        (leak, np.ones(leak.shape[:2], np.float32))
    ]

    try:
        sift = cv2.SIFT_create(nfeatures=1400, contrastThreshold=0.015, edgeThreshold=10)
        ok, od = sift.detectAndCompute(og, None)
        lk, ld = sift.detectAndCompute(lg, None)

        if od is not None and ld is not None and len(ok) >= 8 and len(lk) >= 8:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            matches = matcher.knnMatch(ld, od, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

            if len(good) >= 6:
                src = np.float32([lk[m.queryIdx].pt for m in good])
                dst = np.float32([ok[m.trainIdx].pt for m in good])

                for partial in (False, True):
                    try:
                        cand = _affine_candidate(
                            leak,
                            orig.shape[:2],
                            src / max(ls, 1e-8),
                            dst / max(os, 1e-8),
                            partial,
                        )
                        if cand is not None:
                            candidates.append(cand)
                    except cv2.error:
                        pass

                hmat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=4000, confidence=0.995)
                if hmat is not None and mask is not None and int(mask.sum()) >= 6:
                    full = (
                        np.diag([1.0 / os, 1.0 / os, 1.0])
                        @ hmat
                        @ np.diag([ls, ls, 1.0])
                    )
                    if _homography_is_reasonable(full, leak.shape[:2], orig.shape[:2]):
                        warped = cv2.warpPerspective(
                            leak, full, (orig.shape[1], orig.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                        )
                        valid = cv2.warpPerspective(
                            np.ones(leak.shape[:2], np.float32), full,
                            (orig.shape[1], orig.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        )
                        candidates.append((warped, valid))
                        if refine_ecc:
                            refined, refined_valid = _ecc_refine(orig, warped, valid)
                            candidates.append((refined, refined_valid))
    except cv2.error:
        pass

    
    if len(candidates) < 3:
        try:
            orb = cv2.ORB_create(nfeatures=1600, fastThreshold=5)
            ok, od = orb.detectAndCompute(og, None)
            lk, ld = orb.detectAndCompute(lg, None)
            if od is not None and ld is not None and len(ok) >= 8 and len(lk) >= 8:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = matcher.knnMatch(ld, od, k=2)
                good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.78 * m[1].distance]
                if len(good) >= 8:
                    src = np.float32([lk[m.queryIdx].pt for m in good])
                    dst = np.float32([ok[m.trainIdx].pt for m in good])
                    hmat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=3000)
                    if hmat is not None and mask is not None and int(mask.sum()) >= 6:
                        full = (
                            np.diag([1.0 / os, 1.0 / os, 1.0])
                            @ hmat
                            @ np.diag([ls, ls, 1.0])
                        )
                        if _homography_is_reasonable(full, leak.shape[:2], orig.shape[:2]):
                            warped = cv2.warpPerspective(
                                leak, full, (orig.shape[1], orig.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                            )
                            valid = cv2.warpPerspective(
                                np.ones(leak.shape[:2], np.float32), full,
                                (orig.shape[1], orig.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            )
                            candidates.append((warped, valid))
        except cv2.error:
            pass

    if orig.shape[:2] != leak.shape[:2]:
        resized = cv2.resize(leak, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)
        candidates.append((resized, np.ones(orig.shape[:2], np.float32)))

    cleaned = []
    for cand, valid in candidates:
        if valid is not None and np.any(valid < 0.99):
            valid = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 0.8)
            valid[valid < 0.12] = 0.0
        cleaned.append((cand, valid))
    return cleaned


def _geometry_variants(
    h: int,
    w: int | None = None,
) -> list[tuple[float, float, float, float, float]]:
    w = int(w or h)
    scale = min(0.025, 8.0 / max(h, 1))
    trans_x = min(0.008, 4.0 / max(w, 1))
    trans_y = min(0.008, 4.0 / max(h, 1))
    return [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-scale, 0.0, 0.0, 0.0, 0.0),
        (scale, 0.0, 0.0, 0.0, 0.0),
        (0.0, -scale, 0.0, 0.0, 0.0),
        (0.0, scale, 0.0, 0.0, 0.0),
        (-scale, -scale, 0.0, 0.0, 0.0),
        (scale, scale, 0.0, 0.0, 0.0),
        (0.0, 0.0, -0.75, 0.0, 0.0),
        (0.0, 0.0, 0.75, 0.0, 0.0),
        (0.0, 0.0, -1.25, 0.0, 0.0),
        (0.0, 0.0, 1.25, 0.0, 0.0),
        (0.0, 0.0, 0.0, -trans_x, 0.0),
        (0.0, 0.0, 0.0, trans_x, 0.0),
        (0.0, 0.0, 0.0, 0.0, -trans_y),
        (0.0, 0.0, 0.0, 0.0, trans_y),
    ]


def _warp_y(
    y: np.ndarray,
    scale_x: float,
    scale_y: float,
    rotation: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> np.ndarray:
    h, w = y.shape

    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    theta = np.deg2rad(float(rotation))
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    sx = 1.0 + float(scale_x)
    sy = 1.0 + float(scale_y)

    matrix = np.array(
        [
            [
                c * sx,
                -s * sy,
                (1.0 - c * sx) * cx + s * sy * cy
                + float(translate_x) * w,
            ],
            [
                s * sx,
                c * sy,
                (1.0 - c * sy) * cy - s * sx * cx
                + float(translate_y) * h,
            ],
        ],
        dtype=np.float32,
    )

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
    variants,
    valid: np.ndarray | None = None,
) -> tuple[int, float, bool]:
    best_uid = 0
    best_metric = float("-inf")
    best_ok = False

    for variant in variants:
        if len(variant) == 2:
            sx, sy = variant
            rot = tx = ty = 0.0
        else:
            sx, sy, rot, tx, ty = variant
        warped = _warp_y(leak_y, sx, sy, rot, tx, ty)

        warped_valid = None
        if valid is not None:
            warped_valid = _warp_y(valid.astype(np.float32), sx, sy, rot, tx, ty)

        for candidate in (
            warped,
            cv2.GaussianBlur(warped, (3, 3), 0),
        ):
            scores = _score(oy, candidate, key, reveal_id, group, valid=warped_valid)
            uid, ok = decode_payload(scores)
            metric = float(np.mean(np.abs(scores)))
            if ok and metric > best_metric:
                best_uid = int(uid)
                best_metric = metric
                best_ok = True

    return best_uid, best_metric, best_ok


def _group_hypotheses(
    source_frame_idx: int,
    source_fps: float,
    delivery_fps: float | None = None,
) -> list[int]:
    """
    Return watermark-group hypotheses for a source frame.

    New embeds use the source timeline for the chip group. Legacy embeds used
    the delivered/output frame index. Screenshot tracing has no delivery FPS,
    so the legacy hypotheses cover the encoder's common FPS choices.
    """
    source_fps = max(float(source_fps), 1.0)
    source_frame_idx = max(0, int(source_frame_idx))

    ordered: list[int] = []
    seen: set[int] = set()

    def add_group(center: int) -> None:
        center = max(0, int(center))
        for delta in (0, -1, 1):
            value = center + delta
            if value < 0 or value in seen:
                continue
            seen.add(value)
            ordered.append(value)

    
    add_group(source_frame_idx // VIDEO_GROUP)

    if delivery_fps is not None:
        fps_values = [float(delivery_fps)]
    else:
        fps_values = [source_fps, *VIDEO_FPS_HYPOTHESES]

    unique_fps: list[float] = []
    for fps in fps_values:
        fps = float(fps)
        if fps <= 0 or fps > source_fps + 0.5:
            continue
        if not any(abs(fps - other) < 1e-3 for other in unique_fps):
            unique_fps.append(fps)

    
    for fps in unique_fps:
        delivered_frame_idx = int(
            round(source_frame_idx * fps / source_fps)
        )
        add_group(delivered_frame_idx // VIDEO_GROUP)

    return ordered


def embed_image_array(
    rgb: np.ndarray,
    user_id: int,
    key: bytes,
    reveal_id: str,
    amp: float = AMP,
) -> np.ndarray:
    h, w, _ = rgb.shape
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y = ycc[:, :, 0].astype(np.float32)
    d = _delta(y, encode_payload(user_id), key, reveal_id, 0, amp)

    y2 = np.clip(y + d, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(
        cv2.merge((y2, ycc[:, :, 1], ycc[:, :, 2])),
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
    candidates = _registration_candidates(o, l)
    variants = _geometry_variants(oy.shape[0], oy.shape[1])

    best = None
    for candidate, valid in candidates:
        if candidate.shape[:2] != oy.shape:
            candidate = cv2.resize(
                candidate, (oy.shape[1], oy.shape[0]), interpolation=cv2.INTER_AREA
            )
            valid = cv2.resize(
                valid.astype(np.float32), (oy.shape[1], oy.shape[0]), interpolation=cv2.INTER_AREA
            )
        candidate_y = candidate @ RGB2Y
        uid, metric, ok = _decode_registered(
            oy,
            candidate_y,
            key,
            reveal_id,
            0,
            variants,
            valid=valid,
        )
        if ok and (best is None or metric > best["metric"]):
            best = {"user_id": int(uid), "metric": metric, "ok": True}

    if best:
        return best
    return {"user_id": 0, "metric": 0.0, "ok": False}


def render_image_delivery(
    src: Path,
    dst: Path,
    user_id: int,
    key: bytes,
    reveal_id: str,
    *,
    amp: float = AMP,
    target_bytes: Optional[int] = None,
) -> dict:
    rgb = np.asarray(_load_rgb(src), np.uint8)
    watermarked = embed_image_array(rgb, user_id, key, reveal_id, amp)
    del rgb

    img = Image.fromarray(watermarked, "RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)

    qualities = (95, 92, 90, 88, 85, 82, 78)
    last_size = 0

    for quality in qualities:
        img.save(
            dst,
            "JPEG",
            quality=quality,
            optimize=True,
            progressive=False,
            subsampling=0,
        )
        last_size = dst.stat().st_size
        if not target_bytes or last_size <= int(target_bytes):
            return {
                "width": img.width,
                "height": img.height,
                "quality": quality,
                "size": last_size,
                "downscaled": False,
            }

    if target_bytes:
        scale = min(1.0, (float(target_bytes) / max(last_size, 1)) ** 0.5)
        for _ in range(5):
            if scale >= 0.999:
                break
            width = max(320, int(round(img.width * scale)))
            height = max(320, int(round(img.height * scale)))
            if width >= img.width or height >= img.height:
                scale *= 0.82
                continue
            resized = img.resize((width, height), Image.Resampling.LANCZOS)
            for quality in (90, 86, 82, 78, 74):
                resized.save(
                    dst,
                    "JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=False,
                    subsampling=0,
                )
                last_size = dst.stat().st_size
                if last_size <= int(target_bytes):
                    return {
                        "width": width,
                        "height": height,
                        "quality": quality,
                        "size": last_size,
                        "downscaled": True,
                    }
            scale *= 0.82

    raise RuntimeError(
        f"Image could not be encoded below the delivery limit "
        f"({last_size / 1048576:.1f} MiB generated, "
        f"target {int(target_bytes) / 1048576:.1f} MiB)."
    )


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


def _choose_video_preset(preset: str, duration: float) -> str:
    requested = (preset or "").strip().lower()
    if requested in {"fast", "medium", "slow", "slower", "veryslow"}:
        return requested
    if requested not in {"", "auto"}:
        return "fast"
    if duration >= 180:
        return "slow"
    if duration >= 90:
        return "medium"
    return "fast"


def _adaptive_video_shape(
    source_width: int,
    source_height: int,
    duration: float,
    requested_max: int,
    target_bytes: Optional[int],
    audio_bps: int,
    source_fps: float,
) -> tuple[int, float]:
    ceiling = max(144, min(int(source_height), int(requested_max)))
    source_fps = max(1.0, min(float(source_fps), 30.0))
    if not target_bytes or duration <= 0 or ceiling <= 360:
        return ceiling, source_fps

    usable_bits = max(
        1.0,
        float(target_bytes) * 8.0 * 0.955 - float(audio_bps) * float(duration),
    )
    average_video_bps = usable_bits / max(float(duration), 1.0)
    bpp_floor = 0.022

    fps_candidates = [source_fps]
    for f in (30.0, 27.0, 24.0, 20.0, 18.0, 15.0, 12.0):
        f = min(source_fps, f)
        if f >= 12.0:
            fps_candidates.append(round(f, 3))
    fps_candidates = sorted(set(fps_candidates), reverse=True)

    for candidate_h in (1080, 900, 720, 648, 576, 540, 480, 360):
        if candidate_h > ceiling:
            continue
        candidate_w = max(2, round(source_width * candidate_h / max(source_height, 1)))
        for candidate_fps in fps_candidates:
            required = candidate_w * candidate_h * candidate_fps * bpp_floor
            if average_video_bps >= required:
                return candidate_h, candidate_fps

    return min(480, ceiling), min(source_fps, 15.0)


def _quality_crf(
    *,
    width: int,
    height: int,
    fps: float,
    target: Optional[int],
    duration: float,
) -> float:
    if not target or duration <= 0:
        return 18.5 if height >= 900 else 19.0

    avg_video_bps = max(
        1.0,
        float(target) * 8.0 * 0.90 / max(float(duration), 1.0),
    )
    bpp = avg_video_bps / max(
        float(width * height) * max(float(fps), 1.0),
        1.0,
    )
    if bpp >= 0.090:
        return 17.5
    if bpp >= 0.070:
        return 18.0
    if bpp >= 0.055:
        return 18.5
    if bpp >= 0.043:
        return 19.0
    if bpp >= 0.034:
        return 20.0
    if bpp >= 0.028:
        return 21.0
    if bpp >= 0.023:
        return 22.0
    return 23.0


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
    del cell, max_bpp
    duration = min(float(max_seconds), max(0.0, _video_duration_seconds(src)))
    if duration <= 0:
        duration = float(max_seconds)

    target = max(1, int(target_bytes)) if target_bytes else None
    bits = encode_payload(user_id)

    requested_audio = max(64, min(128, int(audio_kbps)))
    if target and duration >= 180:
        requested_audio = min(requested_audio, 64)
    elif target and duration >= 90:
        requested_audio = min(requested_audio, 80)
    audio_bps = requested_audio * 1000

    source_w, source_h, source_fps, _ = _probe_video(src, max_height)
    delivery_height, delivery_fps = _adaptive_video_shape(
        source_w, source_h, duration, max_height, target, audio_bps, source_fps
    )

    selected_preset = _choose_video_preset(preset, duration)
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_size = 0
    last_profile = None

    def encode_once(
        out_path: Path,
        w: int,
        h: int,
        fps: float,
        crf: Optional[float] = None,
        video_bitrate: Optional[int] = None,
    ) -> int:
        vf = f"fps={fps:.6f}," + _scale_filter(h, "lanczos")
        if video_bitrate is not None:
            rate_args = [
                "-b:v", str(int(video_bitrate)),
                "-maxrate", str(int(video_bitrate * 1.05)),
                "-bufsize", str(max(64_000, int(video_bitrate * 2.0))),
            ]
        else:
            if crf is None:
                raise ValueError("Either crf or video_bitrate must be supplied")
            rate_args = ["-crf", f"{float(crf):.2f}"]
        gop = max(1, min(120, int(round(fps * 2.0))))
        err = tempfile.TemporaryFile()
        enc = subprocess.Popen(
            [
                get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{w}x{h}", "-framerate", f"{fps:.6f}", "-i", "-",
                "-t", f"{duration:.3f}", "-i", str(src),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", selected_preset,
                *rate_args,
                "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
                "-g", str(gop),
                "-x264-params", "aq-mode=1:aq-strength=0.9:open-gop=0",
                "-tag:v", "avc1",
                "-c:a", "aac", "-b:a", str(audio_bps), "-ar", "48000", "-ac", "2",
                "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
                "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
                "-t", f"{duration:.3f}", str(out_path),
            ],
            stdin=subprocess.PIPE,
            stderr=err,
            bufsize=1024 * 1024,
        )

        frames = 0
        cached_group = -1
        add_plane = sub_plane = None
        try:
            for i, buf in _iter_frames(src, vf, w, h, max_seconds=max_seconds):
                
                
                
                source_frame_idx = int(
                    round(
                        i * source_fps / max(float(fps), 1e-6)
                    )
                )
                group_idx = source_frame_idx // max(1, int(group))

                frame = bytearray(buf)
                y = np.frombuffer(frame, np.uint8, count=w * h).reshape(h, w)

                if group_idx != cached_group:
                    add_plane, sub_plane = _delta_planes(
                        y, bits, key, reveal_id, group_idx, amp
                    )
                    cached_group = group_idx

                y[:, :] = cv2.subtract(cv2.add(y, add_plane), sub_plane)
                try:
                    enc.stdin.write(frame)
                except BrokenPipeError:
                    break
                frames += 1

            try:
                enc.stdin.close()
            except BrokenPipeError:
                pass
            rc = enc.wait(timeout=max(120, int(duration * 8)))
        except subprocess.TimeoutExpired as exc:
            enc.kill(); enc.wait()
            raise RuntimeError("FFmpeg video encode timed out.") from exc
        finally:
            if enc.poll() is None:
                enc.kill(); enc.wait()
            err.seek(0)
            error_text = err.read().decode("utf-8", "replace")[-4000:]
            err.close()

        if rc != 0 or frames == 0 or not out_path.exists() or out_path.stat().st_size <= 0:
            raise RuntimeError(
                "FFmpeg encode failed" + (f": {error_text}" if error_text else ".")
            )
        return frames

    height_candidates = [delivery_height]
    for candidate in (1080, 900, 720, 648, 576, 540, 480, 360):
        if candidate < delivery_height and candidate <= source_h and candidate <= max_height:
            height_candidates.append(candidate)
    height_candidates = list(dict.fromkeys(height_candidates))[:4]

    for profile_no, height in enumerate(height_candidates, 1):
        _, h0, probed_fps, _ = _probe_video(src, int(height))
        base_fps = min(float(probed_fps), float(delivery_fps), 30.0)
        if profile_no > 1:
            base_fps = min(base_fps, 24.0 if height <= 720 else 27.0)

        fps_candidates = []
        for f in (base_fps, 27.0, 24.0, 20.0, 18.0, 15.0):
            f = min(base_fps, f)
            if f >= 12.0:
                fps_candidates.append(round(float(f), 3))
        fps_candidates = list(dict.fromkeys(fps_candidates))

        for fps in fps_candidates:
            w, h, _, _ = _probe_video(src, int(height))
            crf = _quality_crf(
                width=w, height=h, fps=fps, target=target, duration=duration
            )

            for attempt in range(3):
                tmp_out = dst.with_name(
                    f"{dst.stem}.p{profile_no}_{int(round(fps))}_{attempt}.tmp{dst.suffix}"
                )
                try:
                    encode_once(tmp_out, w, h, fps, crf=crf)
                    last_size = tmp_out.stat().st_size
                    last_profile = (w, h, fps, crf)
                    if not target or last_size <= target:
                        tmp_out.replace(dst)
                        return {
                            "width": w, "height": h, "fps": round(fps, 3),
                            "crf": round(crf, 2), "preset": selected_preset,
                            "attempts": attempt + 1, "profile_attempt": profile_no,
                            "size": last_size,
                        }

                    ratio = max(1.01, last_size / max(target, 1))
                    crf = min(31.0, max(crf + 0.5, crf + 2.5 * float(np.log2(ratio))))
                finally:
                    tmp_out.unlink(missing_ok=True)

    if target and last_profile:
        w, h, fps, _crf = last_profile
        audio_bits = audio_bps * duration
        usable_bits = max(1, int(target * 8 * 0.86 - audio_bits))
        rescue_bps = max(MIN_VIDEO_BPS, int(usable_bits / max(duration, 1.0)))

        for rescue_no, scale in enumerate((1.0, 0.82), 1):
            rescue_rate = max(MIN_VIDEO_BPS, int(rescue_bps * scale))
            tmp_out = dst.with_name(
                f"{dst.stem}.rescue{rescue_no}.tmp{dst.suffix}"
            )
            try:
                encode_once(
                    tmp_out,
                    w,
                    h,
                    fps,
                    video_bitrate=rescue_rate,
                )
                last_size = tmp_out.stat().st_size
                if last_size <= target:
                    tmp_out.replace(dst)
                    return {
                        "width": w,
                        "height": h,
                        "fps": round(fps, 3),
                        "bitrate": rescue_rate,
                        "preset": selected_preset,
                        "attempts": rescue_no,
                        "rescue": True,
                        "size": last_size,
                    }
            finally:
                tmp_out.unlink(missing_ok=True)

    profile_text = (
        f"last profile {last_profile}, last size {last_size / 1048576:.1f} MiB"
        if last_profile else "no successful encode"
    )
    raise RuntimeError(
        "Video could not be encoded below the delivery limit "
        f"({profile_text}, target {target / 1048576:.1f} MiB)."
    )

def _trace_heights(orig: Path, leak_shape: tuple[int, int], max_height: int) -> list[int]:
    source_w, source_h, _, _ = _probe_video(orig, max_height)
    limit = min(source_h, int(max_height))
    leak_h = max(144, int(leak_shape[0]))
    common = [1080, 900, 768, 720, 648, 576, 540, 480, 360]
    values = [min(leak_h, limit)]
    values.extend(v for v in common if v <= limit)
    values = sorted(set(max(144, int(v)) for v in values), key=lambda v: abs(v - leak_h))
    return values

def _seek_frame(
    src: Path,
    time_s: float,
    max_height: int,
) -> tuple[np.ndarray, float, float, int, int]:
    w, h, fps, _ = _probe_video(src, max_height)
    t = max(0.0, float(time_s))
    vf = _scale_filter(h, "lanczos")
    cmd = [
        get_ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{t:.4f}",
        "-i", str(src),
        "-an",
        "-vf", vf,
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    expected = w * h * 3
    if len(proc.stdout) < expected:
        detail = proc.stderr.decode("utf-8", "replace")[-1000:]
        raise RuntimeError(f"FFmpeg could not seek video frame. {detail}".strip())
    frame = np.frombuffer(proc.stdout[:expected], np.uint8).reshape(h, w, 3).copy()
    return frame, fps, t, w, h



def _coarse_hits(
    orig: Path,
    leak_rgb: np.ndarray,
    max_height: int,
    max_seconds: int = 300,
    top_k: int = TRACE_FRAME_HINTS,
) -> list[float]:
    coarse_height = max(144, min(int(max_height), 360))
    source_w, source_h, source_fps, _ = _probe_video(orig, coarse_height)
    coarse_fps = min(float(TRACE_COARSE_FPS), max(float(source_fps), 1.0))
    vf = f"fps={coarse_fps:.6f}," + _scale_filter(coarse_height, "fast_bilinear")
    leak_gray = cv2.cvtColor(leak_rgb, cv2.COLOR_RGB2GRAY)
    leak_sig = _visual_signature(leak_gray)
    _, leak_desc = _sift_descriptors(leak_gray)
    ranked: list[tuple[float, float]] = []
    for idx, buf in _iter_frames(orig, vf, source_w, source_h, max_seconds=max_seconds):
        gray = np.frombuffer(buf, np.uint8, count=source_w * source_h).reshape(source_h, source_w)
        diff = float(np.mean(np.abs(_visual_signature(gray) - leak_sig)))
        sift_score = 0.0
        if leak_desc is not None and idx % 2 == 0:
            _, frame_desc = _sift_descriptors(gray)
            sift_score = _sift_match_count(leak_desc, frame_desc)
        rank = diff - min(sift_score, 30.0) * 0.004
        ranked.append((rank, idx / max(coarse_fps, 1e-6)))
        if len(ranked) > top_k * 6:
            ranked.sort(key=lambda x: x[0])
            del ranked[top_k * 4:]
    ranked.sort(key=lambda x: x[0])
    times: list[float] = []
    for _, time_s in ranked:
        if all(abs(time_s - old) > 0.30 for old in times):
            times.append(float(time_s))
        if len(times) >= top_k:
            break
    return times

def _registration_decode(
    orig_rgb: np.ndarray,
    leak_rgb: np.ndarray,
    key: bytes,
    reveal_id: str,
    groups: list[int],
) -> Optional[tuple[int, float]]:
    oy = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    direct = leak_rgb
    if direct.shape[:2] != orig_rgb.shape[:2]:
        direct = cv2.resize(
            direct,
            (orig_rgb.shape[1], orig_rgb.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    variants = [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.004, -0.004, 0.0, 0.0, 0.0),
        (0.004, 0.004, 0.0, 0.0, 0.0),
    ]
    direct_y = cv2.cvtColor(direct, cv2.COLOR_RGB2GRAY).astype(np.float32)
    for group in groups:
        uid, metric, ok = _decode_registered(
            oy, direct_y, key, reveal_id, group, variants
        )
        if ok:
            return int(uid), float(metric)
    candidates = _registration_candidates(
        orig_rgb,
        leak_rgb,
        refine_ecc=True,
    )
    for candidate, valid in candidates:
        if candidate.shape[:2] != orig_rgb.shape[:2]:
            candidate = cv2.resize(
                candidate,
                (orig_rgb.shape[1], orig_rgb.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            valid = cv2.resize(
                valid.astype(np.float32),
                (orig_rgb.shape[1], orig_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        leak_y = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
        for group in groups:
            uid, metric, ok = _decode_registered(
                oy, leak_y, key, reveal_id, group, variants, valid=valid
            )
            if ok:
                return int(uid), float(metric)
    return None


def _visual_signature(gray: np.ndarray, width: int = 48, height: int = 27) -> np.ndarray:
    small = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    norm = np.linalg.norm(small)
    if norm > 1e-6:
        small /= norm
    return small


def _sift_descriptors(gray: np.ndarray):
    small, _ = _feature_image(gray, COARSE_H)
    sift = cv2.SIFT_create(
        nfeatures=300,
        contrastThreshold=0.04,
        edgeThreshold=10,
    )
    return sift.detectAndCompute(small, None)


def _sift_match_count(query_desc: np.ndarray | None, frame_desc: np.ndarray | None) -> float:
    if query_desc is None or frame_desc is None or len(query_desc) < 5 or len(frame_desc) < 8:
        return 0.0
    try:
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_desc, frame_desc, k=2)
    except cv2.error:
        return 0.0
    return float(sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.78 * m[1].distance))


def _group_hypotheses(
    source_frame_idx: int,
    source_fps: float,
    delivery_fps: float | None = None,
) -> list[int]:
    source_fps = max(1.0, float(source_fps))
    source_frame_idx = max(0, int(source_frame_idx))
    values: list[int] = []
    seen: set[int] = set()
    def add(value: int) -> None:
        value = max(0, int(value))
        for item in (value, value - 1, value + 1):
            if item >= 0 and item not in seen:
                seen.add(item)
                values.append(item)
    add(source_frame_idx // VIDEO_GROUP)
    rates = [float(delivery_fps)] if delivery_fps else [source_fps, *VIDEO_FPS_HYPOTHESES]
    for rate in rates:
        rate = max(1.0, float(rate))
        if rate > source_fps + 0.5:
            continue
        delivered_idx = int(round(source_frame_idx * rate / source_fps))
        add(delivered_idx // VIDEO_GROUP)
    return values



def _direct_decode(
    orig_rgb: np.ndarray,
    leak_rgb: np.ndarray,
    key: bytes,
    reveal_id: str,
    groups: list[int],
) -> Optional[tuple[int, float]]:
    oy = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    leak = leak_rgb
    if leak.shape[:2] != orig_rgb.shape[:2]:
        leak = cv2.resize(
            leak,
            (orig_rgb.shape[1], orig_rgb.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    leak_y = cv2.cvtColor(leak, cv2.COLOR_RGB2GRAY).astype(np.float32)
    variants = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    for group in groups:
        uid, metric, ok = _decode_registered(
            oy, leak_y, key, reveal_id, group, variants
        )
        if ok:
            return int(uid), float(metric)
    return None


def _exact_video_frame(
    orig: Path,
    leak_rgb: np.ndarray,
    time_s: float,
    key: bytes,
    reveal_id: str,
    max_height: int,
) -> Optional[tuple[int, float, int, float]]:
    offsets = TRACE_FRAME_OFFSETS_SEC
    heights = _trace_heights(orig, leak_rgb.shape[:2], max_height)
    fallback: list[tuple[np.ndarray, float, float, int, int, list[int]]] = []
    for height in heights:
        for offset in offsets:
            candidate_time = max(0.0, float(time_s) + float(offset))
            try:
                frame, source_fps, actual_time, _, _ = _seek_frame(
                    orig, candidate_time, height
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                continue
            source_idx = int(round(actual_time * source_fps))
            groups = _group_hypotheses(source_idx, source_fps)
            direct = _direct_decode(
                frame, leak_rgb, key, reveal_id, groups
            )
            if direct is not None:
                uid, metric = direct
                return int(uid), float(metric), source_idx, actual_time
            fallback.append((frame, source_fps, actual_time, source_idx, height, groups))
    for frame, source_fps, actual_time, source_idx, height, groups in fallback[:2]:
        try:
            hit = _registration_decode(
                frame, leak_rgb, key, reveal_id, groups
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        if hit is not None:
            uid, metric = hit
            return int(uid), float(metric), source_idx, actual_time
    return None

def extract_video_frame(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    *,
    cell: int = CELL,
    max_height: int = 1080,
    start_frame: int = 0,
) -> dict:
    del cell
    leak_rgb = np.asarray(_load_rgb(leak), np.uint8)
    leak_gray = cv2.cvtColor(leak_rgb, cv2.COLOR_RGB2GRAY)
    source_fps = _probe_video(orig, max_height)[2]
    if start_frame > 0:
        times = [float(start_frame) / max(source_fps, 1.0)]
    else:
        times = _coarse_hits(orig, leak_rgb, max_height)
    if not times:
        times = [0.0]
    for time_s in times[:TRACE_FRAME_HINTS]:
        hit = _exact_video_frame(
            orig,
            leak_rgb,
            time_s,
            key,
            reveal_id,
            max_height,
        )
        if hit is None:
            continue
        uid, metric, frame, actual_time = hit
        return {
            "user_id": int(uid),
            "metric": float(metric),
            "frame": int(frame),
            "time": float(actual_time),
            "ok": True,
            "valid": True,
        }
    return _fail(frame=0, time=0.0)


def _sample_leak_video(
    leak: Path,
    max_height: int,
    count: int = 6,
) -> tuple[list[tuple[int, np.ndarray]], float]:
    w, h, fps, vf = _probe_video(leak, max_height)
    total = max(1, int(round(_video_duration_seconds(leak) * fps)))
    count = max(1, min(int(count), TRACE_MAX_SAMPLES))
    step = max(1, total // count)
    wanted = sorted({min(total - 1, i * step + step // 2) for i in range(count)})
    out: list[tuple[int, np.ndarray]] = []
    for idx, buf in _iter_frames(leak, vf, w, h):
        if idx in wanted:
            frame = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w).copy()
            out.append((idx, frame))
        if idx >= wanted[-1]:
            break
    return out, fps


def _match_and_decode(
    orig: Path,
    samples: list[tuple[int, np.ndarray]],
    leak_fps: float,
    key: bytes,
    reveal_id: str,
    max_height: int,
    max_seconds: int = 300,
) -> Optional[tuple]:
    if not samples:
        return None
    w, h, source_fps, vf = _probe_video(orig, max_height)
    refs = [_visual_signature(frame) for _, frame in samples]
    best: list[Optional[tuple]] = [None] * len(samples)
    for idx, buf in _iter_frames(orig, vf, w, h, max_seconds=max_seconds):
        fr = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w)
        sig = _visual_signature(fr)
        for k, ref in enumerate(refs):
            diff = float(np.mean(np.abs(sig - ref)))
            if best[k] is None or diff < best[k][0]:
                best[k] = (diff, idx, fr.copy())
    for k, hit in enumerate(best):
        if hit is None:
            continue
        _, source_idx, fr = hit
        leak_idx, leak_frame = samples[k]
        leak = leak_frame
        groups = _group_hypotheses(source_idx, source_fps, leak_fps)
        decoded = _registration_decode(
            cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(leak.astype(np.uint8), cv2.COLOR_GRAY2RGB),
            key,
            reveal_id,
            groups,
        )
        if decoded is not None:
            uid, metric = decoded
            return int(uid), float(metric), int(source_idx)
    return None


def extract_video(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    *,
    max_height: int = 1080,
    start_frame: int = 0,
) -> dict:
    del start_frame
    samples, leak_fps = _sample_leak_video(leak, max_height)
    hit = _match_and_decode(
        orig,
        samples,
        leak_fps,
        key,
        reveal_id,
        max_height,
    )
    if hit is None:
        return _fail(frame=0, time=0.0)
    uid, metric, frame = hit
    source_fps = _probe_video(orig, max_height)[2]
    return {
        "user_id": int(uid),
        "metric": float(metric),
        "frame": int(frame),
        "time": float(frame / max(source_fps, 1.0)),
        "ok": True,
        "valid": True,
    }


def _fail(**extra) -> dict:
    return {"user_id": 0, "metric": 0.0, "ok": False, "valid": False, **extra}


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
    start_frame: int = 0,
) -> dict:
    if kind == "image":
        res = extract_image(orig, leak, key, reveal_id, image_cell)
    elif kind == "video_frame":
        res = extract_video_frame(
            orig,
            leak,
            key,
            reveal_id,
            cell=video_cell,
            max_height=max_height,
            start_frame=start_frame,
        )
    elif kind == "video":
        res = extract_video(
            orig,
            leak,
            key,
            reveal_id,
            max_height=max_height,
            start_frame=start_frame,
        )
    else:
        raise ValueError(f"Unsupported extraction kind: {kind}")
    res["valid"] = bool(res.get("valid", res.get("ok", False)))
    return res
