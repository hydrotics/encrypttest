from __future__ import annotations

import hashlib
import hmac
import io
import itertools
import logging
import os
import re
import subprocess
import tempfile
import time
import zlib
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

                                                                               
                                                                           
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2              
import imageio_ffmpeg              
import numpy as np              
from PIL import Image              

log = logging.getLogger(__name__)

                                                                             
           
                                                                             

N_BITS = 96
CELL = 16
AMP = 4.0
VIDEO_GROUP = 8
MATCH_H = 960                                                                
COARSE_H = 640                                                                   
COARSE_FEATURE_DIM = 1024                                              
COARSE_FPS = 0.50
VIDEO_MAX_FPS = 30.0
TRACE_MAX_SAMPLES = 4

WINDOW_HALF_SEC = 1.5                                                       
WINDOW_TOP_FRAMES = 6                                                            
WINDOW_READ_TIMEOUT = 60.0
DEFAULT_TIME_BUDGET = 90.0                                                   
ECC_MIN_CC = 0.5

                                                                
CV_THREADS = max(1, int(os.getenv("CV_THREADS", "1")))
FFMPEG_THREADS = max(1, int(os.getenv("FFMPEG_THREADS", "1")))
cv2.setNumThreads(CV_THREADS)

                                                                                    
                        
TRACE_IMAGE_MAX_FEATURES = 1100
TRACE_IMAGE_MAX_CANDIDATES = 2
TRACE_IMAGE_ECC = True
TRACE_IMAGE_TRANSLATION_RADIUS = 2
TRACE_IMAGE_DENSE_SHIFTS = False                                                                      
TRACE_IMAGE_MAX_PIXELS = int(float(os.getenv("TRACE_IMAGE_MAX_MEGAPIXELS", "18")) * 1_000_000)
EMBED_IMAGE_MAX_PIXELS = int(float(os.getenv("EMBED_IMAGE_MAX_MEGAPIXELS", "40")) * 1_000_000)
EMBED_BAND_ROWS = 512                                          
assert EMBED_BAND_ROWS % CELL == 0

VIDEO_FPS_HYPOTHESES = (30.0, 29.97, 27.0, 25.0, 24.0, 23.976, 20.0, 18.0, 15.0, 12.0)
HEIGHT_LADDER = (1080, 900, 720, 648, 576, 540, 480, 360)

MIN_VIDEO_BPS = 128_000
USE_VBV_CAP = True                                                                              

                                                                          
                                                                         
FFMPEG_FORMAT_WHITELIST = os.getenv(
    "FFMPEG_FORMAT_WHITELIST",
    "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,m4v",
)

_x, _y = np.mgrid[:CELL, :CELL].astype(np.float32)
_CARRIER = (
    np.cos(np.pi * (_y + .5) / CELL) * np.cos(np.pi * 2 * (_x + .5) / CELL)
    + .8 * np.cos(np.pi * 2 * (_y + .5) / CELL) * np.cos(np.pi * (_x + .5) / CELL)
).astype(np.float32)
_CARRIER /= np.sqrt(np.mean(_CARRIER * _CARRIER)) + 1e-8
_CARRIER_SQ = (_CARRIER * _CARRIER).astype(np.float32)
_CARRIER_SQ_SUM = np.float32(_CARRIER_SQ.sum())

_ORIENT = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


@lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not exe:
        raise RuntimeError("No usable FFmpeg executable was found.")
    return str(exe)


def _ff() -> list[str]:
    return [get_ffmpeg(), "-hide_banner", "-loglevel", "error"]


def _in(path) -> list[str]:
    """Input options for an untrusted media file (threads, protocol + demuxer limits)."""
    opts = ["-threads", str(FFMPEG_THREADS), "-protocol_whitelist", "file,pipe"]
    if FFMPEG_FORMAT_WHITELIST:
        opts += ["-format_whitelist", FFMPEG_FORMAT_WHITELIST]
    return [*opts, "-i", str(path)]


def _fail(**extra) -> dict:
    return {"user_id": 0, "metric": 0.0, "ok": False, "valid": False, **extra}


                                                                             
         
                                                                             

def encode_payload(user_id: int) -> np.ndarray:
    uid = int(user_id) & ((1 << 64) - 1)
    crc = zlib.crc32(uid.to_bytes(8, "big"))
    value = (uid << 32) | crc
    return np.array(
        [(value >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS)],
        np.float32,
    ) * 2.0 - 1.0


def _crc_ok(v: int) -> bool:
    return zlib.crc32((v >> 32).to_bytes(8, "big")) == (v & 0xFFFFFFFF)


def decode_payload(scores: np.ndarray) -> tuple[int, bool]:
    """Hard-decide the bits, then try flipping the 1-3 least confident of the 12 weakest."""
    s = np.asarray(scores)
    value = int.from_bytes(np.packbits((s > 0).astype(np.uint8)).tobytes(), "big")           
    if _crc_ok(value):
        return value >> 32, True

    masks = [1 << (N_BITS - 1 - int(i)) for i in np.argsort(np.abs(s))[:12]]
    for flips in (1, 2, 3):
        for combo in itertools.combinations(masks, flips):
            t = value
            for m in combo:
                t ^= m
            if _crc_ok(t):
                return t >> 32, True
    return value >> 32, False


                                                                             
                                
                                                                             

def _seed(key: bytes, *parts) -> int:
    message = ":".join(map(str, parts)).encode()
    return int.from_bytes(hmac.new(key, message, hashlib.sha256).digest()[:16], "big")


@lru_cache(maxsize=16)
def _layout(key: bytes, reveal_id: str, h: int, w: int) -> np.ndarray:
    ch, cw = -(-h // CELL), -(-w // CELL)
    rng = np.random.default_rng(_seed(key, reveal_id, "layout", ch, cw))
    out = (rng.permutation(ch * cw).reshape(ch, cw) % N_BITS).astype(np.int16)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=32)
def _chips(key: bytes, reveal_id: str, group: int, ch: int, cw: int) -> np.ndarray:
    rng = np.random.default_rng(_seed(key, reveal_id, "chips", group))
    out = rng.choice(np.array([-1, 1], np.int8), (ch, cw))                                         
    out.setflags(write=False)
    return out


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


def _tone_match(
    orig: np.ndarray,
    leak: np.ndarray,
    valid: np.ndarray | None,
    max_samples: int = 300_000,
) -> np.ndarray:
    """Gain/offset fit of leak -> orig on a sparse, odd-strided sample of valid pixels.
    The stride is odd so it never locks onto the 16 px carrier period."""
    if valid is None:
        n = int(orig.size)
        m = None
    else:
        m = valid > 0.35
        n = int(np.count_nonzero(m))
    if n < 500:
        return leak

    step = max(1, int((n / max_samples) ** 0.5)) | 1
    if m is None:
        x = orig[::step, ::step].ravel().astype(np.float64)
        y = leak[::step, ::step].ravel().astype(np.float64)
    else:
        mm = m[::step, ::step]
        x = orig[::step, ::step][mm].astype(np.float64)
        y = leak[::step, ::step][mm].astype(np.float64)
    del m

    lo_x, hi_x = np.percentile(x, [2, 98])
    lo_y, hi_y = np.percentile(y, [2, 98])
    keep = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    x, y = x[keep], y[keep]

    if x.size < 500 or x.var() < 1e-3:
        return leak

    a = float(((x - x.mean()) * (y - y.mean())).mean() / max(y.var(), 1e-6))
    b = float(x.mean() - a * y.mean())
    a = float(np.clip(a, 0.65, 1.6))
    out = leak.astype(np.float32)                                                           
    out *= a
    out += b
    return out


def _score(
    orig: np.ndarray,
    leak: np.ndarray,
    key: bytes,
    reveal_id: str,
    group: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Per-bit correlation scores. `valid=None` means "every pixel valid" and takes a
    cheaper path that never allocates a full-size mask."""
    h, w = orig.shape
    ch, cw = _grid(h, w)
    H, W = ch * CELL, cw * CELL

    leak = _tone_match(orig, leak, valid)
    p = _pad(np.subtract(leak, orig, dtype=np.float32), H, W)                                    
    b = p.reshape(ch, CELL, cw, CELL).transpose(0, 2, 1, 3)                

    if valid is None:
        vb = None
        b -= b.mean((2, 3), keepdims=True)
    else:
        vm = _pad(valid if valid.dtype == np.float32 else valid.astype(np.float32), H, W)
        vb = vm.reshape(ch, CELL, cw, CELL).transpose(0, 2, 1, 3)
        weight = vb.sum((2, 3))
        mean = np.einsum("abij,abij->ab", b, vb) / np.maximum(weight, 1e-5)
        b -= mean[:, :, None, None]

                                                                                       
                                                                                            
    smooth = cv2.GaussianBlur(
        b.reshape(ch * cw, CELL, CELL),
        (0, 0),
        1.15,
    ).reshape(ch, cw, CELL, CELL)
    smooth *= 0.30
    b -= smooth
    del smooth

    chips = _chips(key, reveal_id, group, ch, cw)
    if vb is None:
        corr = np.einsum("abij,ij->ab", b, _CARRIER)
        den = np.sqrt(_CARRIER_SQ_SUM * np.einsum("abij,abij->ab", b, b) + 1e-5)
        corr /= np.maximum(den, 1e-4)
        vals = corr * chips
    else:
        corr = np.einsum("abij,ij,abij->ab", b, _CARRIER, vb)
        den = np.sqrt(
            np.einsum("abij,ij->ab", vb, _CARRIER_SQ)
            * np.einsum("abij,abij,abij->ab", vb, b, b)
            + 1e-5
        )
        corr /= np.maximum(den, 1e-4)
        coverage = np.clip(vb.mean((2, 3)), 0, 1)
        vals = corr * chips * coverage
        vals[coverage < 0.25] = 0.0

    return np.bincount(
        _layout(key, reveal_id, h, w).ravel(),
        vals.ravel(),
        minlength=N_BITS,
    )


def _open_oriented(path: Path, mode: str, max_pixels: int) -> Image.Image:
    """Open, size-check from the header (before decoding), convert, then apply EXIF
    orientation. Rotating after conversion keeps the peak at one decoded copy."""
    with Image.open(path) as im:
        px = int(im.width) * int(im.height)
        if px > max_pixels:
            raise RuntimeError(
                f"Image is too large ({px / 1e6:.1f} MP; limit {max_pixels / 1e6:.1f} MP)."
            )
        orient = im.getexif().get(0x0112, 1)
        out = im.convert(mode)
    op = _ORIENT.get(orient)
    if op is not None:
        out = out.transpose(op)
    return out


def _load_rgb(path: Path, max_pixels: int = EMBED_IMAGE_MAX_PIXELS) -> Image.Image:
    return _open_oriented(path, "RGB", max_pixels)


                                                                             
                                                    
                                                                             

def _to_gray(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a


def _feature_image(
    rgb: np.ndarray,
    height: int = MATCH_H,
) -> tuple[np.ndarray, float]:
    """Returns a CLAHE'd gray image whose longest side is <= `height`, and the scale used."""
    g = _to_gray(rgb) if rgb.ndim == 3 else rgb.astype(np.uint8)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
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


def _corner_gap(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    """Max corner displacement (px) between two leak->orig maps; used to drop duplicates."""
    sh, sw = shape
    pts = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]]).reshape(-1, 1, 2)
    try:
        qa = cv2.perspectiveTransform(pts, a)
        qb = cv2.perspectiveTransform(pts, b)
    except cv2.error:
        return float("inf")
    return float(np.max(np.abs(qa - qb)))


def _estimate_scale(mat3: np.ndarray, src_shape: tuple[int, int]) -> float:
    """Approximate linear scale of the map leak -> orig (<1 means the leak is being shrunk)."""
    sh, sw = src_shape
    pts = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]]).reshape(-1, 1, 2)
    try:
        q = cv2.perspectiveTransform(pts, mat3).reshape(-1, 2)
        area = abs(float(cv2.contourArea(q.astype(np.float32).reshape(-1, 1, 2))))
    except cv2.error:
        return 1.0
    if not np.isfinite(area) or area <= 0:
        return 1.0
    return float(np.sqrt(area / max(sh * sw, 1)))


def _prefilter(leak: np.ndarray, scale: float) -> np.ndarray:
    """Low-pass before shrinking (e.g. retina screenshots) so INTER_LINEAR does not alias."""
    if scale >= 0.85 or scale <= 0:
        return leak
    sigma = min(4.0, 0.45 / scale)
    return cv2.GaussianBlur(leak, (0, 0), sigma)


def _warp_candidate(
    leak: np.ndarray,
    mat3: np.ndarray,
    orig_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = orig_shape
    src = _prefilter(leak, _estimate_scale(mat3, leak.shape[:2]))
    warped = cv2.warpPerspective(
        src, mat3, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    valid = cv2.warpPerspective(
        np.ones(leak.shape[:2], np.uint8), mat3, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    return warped, valid


def _clean_valid(valid: np.ndarray | None) -> np.ndarray | None:
    """Soften mask edges; return None when the mask is all ones (cheaper scoring path)."""
    if valid is None:
        return None
    if not np.any(valid < 0.99):
        return None
    valid = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 0.8)
    valid[valid < 0.12] = 0.0
    return valid


def _estimate_affine(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    partial: bool,
) -> Optional[tuple[np.ndarray, int]]:
    fn = cv2.estimateAffinePartial2D if partial else cv2.estimateAffine2D
    mat, mask = fn(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=4000, confidence=0.995, refineIters=20,
    )
    if mat is None or mask is None or int(mask.sum()) < 6:
        return None
    return np.vstack([mat, [0.0, 0.0, 1.0]]).astype(np.float64), int(mask.sum())


def _ecc_refine(
    orig_rgb: np.ndarray,
    warped_rgb: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel affine refinement. ECC runs on a reduced copy, so the estimated
    matrix is converted back to full-resolution coordinates before it is applied."""
    try:
        h, w = orig_rgb.shape[:2]
        scale = min(1.0, MATCH_H / max(h, w))
        og = _to_gray(orig_rgb)
        wg = _to_gray(warped_rgb)
        vs = valid.astype(np.float32)
        sx = sy = 1.0
        if scale < 1.0:
            size = (max(32, round(w * scale)), max(32, round(h * scale)))
            og = cv2.resize(og, size, interpolation=cv2.INTER_AREA)
            wg = cv2.resize(wg, size, interpolation=cv2.INTER_AREA)
            vs = cv2.resize(vs, size, interpolation=cv2.INTER_AREA)
            sx, sy = size[0] / w, size[1] / h

        template = cv2.GaussianBlur(og.astype(np.float32) / 255.0, (0, 0), 1.0)
        moving = cv2.GaussianBlur(wg.astype(np.float32) / 255.0, (0, 0), 1.0)
        mask = (vs > 0.9).astype(np.uint8)
        del og, wg, vs
        if int(mask.sum()) < 100:
            return warped_rgb, valid

        matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        cc, matrix = cv2.findTransformECC(
            template, moving, matrix, cv2.MOTION_AFFINE, criteria,
            inputMask=mask, gaussFiltSize=5,
        )
        if not np.isfinite(cc) or cc < ECC_MIN_CC:
            return warped_rgb, valid

                                                                             
        matrix[0, 1] *= sy / sx
        matrix[1, 0] *= sx / sy
        matrix[0, 2] /= sx
        matrix[1, 2] /= sy

        if (
            np.max(np.abs(matrix[:, :2] - np.eye(2, dtype=np.float32))) > 0.1
            or np.max(np.abs(matrix[:, 2])) > 0.05 * max(h, w)
        ):
            return warped_rgb, valid

        refined = cv2.warpAffine(
            warped_rgb, matrix, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        refined_valid = cv2.warpAffine(
            valid.astype(np.float32), matrix, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return refined, refined_valid
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return warped_rgb, valid


def _registration_matrices(
    orig: np.ndarray,
    leak: np.ndarray,
    *,
    max_candidates: int = TRACE_IMAGE_MAX_CANDIDATES,
) -> list[np.ndarray]:
    """Return a tiny set of promising leak->original registration matrices (full-res
    coordinates). Only compact 3x3 matrices are retained; callers materialise one
    full-resolution warp at a time."""
    cap = max(1, int(max_candidates))
    o_g, o_sc = _feature_image(orig)
    l_g, l_sc = _feature_image(leak)
    o_shape = orig.shape[:2]
    l_shape = leak.shape[:2]
    found: list[tuple[int, np.ndarray]] = []

    def to_full(hmat: np.ndarray) -> np.ndarray:
        """reduced-leak -> reduced-orig  ==>  full-leak -> full-orig"""
        return (
            np.diag([1.0 / max(o_sc, 1e-8), 1.0 / max(o_sc, 1e-8), 1.0])
            @ hmat
            @ np.diag([max(l_sc, 1e-8), max(l_sc, 1e-8), 1.0])
        ).astype(np.float64)

    def add_mat(full: np.ndarray, inliers: int) -> None:
        if len(found) >= cap:
            return
        if not _homography_is_reasonable(full, l_shape, o_shape):
            return
                                                                                   
        if any(_corner_gap(full, m, l_shape) < 2.0 for _, m in found):
            return
        found.append((int(inliers), full))

    try:
        sift = cv2.SIFT_create(
            nfeatures=TRACE_IMAGE_MAX_FEATURES,
            contrastThreshold=0.02,
            edgeThreshold=10,
        )
        okp, od = sift.detectAndCompute(o_g, None)
        lkp, ld = sift.detectAndCompute(l_g, None)
        if od is not None and ld is not None and len(okp) >= 8 and len(lkp) >= 8:
            matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ld, od, k=2)
            good = [
                m[0] for m in matches
                if len(m) == 2 and m[0].distance < 0.80 * m[1].distance
            ]
            if len(good) >= 6:
                src = np.float32([lkp[m.queryIdx].pt for m in good])                   
                dst = np.float32([okp[m.trainIdx].pt for m in good])                   

                                                                          
                try:
                    hmat, mask = cv2.findHomography(
                        src, dst, cv2.RANSAC, 3.0, maxIters=2500, confidence=0.99,
                    )
                    if hmat is not None and mask is not None:
                        inliers = int(mask.sum())
                        if inliers >= 6:
                            add_mat(to_full(hmat), inliers)
                except cv2.error:
                    pass

                                                                                 
                                                                                       
                if len(found) < cap:
                    try:
                        res = _estimate_affine(src, dst, True)
                        if res is not None:
                            add_mat(to_full(res[0]), res[1])
                    except cv2.error:
                        pass
    except cv2.error as exc:
        log.debug("SIFT registration failed: %s", exc)

    if not found:
        try:
            orb = cv2.ORB_create(nfeatures=1200, fastThreshold=8)
            okp, od = orb.detectAndCompute(o_g, None)
            lkp, ld = orb.detectAndCompute(l_g, None)
            if od is not None and ld is not None and len(okp) >= 8 and len(lkp) >= 8:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ld, od, k=2)
                good = [
                    m[0] for m in matches
                    if len(m) == 2 and m[0].distance < 0.78 * m[1].distance
                ]
                if len(good) >= 8:
                    src = np.float32([lkp[m.queryIdx].pt for m in good])
                    dst = np.float32([okp[m.trainIdx].pt for m in good])
                    hmat, mask = cv2.findHomography(
                        src, dst, cv2.RANSAC, 4.0, maxIters=1800,
                    )
                    if hmat is not None and mask is not None:
                        inliers = int(mask.sum())
                        if inliers >= 6:
                            add_mat(to_full(hmat), inliers)
        except cv2.error as exc:
            log.debug("ORB registration failed: %s", exc)

    found.sort(key=lambda item: item[0], reverse=True)
    return [mat for _, mat in found[:cap]]


def _aspect_close(a_shape: tuple[int, int], b_shape: tuple[int, int], tol: float = 0.01) -> bool:
    ra = a_shape[1] / max(a_shape[0], 1)
    rb = b_shape[1] / max(b_shape[0], 1)
    return abs(ra / max(rb, 1e-9) - 1.0) <= tol


def _iter_registration_candidates(
    orig: np.ndarray,
    leak: np.ndarray,
    *,
    refine_ecc: bool = True,
) -> Iterator[tuple[np.ndarray, Optional[np.ndarray]]]:
    """Lazily yield (aligned_leak, valid) pairs, one at a time, so only one full-size
    candidate is alive at once. `valid` is None when every pixel is valid.
    Every matrix gets its warped AND ECC-refined candidate (the old shared cap of 3
    meant a second matrix was rarely tried)."""
    o_shape = orig.shape[:2]
    if leak.shape[:2] == o_shape:
        yield leak, None

    for mat3 in _registration_matrices(orig, leak):
        warped, valid = _warp_candidate(leak, mat3, o_shape)
        refined = rvalid = None
        if refine_ecc:
            refined, rvalid = _ecc_refine(orig, warped, valid)
        yield warped, _clean_valid(valid)
        if refined is not None and refined is not warped:
            yield refined, _clean_valid(rvalid)
        del warped, valid, refined, rvalid

    if leak.shape[:2] != o_shape and _aspect_close(leak.shape[:2], o_shape):
        yield cv2.resize(leak, (o_shape[1], o_shape[0]), interpolation=cv2.INTER_AREA), None


def _image_trace_variants(h: int, w: int) -> list[tuple[float, float, float, float, float]]:
    """Small residual-registration search used after feature registration: a translation
    neighbourhood only (SIFT/ORB/ECC already recovered scale and rotation)."""
    radius = max(1, int(TRACE_IMAGE_TRANSLATION_RADIUS))
    offsets = range(-radius, radius + 1) if TRACE_IMAGE_DENSE_SHIFTS else (-radius, 0, radius)
    return [
        (0.0, 0.0, 0.0, dx / max(w, 1), dy / max(h, 1))
        for dy in offsets
        for dx in offsets
    ]


def _geometry_variants(
    h: int,
    w: int | None = None,
) -> list[tuple[float, float, float, float, float]]:
    """(scale_x, scale_y, rotation_deg, translate_x_frac, translate_y_frac).
    Translations are a full 2D +/-2 px grid; registration error is mostly translation."""
    w = int(w or h)
    out: list[tuple[float, float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if dx or dy:
                out.append((0.0, 0.0, 0.0, dx / max(w, 1), dy / max(h, 1)))
    s = min(0.006, 4.0 / max(h, w, 1))
    out += [
        (-s, -s, 0.0, 0.0, 0.0),
        (s, s, 0.0, 0.0, 0.0),
        (0.0, 0.0, -0.5, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.0, 0.0),
    ]
    return out


def _warp_y(
    y: np.ndarray,
    scale_x: float,
    scale_y: float,
    rotation: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> np.ndarray:
                                                                      
    if not (scale_x or scale_y or rotation or translate_x or translate_y):
        return y

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


def _top_scores(
    oy: np.ndarray,
    leak_y: np.ndarray,
    key: bytes,
    reveal_id: str,
    group: int,
    variants,
    valid: np.ndarray | None = None,
    keep: int = 3,
) -> list[tuple[float, np.ndarray]]:
    """Scores every geometry variant, then re-scores the best `keep` with a light blur.
    Returns (metric, scores) pairs sorted best first. Decoding is left to the caller so the
    expensive error-correction search only runs on promising candidates."""
    valid_f = valid.astype(np.float32, copy=False) if valid is not None else None
    plain: list[tuple[float, np.ndarray, tuple]] = []
    for v in variants:
        p = tuple(float(x) for x in v)
        warped = _warp_y(leak_y, *p)
        wv = _warp_y(valid_f, *p) if valid_f is not None else None
        s = _score(oy, warped, key, reveal_id, group, valid=wv)
        plain.append((float(np.mean(np.abs(s))), s, p))
    plain.sort(key=lambda t: -t[0])

    out = [(m, s) for m, s, _ in plain[:keep]]
    for _, _, p in plain[:keep]:
        warped = cv2.GaussianBlur(_warp_y(leak_y, *p), (3, 3), 0)
        wv = _warp_y(valid_f, *p) if valid_f is not None else None
        s = _score(oy, warped, key, reveal_id, group, valid=wv)
        out.append((float(np.mean(np.abs(s))), s))
    out.sort(key=lambda t: -t[0])
    return out


def _decode_registered(
    oy: np.ndarray,
    leak_y: np.ndarray,
    key: bytes,
    reveal_id: str,
    group: int,
    variants,
    valid: np.ndarray | None = None,
) -> tuple[int, float, bool]:
    best_uid, best_metric, best_ok = 0, float("-inf"), False
    for metric, scores in _top_scores(oy, leak_y, key, reveal_id, group, variants, valid, keep=4):
        uid, ok = decode_payload(scores)
        if ok and metric > best_metric:
            best_uid, best_metric, best_ok = int(uid), metric, True
    return best_uid, best_metric, best_ok


def _group_hypotheses(
    source_frame_idx: int,
    source_fps: float,
    delivery_fps: float | None = None,
) -> list[int]:
    """Return a small phase-tolerant set of watermark groups.

    The encoder's fps filter is timestamp based, so a source-frame -> delivery-frame
    conversion can land one frame either side of the ideal rounded index. The old
    implementation only covered +/- one group around a single rounded conversion.
    A group boundary is exactly where that assumption fails. Keep the search cheap,
    but cover floor/round/ceil plus +/-2 groups around each estimate.
    """
    source_fps = max(float(source_fps), 1.0)
    source_frame_idx = max(0, int(source_frame_idx))

    ordered: list[int] = []
    seen: set[int] = set()

    def add_group(center: int, radius: int = 2) -> None:
        center = max(0, int(center))
        for delta in range(-radius, radius + 1):
            value = center + delta
            if value < 0 or value in seen:
                continue
            seen.add(value)
            ordered.append(value)

                                                           
    add_group(source_frame_idx // VIDEO_GROUP, radius=1)

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
        ratio = source_frame_idx * fps / source_fps
        for delivered_frame_idx in {int(ratio // 1), int(round(ratio)), int(np.ceil(ratio))}:
            add_group(delivered_frame_idx // VIDEO_GROUP)

    return ordered


                                                                             
        
                                                                             

def embed_image_array(
    rgb: np.ndarray,
    user_id: int,
    key: bytes,
    reveal_id: str,
    amp: float = AMP,
    *,
    inplace: bool = False,
    band_rows: int = EMBED_BAND_ROWS,
) -> np.ndarray:
    """Embed the watermark band by band (cell-aligned, so the result equals the
    whole-image computation). With inplace=True the input array is overwritten and
    no second full-size buffer is allocated."""
    band_rows = max(CELL, (int(band_rows) // CELL) * CELL)
    h, w, _ = rgb.shape
    ch, cw = _grid(h, w)
    cell_bits = (
        encode_payload(user_id)[_layout(key, reveal_id, h, w)]
        * _chips(key, reveal_id, 0, ch, cw)
    )
    out = rgb if inplace else rgb.copy()

    for r0 in range(0, h, band_rows):
        r1 = min(h, r0 + band_rows)
        c0, c1 = r0 // CELL, -(-r1 // CELL)
        ycc = cv2.cvtColor(out[r0:r1], cv2.COLOR_RGB2YCrCb)
        y = ycc[:, :, 0].astype(np.float32)
        cells = cell_bits[c0:c1] * _texture(y)
        d = cells[:, :, None, None] * _CARRIER[None, None] * float(amp)
        d = d.transpose(0, 2, 1, 3).reshape((c1 - c0) * CELL, cw * CELL)[: r1 - r0, :w]
        ycc[:, :, 0] = np.clip(y + d, 0, 255).astype(np.uint8)
        out[r0:r1] = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    return out


def _save_image(img: Image.Image, dst: Path) -> None:
    suffix = dst.suffix.lower()

    if suffix in (".jpg", ".jpeg"):
        img.save(dst, "JPEG", quality=95, subsampling=0)
    elif suffix == ".webp":
        img.save(dst, "WEBP", lossless=True)
    else:
        img.save(dst, "PNG", compress_level=4)


def _watermarked_pil(src: Path, user_id: int, key: bytes, reveal_id: str, amp: float) -> Image.Image:
    rgb = np.array(_load_rgb(src), np.uint8)                              
    embed_image_array(rgb, user_id, key, reveal_id, amp, inplace=True)
    return Image.fromarray(rgb)


def embed_image(
    src: Path,
    dst: Path,
    user_id: int,
    key: bytes,
    reveal_id: str,
    amp: float = AMP,
    **_legacy,                                                                     
) -> None:
    _save_image(_watermarked_pil(src, user_id, key, reveal_id, amp), dst)


def _write_preview(img: Image.Image, preview: Path, max_dim: int, quality: int) -> None:
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)             
    preview.parent.mkdir(parents=True, exist_ok=True)
    img.save(
        preview,
        "JPEG",
        quality=max(70, min(95, int(quality))),
        optimize=True,
        progressive=False,
        subsampling=2,
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
    _write_preview(_watermarked_pil(src, user_id, key, reveal_id, amp), preview, max_dim, quality)


def _write_delivery(img: Image.Image, dst: Path, target_bytes: Optional[int]) -> dict:
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
        f"target {int(target_bytes or 0) / 1048576:.1f} MiB)."
    )


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
    img = _watermarked_pil(src, user_id, key, reveal_id, amp)
    return _write_delivery(img, dst, target_bytes)


def render_image_both(
    src: Path,
    preview: Path,
    delivery: Path,
    user_id: int,
    key: bytes,
    reveal_id: str,
    *,
    amp: float = AMP,
    max_dim: int = 2048,
    quality: int = 88,
    target_bytes: Optional[int] = None,
) -> dict:
    """Embed ONCE and write both outputs (use this instead of calling the preview and
    delivery renderers back to back). Returns the delivery info dict."""
    img = _watermarked_pil(src, user_id, key, reveal_id, amp)
    info = _write_delivery(img, delivery, target_bytes)
    _write_preview(img, preview, max_dim, quality)                                        
    return info


def _load_luma(path: Path) -> np.ndarray:
    """Load one image as compact uint8 luma. The size guard runs on the header, before decode."""
    return np.array(_open_oriented(path, "L", TRACE_IMAGE_MAX_PIXELS), dtype=np.uint8)


def extract_image(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    time_budget: float = 30.0,
) -> dict:
    """Trace an edited/cropped image with bounded peak memory.

    Order: (1) cheap identity / pure-resize decode, (2) feature registration (SIFT/ORB) only
    if that fails, then one registered warp at a time. Only luma is kept at full resolution."""
    deadline = time.monotonic() + max(1.0, float(time_budget))
    oy = _load_luma(orig)
    leak_y = _load_luma(leak)
    h, w = oy.shape
    variants = _image_trace_variants(h, w)

    def ok_result(uid, metric):
        return {"user_id": int(uid), "metric": float(metric), "ok": True, "valid": True}

                                                                                       
    base = None
    if leak_y.shape == oy.shape:
        base = leak_y
    elif _aspect_close(leak_y.shape, oy.shape):
        base = cv2.resize(leak_y, (w, h), interpolation=cv2.INTER_AREA)
    if base is not None:
        uid, metric, ok = _decode_registered(oy, base, key, reveal_id, 0, variants, valid=None)
        if ok:
            return ok_result(uid, metric)
        del base

    if time.monotonic() > deadline:
        return _fail()

                                                                   
    matrices = _registration_matrices(oy, leak_y)
    log.info("image extract: %d compact registration matrices", len(matrices))

    for mat3 in matrices:
        if time.monotonic() > deadline:
            break
        warped, valid = _warp_candidate(leak_y, mat3, (h, w))
        if TRACE_IMAGE_ECC:
            warped, valid = _ecc_refine(oy, warped, valid)
        valid = _clean_valid(valid)

        if time.monotonic() > deadline:
            break
        uid, metric, ok = _decode_registered(oy, warped, key, reveal_id, 0, variants, valid=valid)
        del warped, valid
        if ok:
            return ok_result(uid, metric)

    return _fail()


                                                                             
               
                                                                             

def _scale_filter(height: int, flags: str) -> str:
    chosen = os.getenv("VIDEO_SCALE_FILTER", "bicubic").strip().lower() or "bicubic"
    if chosen not in {"fast_bilinear", "bilinear", "bicubic", "lanczos", "spline", "area"}:
        chosen = flags
    return rf"scale=-2:trunc(min(ih\,{int(height)})/2)*2:flags={chosen},setsar=1"


def _file_sig(path: Path) -> tuple[str, int, int]:
    p = Path(path)
    st = p.stat()
    return str(p), st.st_mtime_ns, st.st_size


@lru_cache(maxsize=16)
def _ffmpeg_info_cached(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    return subprocess.run(
        [get_ffmpeg(), "-hide_banner", *_in(path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    ).stderr


def _ffmpeg_info(src: Path) -> str:
    return _ffmpeg_info_cached(*_file_sig(src))


@lru_cache(maxsize=16)
def _probe_video_cached(path: str, mtime_ns: int, size: int, max_height: int):
    info = _ffmpeg_info_cached(path, mtime_ns, size)

    fps = None
    for pattern in (r"(\d+(?:\.\d+)?)\s*fps", r"(\d+(?:\.\d+)?)\s*tbr"):
        match = re.search(pattern, info)
        if match and float(match.group(1)) > 0:
            fps = float(match.group(1))
            break

    fps = min(max(fps or 30.0, 1.0), VIDEO_MAX_FPS)

    max_height = max(144, int(max_height))
    vf = f"fps={fps:.6f}," + _scale_filter(max_height, "lanczos")

                                                                                        
    probe = subprocess.run(
        [*_ff(), *_in(path), "-vf", vf, "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "ppm", "-"],
        capture_output=True,
        timeout=45,
    )

    if not probe.stdout:
        details = probe.stderr.decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"FFmpeg could not decode the video. {details}".strip())

    with Image.open(io.BytesIO(probe.stdout)) as im:
        w, h = im.size

    return w, h, fps, vf


def _probe_video(src: Path, max_height: int):
    return _probe_video_cached(*_file_sig(src), int(max_height))


def video_info(src: Path, max_height: int = 1080) -> dict:
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
    max_seconds: Optional[int] = None,
) -> Iterator[tuple[int, bytearray]]:
    """Yield (index, yuv420p buffer). The SAME bytearray is reused for every frame."""
    size = w * h * 3 // 2

    cmd = _ff()
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += [*_in(src), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=size * 2,
    )

    buf = bytearray(size)
    idx = 0

    try:
        while _read_full(proc.stdout, buf):
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


_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")


def _choose_video_preset(preset: str, duration: float) -> str:
    requested = (preset or "").strip().lower()
    if requested in _PRESETS:
        chosen = requested
    elif requested not in {"", "auto"}:
        chosen = "fast"
    elif duration >= 180:
        chosen = "slow"
    elif duration >= 90:
        chosen = "medium"
    else:
        chosen = "fast"

                                                                        
    cap = os.getenv("VIDEO_PRESET_MAX", "").strip().lower()
    if cap in _PRESETS and _PRESETS.index(chosen) > _PRESETS.index(cap):
        chosen = cap
    return chosen


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

                                                                              
                                                                             
                                                                                
                                                                              
                                                    
    long_seconds = max(0.0, float(os.getenv("VIDEO_LONG_PROFILE_SECONDS", "90")))
    if long_seconds and duration >= long_seconds:
        long_height = max(360, int(os.getenv("VIDEO_LONG_PROFILE_HEIGHT", "540")))
        long_fps = max(12.0, min(24.0, float(os.getenv("VIDEO_LONG_PROFILE_FPS", "20"))))
        ceiling = min(ceiling, long_height)
        source_fps = min(source_fps, long_fps)

    if not target_bytes or duration <= 0 or ceiling <= 360:
        return ceiling, source_fps

    usable_bits = max(
        1.0,
        float(target_bytes) * 8.0 * 0.955 - float(audio_bps) * float(duration),
    )
    average_video_bps = usable_bits / max(float(duration), 1.0)
    bpp_floor = 0.022

    fps_candidates = [source_fps]
    for f in VIDEO_FPS_HYPOTHESES:
        f = min(source_fps, f)
        if f >= 1.0:
            fps_candidates.append(round(f, 3))
    fps_candidates = sorted(set(fps_candidates), reverse=True)

    for candidate_h in HEIGHT_LADDER:
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
    preset: str = "fast",
    max_seconds: int = 300,
    max_height: int = 1080,
    group: int = VIDEO_GROUP,
    target_bytes: Optional[int] = None,
    audio_kbps: int = 96,
    **_legacy,                                                                                    
) -> dict:
    """Returns the delivery profile. STORE `height` and `fps` (and width) per reveal and pass
    them back to `extract(..., delivery_height=..., delivery_fps=...)`: the watermark layout is
    derived from the delivered frame size, so tracing needs the reference rendered at that size."""
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

    base_cap: Optional[int] = None
    if USE_VBV_CAP and target:
        bits_left = target * 8.0 * 0.95 - audio_bps * duration
        base_cap = max(MIN_VIDEO_BPS, int(bits_left / max(duration, 1.0)))

    def encode_once(
        out_path: Path,
        w: int,
        h: int,
        fps: float,
        crf: Optional[float] = None,
        video_bitrate: Optional[int] = None,
        maxrate: Optional[int] = None,
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
            if maxrate:
                rate_args += ["-maxrate", str(int(maxrate)), "-bufsize", str(int(maxrate) * 2)]
        gop = max(1, min(120, int(round(fps * 2.0))))
        err = tempfile.TemporaryFile()
        enc = subprocess.Popen(
            [
                *_ff(), "-y",
                "-threads", str(FFMPEG_THREADS), "-protocol_whitelist", "pipe,fd",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{w}x{h}", "-framerate", f"{fps:.6f}", "-i", "-",
                "-t", f"{duration:.3f}", *_in(src),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", selected_preset,
                *rate_args,
                "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
                "-g", str(gop),
                "-x264-params", "aq-mode=1:aq-strength=0.9:open-gop=0",
                "-tag:v", "avc1",
                "-c:a", "aac", "-b:a", str(audio_bps), "-ar", "48000", "-ac", "2",
                "-threads", str(FFMPEG_THREADS),
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
        rc = -1
        try:
            with closing(_iter_frames(src, vf, w, h, max_seconds=max_seconds)) as it:
                for i, buf in it:
                    source_frame_idx = int(round(i * source_fps / max(float(fps), 1e-6)))
                    group_idx = source_frame_idx // max(1, int(group))

                    y = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w)

                    if group_idx != cached_group:
                        add_plane, sub_plane = _delta_planes(
                            y, bits, key, reveal_id, group_idx, amp
                        )
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
            rc = enc.wait(timeout=max(120, int(duration * 8)))
        except subprocess.TimeoutExpired as exc:
            enc.kill()
            enc.wait()
            raise RuntimeError("FFmpeg video encode timed out.") from exc
        finally:
            if enc.poll() is None:
                enc.kill()
                enc.wait()
            err.seek(0)
            error_text = err.read().decode("utf-8", "replace")[-4000:]
            err.close()

        if rc != 0 or frames == 0 or not out_path.exists() or out_path.stat().st_size <= 0:
            raise RuntimeError(
                "FFmpeg encode failed" + (f": {error_text}" if error_text else ".")
            )
        return frames

    height_candidates = [delivery_height]
    for candidate in HEIGHT_LADDER:
        if candidate < delivery_height and candidate <= source_h and candidate <= max_height:
            height_candidates.append(candidate)
    height_candidates = list(dict.fromkeys(height_candidates))[:4]

    for profile_no, height in enumerate(height_candidates, 1):
        w, h, probed_fps, _ = _probe_video(src, int(height))
        base_fps = min(float(probed_fps), float(delivery_fps), 30.0)
        if profile_no > 1:
            base_fps = min(base_fps, 24.0 if height <= 720 else 27.0)

        fps_candidates = []
        for f in (base_fps, 27.0, 24.0, 20.0, 18.0, 15.0):
            f = min(base_fps, f)
            if f >= 1.0:
                fps_candidates.append(round(float(f), 3))
        fps_candidates = list(dict.fromkeys(fps_candidates))

        for fps in fps_candidates:
            crf = _quality_crf(
                width=w, height=h, fps=fps, target=target, duration=duration
            )

            for attempt in range(3):
                tmp_out = dst.with_name(
                    f"{dst.stem}.p{profile_no}_{int(round(fps))}_{attempt}.tmp{dst.suffix}"
                )
                cap = int(base_cap * (0.93 ** attempt)) if base_cap else None
                try:
                    encode_once(tmp_out, w, h, fps, crf=crf, maxrate=cap)
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
            tmp_out = dst.with_name(f"{dst.stem}.rescue{rescue_no}.tmp{dst.suffix}")
            try:
                encode_once(tmp_out, w, h, fps, video_bitrate=rescue_rate)
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
    limit_text = f", target {target / 1048576:.1f} MiB" if target else ""
    raise RuntimeError(f"Video could not be encoded below the delivery limit ({profile_text}{limit_text}).")


                                                                             
               
                                                                             

def _visual_signature(gray: np.ndarray, width: int = 48, height: int = 27) -> np.ndarray:
    small = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    norm = np.linalg.norm(small)
    if norm > 1e-6:
        small /= norm
    return small


def _sift_descriptors(gray: np.ndarray, sift, max_dim: int = COARSE_FEATURE_DIM):
    small, _ = _feature_image(gray, max_dim)
    return sift.detectAndCompute(small, None)


def _sift_match_count(matcher, query_desc: np.ndarray | None, frame_desc: np.ndarray | None) -> float:
    if query_desc is None or frame_desc is None or len(query_desc) < 5 or len(frame_desc) < 8:
        return 0.0
    try:
        matches = matcher.knnMatch(query_desc, frame_desc, k=2)
    except cv2.error:
        return 0.0
    return float(sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.78 * m[1].distance))


def _height_ladder(orig: Path, max_height: int) -> list[int]:
    """Candidate delivery heights to try when the stored profile is unknown. Probes once
    at the requested ceiling; the real height is validated when a window is decoded."""
    ceiling = max(144, int(max_height))
    try:
        _, actual_h, _, _ = _probe_video(orig, ceiling)
        actual_h = max(144, int(actual_h))
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        actual_h = ceiling

    if actual_h < ceiling:
        return [actual_h]

    out: list[int] = []
    seen: set[int] = set()
                                                                                     
                                                                                        
    for c in (ceiling, 720, 540, 480):
        c = int(c)
        if c > ceiling or c < 144:
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out or [ceiling]


def _coarse_rank(
    orig: Path,
    queries: list[np.ndarray],
    max_height: int,
    *,
    max_seconds: int = 300,
    top_k: int = 5,
    deadline: float | None = None,
) -> list[list[float]]:
    """One pass over the source video. For every query (RGB leak image) returns candidate
    timestamps ranked by SIFT matches, which survives crops, letterboxing and player UI far
    better than a whole-frame thumbnail comparison."""
    if not queries:
        return []
    coarse_h = max(144, min(COARSE_H, int(max_height)))
    w, h, source_fps, _ = _probe_video(orig, coarse_h)
    coarse_fps = min(COARSE_FPS, max(1.0, float(source_fps)))
    vf = f"fps={coarse_fps:.6f}," + _scale_filter(coarse_h, "fast_bilinear")

    sift_q = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.04, edgeThreshold=10)
    sift_f = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.04, edgeThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    q_gray = [_to_gray(q) for q in queries]
    q_sig = [_visual_signature(g) for g in q_gray]
    q_desc = [_sift_descriptors(g, sift_q)[1] for g in q_gray]
    del q_gray
    ranked: list[list[tuple[float, int]]] = [[] for _ in queries]

    with closing(_iter_frames(orig, vf, w, h, max_seconds=max_seconds)) as frames:
        for idx, buf in frames:
            if deadline is not None and time.monotonic() > deadline:
                log.info("coarse search stopped by time budget at frame %d", idx)
                break
            gray = np.frombuffer(buf, np.uint8, count=w * h).reshape(h, w)
            sig = _visual_signature(gray)
            desc = _sift_descriptors(gray, sift_f)[1]
            for n in range(len(queries)):
                sift = _sift_match_count(matcher, q_desc[n], desc)
                diff = float(np.mean(np.abs(sig - q_sig[n])))
                ranked[n].append((-sift + 100.0 * diff, idx))

    out: list[list[float]] = []
    for scored in ranked:
        scored.sort(key=lambda x: x[0])
        times: list[float] = []
        for _, idx in scored:
            t = idx / coarse_fps
            if all(abs(t - old) >= 0.75 for old in times):
                times.append(float(t))
            if len(times) >= top_k:
                break
        out.append(times)
    return out


def _ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    b = cv2.GaussianBlur(b, (0, 0), 1.0)
    mm = m > 0.9
    if int(mm.sum()) < 200:
        return -1.0
    x = a[mm] - a[mm].mean()
    y = b[mm] - b[mm].mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-6))


def _iter_window(
    orig: Path,
    t0: float,
    span: float,
    w: int,
    h: int,
    fps: float,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream the RGB frames of [t0, t0+span] at the delivery size. The yielded array is a
    view of ONE reused buffer: copy anything you want to keep."""
    vf = f"fps={fps:.6f}," + _scale_filter(h, "lanczos")
    cmd = [
        *_ff(), "-ss", f"{t0:.3f}", *_in(orig), "-t", f"{span:.3f}",
        "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 2,
    )
    buf = bytearray(w * h * 3)
    t_end = time.monotonic() + WINDOW_READ_TIMEOUT
    k = 0
    try:
        while _read_full(proc.stdout, buf):
            yield k, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
            if time.monotonic() > t_end:
                log.warning("window read timed out at t0=%.2f", t0)
                break
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def _scan_window(orig: Path, center: float, height: int, half: float):
    """Pass 1: stream the window keeping only small gray thumbnails (for NCC ranking) and the
    middle RGB frame (for registration). Memory is a few MB instead of ~190 MB at 1080p.
    Returns (w, h, fps, [(frame_idx, thumb_uint8)], mid_rgb, t0)."""
    w, h, fps, _ = _probe_video(orig, height)
    t0 = max(0.0, float(center) - half)
    span = 2.0 * half
    base = int(round(t0 * fps))
    mid_k = int(span * fps / 2.0)
    sh = max(64, min(h, 270))
    sw = max(64, int(round(w * sh / h)))

    thumbs: list[tuple[int, np.ndarray]] = []
    mid_rgb: Optional[np.ndarray] = None
    with closing(_iter_window(orig, t0, span, w, h, fps)) as frames:
        for k, rgb in frames:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            thumbs.append((base + k, cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)))
            if k == mid_k:
                mid_rgb = rgb.copy()

    if thumbs and mid_rgb is None:                                                        
        want = base + len(thumbs) // 2
        mid_rgb = _collect_window(orig, t0, span, w, h, fps, {want}, rgb=True).get(want)
    return w, h, fps, thumbs, mid_rgb, t0


def _collect_window(
    orig: Path,
    t0: float,
    span: float,
    w: int,
    h: int,
    fps: float,
    want: set[int],
    *,
    rgb: bool = False,
) -> dict[int, np.ndarray]:
    """Pass 2: decode the window again but keep only the requested frame indices
    (full-res uint8 luma, or RGB). Stops as soon as all of them are collected."""
    base = int(round(t0 * fps))
    last = max(want)
    out: dict[int, np.ndarray] = {}
    with closing(_iter_window(orig, t0, span, w, h, fps)) as frames:
        for k, frame in frames:
            idx = base + k
            if idx in want:
                out[idx] = frame.copy() if rgb else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if len(out) == len(want):
                    break
            elif idx > last:
                break
    return out


def _decode_window(
    orig: Path,
    leak_rgb: np.ndarray,
    center: float,
    key: bytes,
    reveal_id: str,
    height: int,
    leak_fps: float | None = None,
    half: float = WINDOW_HALF_SEC,
    top: int = WINDOW_TOP_FRAMES,
) -> Optional[dict]:
    """Register the leak against the source around `center`, find which source frame(s) it
    came from by NCC, then decode against those frames. Scores from several frames are
    accumulated if no single frame decodes on its own."""
    try:
        w, h, fps, thumbs, mid_rgb, t0 = _scan_window(orig, center, height, half)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        log.warning("window read failed at t=%.2f h=%d: %s", center, height, exc)
        return None
    if not thumbs or mid_rgb is None:
        log.info("no frames at t=%.2f h=%d", center, height)
        return None

    sh, sw = thumbs[0][1].shape

                                                                                      
                                                                           
    best = None
    for cand, valid in _iter_registration_candidates(mid_rgb, leak_rgb):
        if cand.shape[:2] != (h, w):
            continue
        cy = cv2.cvtColor(cand.astype(np.float32), cv2.COLOR_RGB2GRAY)
        del cand
        cy_s = cv2.resize(cy, (sw, sh), interpolation=cv2.INTER_AREA)
        if valid is None:
            v_s = np.ones((sh, sw), np.float32)
        else:
            v_s = cv2.resize(valid.astype(np.float32, copy=False), (sw, sh), interpolation=cv2.INTER_AREA)
        scored = sorted(
            ((_ncc(f.astype(np.float32), cy_s, v_s), idx) for idx, f in thumbs),
            key=lambda t: -t[0],
        )
        if best is None or scored[0][0] > best[0]:
            best = (scored[0][0], scored[:top], cy, valid)
    del mid_rgb, thumbs

    if best is None:
        log.info("no registration candidate at t=%.2f h=%d", center, height)
        return None
    top_ncc, ranked, cy, valid = best
    if top_ncc < 0.2:
        log.info("low NCC %.3f at t=%.2f h=%d", top_ncc, center, height)
        return None

                                                              
    luma = _collect_window(orig, t0, 2.0 * half, w, h, fps, {idx for _, idx in ranked})

    variants = _geometry_variants(h, w)
    zero = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    total = np.zeros(N_BITS, np.float64)
    accumulated = 0

    for _, idx in ranked:
        if idx not in luma:
            continue
        oy = luma[idx].astype(np.float32)
        groups = _group_hypotheses(idx, fps, leak_fps)
                                                                                          
        group_rank = sorted(
            (
                (_top_scores(oy, cy, key, reveal_id, g, zero, valid, keep=1)[0][0], g)
                for g in groups
            ),
            reverse=True,
        )[:4]

        frame_best = None
        for _, g in group_rank:
            for metric, scores in _top_scores(oy, cy, key, reveal_id, g, variants, valid, keep=3):
                uid, ok = decode_payload(scores)
                if ok:
                    return {
                        "user_id": int(uid), "metric": float(metric),
                        "frame": int(idx), "time": float(idx / fps), "ncc": float(top_ncc),
                    }
                if frame_best is None or metric > frame_best[0]:
                    frame_best = (metric, scores)

        if frame_best is not None:
            s = np.asarray(frame_best[1], np.float64)
            total += s / (np.median(np.abs(s)) + 1e-6)
            accumulated += 1

    if accumulated >= 2:
        uid, ok = decode_payload(total)
        if ok:
            idx0 = ranked[0][1]
            return {
                "user_id": int(uid), "metric": float(np.mean(np.abs(total))),
                "frame": int(idx0), "time": float(idx0 / fps), "ncc": float(top_ncc),
            }

    log.info("CRC failed at t=%.2f h=%d (ncc=%.3f, frames=%d)", center, height, top_ncc, accumulated)
    return None


def _try_times(
    orig: Path,
    leak_rgb: np.ndarray,
    times: list[float],
    heights: list[int],
    key: bytes,
    reveal_id: str,
    group_fps: float | None,
    deadline: float,
    tried: list[float],
) -> Optional[dict]:
                                                                                     
                                                                                 
    for base_t in times:
        for delta in (0.0, -0.25, 0.25, -0.50, 0.50):
            t = max(0.0, float(base_t) + delta)
            if any(abs(t - x) < 0.20 for x in tried):
                continue
            tried.append(t)
            for height in heights:
                if time.monotonic() > deadline:
                    log.info("time budget exhausted")
                    return None
                hit = _decode_window(orig, leak_rgb, t, key, reveal_id, height, group_fps)
                if hit is not None:
                    return hit
    return None


def _scan_fallback(
    orig: Path,
    leak_rgb: np.ndarray,
    heights: list[int],
    key: bytes,
    reveal_id: str,
    group_fps: float | None,
    deadline: float,
    tried: list[float],
) -> Optional[dict]:
    """Last resort: walk the source in back-to-back windows until the budget runs out."""
    duration = min(300.0, _video_duration_seconds(orig))
    step = 2.0 * WINDOW_HALF_SEC
    t = WINDOW_HALF_SEC
    while t <= duration + 1e-6:
        if time.monotonic() > deadline:
            break
        hit = _try_times(orig, leak_rgb, [t], heights, key, reveal_id, group_fps, deadline, tried)
        if hit is not None:
            return hit
        t += step
    return None


def _sample_leak_video(
    leak: Path,
    max_height: int,
    count: int = 5,
) -> tuple[list[tuple[int, np.ndarray]], float]:
    """Samples evenly spaced RGB frames using random-access seeks, so long leaked clips
    are not decoded from frame zero."""
    w, h, fps, _ = _probe_video(leak, max_height)
    duration = _video_duration_seconds(leak)
    count = max(1, min(int(count), TRACE_MAX_SAMPLES))
    times = [0.0] if duration <= 0 else [
        duration * (i + 0.5) / count for i in range(count)
    ]

    vf = _scale_filter(h, "fast_bilinear")
    frame_size = w * h * 3
    out: list[tuple[int, np.ndarray]] = []

    for t in times:
        cmd = [
            *_ff(),
            "-ss", f"{max(0.0, float(t)):.3f}",
            *_in(leak),
            "-an", "-frames:v", "1",
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=min(12.0, max(3.0, WINDOW_READ_TIMEOUT / 4.0)),
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0 or len(proc.stdout) < frame_size:
            continue

        rgb = np.frombuffer(proc.stdout, np.uint8, count=frame_size).reshape(h, w, 3).copy()
        out.append((int(round(float(t) * max(fps, 1.0))), rgb))

    return out, fps


def extract_video_frame(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    *,
    max_height: int = 1080,
    start_frame: int = 0,
    heights: Optional[list[int]] = None,
    delivery_fps: Optional[float] = None,
    time_budget: float = DEFAULT_TIME_BUDGET,
) -> dict:
    """Trace a single screenshot/crop taken from a delivered video."""
    deadline = time.monotonic() + float(time_budget)
    leak_rgb = np.asarray(_load_rgb(leak, TRACE_IMAGE_MAX_PIXELS), np.uint8)
    heights = list(heights) if heights else _height_ladder(orig, max_height)
    source_fps = _probe_video(orig, max_height)[2]

    tried: list[float] = []
    if start_frame > 0:
        times = [float(start_frame) / max(source_fps, 1.0)]
    else:
        times = _coarse_rank(orig, [leak_rgb], max_height, deadline=deadline)[0]
        if not times:
            log.info("coarse search found no candidate times")
            times = [0.0]

    hit = _try_times(orig, leak_rgb, times, heights, key, reveal_id, delivery_fps, deadline, tried)
    if hit is None and start_frame == 0:
        hit = _scan_fallback(orig, leak_rgb, heights, key, reveal_id, delivery_fps, deadline, tried)
    if hit is None:
        return _fail(frame=0, time=0.0)
    return {
        "user_id": int(hit["user_id"]), "metric": float(hit["metric"]),
        "frame": int(hit["frame"]), "time": float(hit["time"]),
        "ok": True, "valid": True,
    }


def extract_video(
    orig: Path,
    leak: Path,
    key: bytes,
    reveal_id: str,
    *,
    max_height: int = 1080,
    start_frame: int = 0,
    heights: Optional[list[int]] = None,
    delivery_fps: Optional[float] = None,
    time_budget: float = DEFAULT_TIME_BUDGET,
) -> dict:
    """Trace a leaked video clip (re-encoded, cropped, screen-recorded...)."""
    deadline = time.monotonic() + float(time_budget)
    heights = list(heights) if heights else _height_ladder(orig, max_height)

    try:
        samples, leak_fps = _sample_leak_video(leak, max_height, count=TRACE_MAX_SAMPLES)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        log.warning("could not sample leak video: %s", exc)
        return _fail(frame=0, time=0.0)
    if not samples:
        return _fail(frame=0, time=0.0)

    source_fps = _probe_video(orig, max_height)[2]
    group_fps = delivery_fps or leak_fps
    picks = samples[:: max(1, len(samples) // 4)][:4]
    tried: list[float] = []
    hit: Optional[dict] = None

    if start_frame > 0:
        hit = _try_times(
            orig, picks[0][1], [float(start_frame) / max(source_fps, 1.0)],
            heights, key, reveal_id, group_fps, deadline, tried,
        )
    else:
        source_duration = min(300.0, _video_duration_seconds(orig))
        leak_duration = _video_duration_seconds(leak)
        ratio = source_duration / leak_duration if leak_duration > 0 and source_duration > 0 else 1.0

                                                                     
        for leak_idx, rgb in picks:
            t = max(0.0, float(leak_idx) / max(leak_fps, 1.0) * ratio)
            hit = _try_times(orig, rgb, [t], heights, key, reveal_id, group_fps, deadline, tried)
            if hit is not None:
                break

                                                                          
        if hit is None:
            chosen = picks[:3]
            ranked_times = _coarse_rank(orig, [rgb for _, rgb in chosen], max_height, deadline=deadline)
            for (_, rgb), times in zip(chosen, ranked_times):
                hit = _try_times(orig, rgb, times, heights, key, reveal_id, group_fps, deadline, tried)
                if hit is not None:
                    break

                              
        if hit is None:
            hit = _scan_fallback(orig, picks[len(picks) // 2][1], heights, key, reveal_id, group_fps, deadline, tried)

    if hit is None:
        return _fail(frame=0, time=0.0)
    return {
        "user_id": int(hit["user_id"]), "metric": float(hit["metric"]),
        "frame": int(hit["frame"]), "time": float(hit["time"]),
        "ok": True, "valid": True,
    }


def extract(
    orig: Path,
    leak: Path,
    kind: str,
    key: bytes,
    reveal_id: str,
    *,
    max_height: int = 1080,
    start_frame: int = 0,
    delivery_height: Optional[int] = None,
    delivery_fps: Optional[float] = None,
    time_budget: float = DEFAULT_TIME_BUDGET,
    **_legacy,                                                                                
) -> dict:
    """`delivery_height` / `delivery_fps` are the `height` / `fps` returned by embed_video for
    this reveal. Pass them whenever you have them: the watermark layout depends on the
    delivered frame size. Without them every height in HEIGHT_LADDER is tried (slow).

    A positive result only says "the payload CRC matched". Before naming anyone, confirm that
    the returned user_id actually received this reveal_id in your database."""
    heights: Optional[list[int]] = None
    if kind in ("video", "video_frame") and delivery_height:
        max_height = int(delivery_height)
        heights = [max_height]

    if kind == "image":
        res = extract_image(orig, leak, key, reveal_id, time_budget=time_budget)
    elif kind == "video_frame":
        res = extract_video_frame(
            orig,
            leak,
            key,
            reveal_id,
            max_height=max_height,
            start_frame=start_frame,
            heights=heights,
            delivery_fps=delivery_fps,
            time_budget=time_budget,
        )
    elif kind == "video":
        res = extract_video(
            orig,
            leak,
            key,
            reveal_id,
            max_height=max_height,
            start_frame=start_frame,
            heights=heights,
            delivery_fps=delivery_fps,
            time_budget=time_budget,
        )
    else:
        raise ValueError(f"Unsupported extraction kind: {kind}")
    res["valid"] = bool(res.get("valid", res.get("ok", False)))
    return res
