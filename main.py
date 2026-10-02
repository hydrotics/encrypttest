import asyncio
import ipaddress
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv
from PIL import Image

import watermark as wm
from health_server import start_health_server

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
SECRET = os.getenv("WATERMARK_SECRET")
if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing.")
if not SECRET or len(SECRET) < 16:
    raise RuntimeError("WATERMARK_SECRET is missing/too short.")
WM_KEY = SECRET.encode("utf-8")

GUILD_ID = os.getenv("GUILD_ID")
BOOSTER_ROLE_ID = int(os.getenv("BOOSTER_ROLE_ID", "0") or 0)
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "300"))
MAX_VIDEO_HEIGHT = int(os.getenv("MAX_VIDEO_HEIGHT", "1080"))
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "auto").strip().lower()
VIDEO_AUDIO_KBPS = int(os.getenv("VIDEO_AUDIO_KBPS", "96"))
# Keep delivered videos modest: large files are what stall on mobile data.
VIDEO_TARGET_MAX_MB = float(os.getenv("VIDEO_TARGET_MAX_MB", "95"))
DEFAULT_UPLOAD_LIMIT = 20 * 1048576  # Current Discord API baseline when no limit is reported.
VIDEO_MAX_BPP = float(os.getenv("VIDEO_MAX_BPP", str(wm.MAX_BITS_PER_PIXEL)))
# Bump this when the encoded-cache naming/target semantics change.
VIDEO_CACHE_VERSION = f"{os.getenv('VIDEO_CACHE_VERSION', 'v21')}-quality-first-crf-trace-v2"
IMAGE_AMP = float(os.getenv("WM_IMAGE_AMP", "3.0"))
IMAGE_CELL = int(os.getenv("WM_IMAGE_CELL", "4"))
VIDEO_AMP = float(os.getenv("WM_VIDEO_AMP", "3.0"))
VIDEO_CELL = int(os.getenv("WM_VIDEO_CELL", "8"))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "250")) * 1048576)
MAX_IMAGE_PIXELS = int(float(os.getenv("MAX_IMAGE_MEGAPIXELS", "100")) * 1_000_000)
UPLOAD_WAIT_SECONDS = int(os.getenv("UPLOAD_WAIT_SECONDS", "90"))
MAX_URL_BYTES = min(MAX_UPLOAD_BYTES, int(float(os.getenv("MAX_MEDIA_URL_MB", "250")) * 1048576))
# Trace is intentionally bounded so one badly edited leak cannot monopolize the
# Render instance. The budget is for the whole /trace command; per-kind budgets
# are ceilings for an individual extractor call.
TRACE_TIME_BUDGET = max(10.0, float(os.getenv("TRACE_TIME_BUDGET", "45")))
TRACE_IMAGE_TIME_BUDGET = max(3.0, float(os.getenv("TRACE_IMAGE_TIME_BUDGET", "15")))
TRACE_VIDEO_TIME_BUDGET = max(5.0, float(os.getenv("TRACE_VIDEO_TIME_BUDGET", "30")))
TRACE_CACHE_CANDIDATES = max(1, int(os.getenv("TRACE_CACHE_CANDIDATES", "6")))
AUTO_DELETE_OLD_ORIGINALS = os.getenv("AUTO_DELETE_OLD_ORIGINALS", "false").lower() == "true"

IMAGE_PREVIEW_MAX_DIM = int(os.getenv("IMAGE_PREVIEW_MAX_DIM", "2048"))
IMAGE_PREVIEW_QUALITY = int(os.getenv("IMAGE_PREVIEW_QUALITY", "92"))
IMAGE_CACHE_VERSION = os.getenv("IMAGE_CACHE_VERSION", "fullres-jpeg-444-v1")

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data" if Path("/var/data").is_dir() else "data"))
REVEALS_DIR = DATA_DIR / "reveals"
CACHE_DIR = DATA_DIR / "cache"
TMP_DIR = DATA_DIR / "tmp"
META_FILE = DATA_DIR / "current_reveal.json"
LEDGER_FILE = DATA_DIR / "served.jsonl"
for d in (REVEALS_DIR, CACHE_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
REVEAL_ID_RE = re.compile(r"^\d+_[0-9a-f]{8}$")
URL_RE = re.compile(r"https?://[^\s<>]+", re.I)
HTTP_USER_AGENT = "RevealBot/1.0 (+https://discord.com/)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("revealbot")

PROCESS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
# One build lock per user is enough because the bot only serves CURRENT_REVEAL.
# Keeping the lock object avoids a race where an old waiter and a new request get
# different locks for the same user.
USER_LOCKS: dict[int, asyncio.Lock] = {}
USER_LOCKS_GUARD = threading.Lock()
PENDING_UPLOADS: dict[tuple[int, int, int], float] = {}


def _prune_pending_uploads() -> None:
    now = time.monotonic()
    for key, deadline in list(PENDING_UPLOADS.items()):
        if deadline <= now:
            PENDING_UPLOADS.pop(key, None)


# Protects AUTO_DELETE_OLD_ORIGINALS/cache pruning from removing a reveal while a
# personalized build is still reading it.
ACTIVE_REVEALS: defaultdict[str, int] = defaultdict(int)
ACTIVE_REVEALS_LOCK = threading.Lock()

# The served ledger can grow over time. Index it once instead of scanning the whole
# JSONL file for every /trace request.
SERVED_INDEX: Optional[set[tuple[str, int]]] = None
SERVED_INDEX_LOCK = threading.RLock()


class RevealRejected(Exception):
    pass


def safe_json_write(path: Path, payload: dict) -> None:
    # Write to a sibling temp file then replace atomically so a crash cannot leave
    # metadata half-written.
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with tmp.open("r+b") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _original_in(directory: Path) -> Optional[Path]:
    try:
        for f in directory.glob("original.*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                return f
    except OSError:
        log.exception("Could not inspect reveal directory %s", directory)
    return None


def _metadata_for(directory: Path) -> Optional[dict]:
    path = directory / "metadata.json"
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Could not read %s", path)
    original = _original_in(directory)
    if not original:
        return None
    return {
        "reveal_id": directory.name,
        "kind": "image" if original.suffix.lower() in IMAGE_EXTS else "video",
        "path": str(original),
        "filename": original.name,
        "created_at": int(directory.stat().st_mtime),
    }


def all_reveals() -> list[dict]:
    out: list[dict] = []
    try:
        directories = list(REVEALS_DIR.iterdir())
    except OSError:
        log.exception("Could not list %s", REVEALS_DIR)
        return out

    for directory in directories:
        if not directory.is_dir() or not REVEAL_ID_RE.fullmatch(directory.name):
            continue
        try:
            row = _metadata_for(directory)
            if row and Path(row["path"]).is_file():
                out.append(row)
        except OSError:
            # A concurrent prune/delete can legitimately make a directory vanish.
            continue
    return sorted(out, key=lambda r: int(r.get("created_at", 0) or 0), reverse=True)


def load_current_reveal() -> Optional[dict]:
    try:
        if META_FILE.exists():
            row = json.loads(META_FILE.read_text(encoding="utf-8"))
            if Path(row.get("path", "")).is_file():
                return row
    except Exception:
        log.exception("Could not read %s", META_FILE)
    rows = all_reveals()
    return rows[0] if rows else None


CURRENT_REVEAL = load_current_reveal()


def find_reveals(reveal_id: Optional[str]) -> list[dict]:
    if reveal_id:
        if not REVEAL_ID_RE.fullmatch(reveal_id):
            return []
        directory = REVEALS_DIR / reveal_id
        row = _metadata_for(directory) if directory.is_dir() else None
        return [row] if row and Path(row["path"]).is_file() else []
    return all_reveals()


def member_is_booster(member: discord.Member) -> bool:
    return bool(BOOSTER_ROLE_ID and any(r.id == BOOSTER_ROLE_ID for r in member.roles)) or member.premium_since is not None


def _load_served_index() -> set[tuple[str, int]]:
    global SERVED_INDEX
    if SERVED_INDEX is not None:
        return SERVED_INDEX
    index: set[tuple[str, int]] = set()
    if LEDGER_FILE.exists():
        try:
            with LEDGER_FILE.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        reveal_id = row.get("reveal_id")
                        user_id = row.get("user_id")
                        if isinstance(reveal_id, str) and user_id is not None:
                            index.add((reveal_id, int(user_id)))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
        except OSError:
            log.exception("Could not read ledger while building its index.")
    SERVED_INDEX = index
    return index


def ledger_append(reveal_id: str, user_id: int) -> None:
    row = (reveal_id, int(user_id))
    try:
        with SERVED_INDEX_LOCK:
            with LEDGER_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": int(time.time()), "reveal_id": reveal_id, "user_id": int(user_id)}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            _load_served_index().add(row)
    except OSError:
        log.exception("Could not write ledger.")


def ledger_served(reveal_id: str, user_id: int) -> bool:
    with SERVED_INDEX_LOCK:
        return (reveal_id, int(user_id)) in _load_served_index()


def served_user_ids(reveal_id: str) -> list[int]:
    """Return users recorded as having received a reveal, without rescanning JSONL."""
    with SERVED_INDEX_LOCK:
        return sorted(
            int(user_id)
            for rid, user_id in _load_served_index()
            if rid == reveal_id
        )


def cached_delivery_paths(reveal_id: str, user_id: int) -> list[Path]:
    """Return recent personalized cache files for one served user/reveal."""
    directory = CACHE_DIR / reveal_id
    if not directory.is_dir():
        return []
    prefix = f"user_{int(user_id)}_"
    try:
        paths = [
            p for p in directory.iterdir()
            if p.is_file()
            and p.name.startswith(prefix)
            and p.suffix.lower() in IMAGE_EXTS | {".mp4"}
        ]
    except OSError:
        return []
    paths.sort(
        key=lambda p: p.stat().st_mtime_ns if p.exists() else 0,
        reverse=True,
    )
    return paths[:TRACE_CACHE_CANDIDATES]


def compute_video_target_bytes(upload_limit: Optional[int]) -> int:
    configured = int(max(1.0, VIDEO_TARGET_MAX_MB) * 1048576)
    if not upload_limit or upload_limit <= 0:
        return configured
    # Leave a little room under Discord's hard limit so encoder/container overhead
    # and any minor size drift do not turn a successful encode into a failed upload.
    margin = min(512 * 1024, max(128 * 1024, int(upload_limit * 0.025)))
    return min(configured, max(256 * 1024, int(upload_limit) - margin))


def is_allowed_media(filename: str, content_type: Optional[str]) -> tuple[bool, Optional[str], Optional[str]]:
    ext = Path(filename).suffix.lower()
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if ext in IMAGE_EXTS or (normalized_type.startswith("image/") and normalized_type != "image/gif"):
        return True, "image", ext if ext in IMAGE_EXTS else ".png"
    if ext in VIDEO_EXTS or normalized_type.startswith("video/"):
        return True, "video", ext if ext in VIDEO_EXTS else ".mp4"
    return False, None, None


def validate_media(path: Path, kind: str) -> dict:
    if kind == "image":
        try:
            with Image.open(path) as probe:
                width, height = probe.size
                pixels = width * height
            if pixels > MAX_IMAGE_PIXELS:
                raise RevealRejected(
                    f"That image is {pixels / 1e6:.0f} MP; the limit is {MAX_IMAGE_PIXELS / 1e6:.0f} MP."
                )
            wm._load_rgb(path)
            return {"width": width, "height": height}
        except RevealRejected:
            raise
        except Exception as exc:
            raise RevealRejected(f"I couldn't read that as an image ({type(exc).__name__}).") from exc

    try:
        info = wm.video_info(path, MAX_VIDEO_HEIGHT)
    except Exception as exc:
        raise RevealRejected("I couldn't decode that video. Try re-exporting it as H.264 MP4.") from exc
    if info["duration"] > MAX_VIDEO_SECONDS:
        info["truncated_to"] = MAX_VIDEO_SECONDS
    return info


async def _save_reveal_file(src: Path, filename: str, kind: str, normalized_ext: str) -> dict:
    global CURRENT_REVEAL
    reveal_id = f"{int(time.time())}_{secrets.token_hex(4)}"
    directory = REVEALS_DIR / reveal_id
    destination = directory / f"original{normalized_ext}"
    try:
        info = await asyncio.to_thread(validate_media, src, kind)
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
        metadata = {
            "reveal_id": reveal_id,
            "kind": kind,
            "path": str(destination),
            "filename": filename,
            "created_at": int(time.time()),
            **info,
        }
        safe_json_write(directory / "metadata.json", metadata)
        safe_json_write(META_FILE, metadata)
        CURRENT_REVEAL = metadata
        await asyncio.to_thread(prune_old_data, reveal_id)
        return metadata
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def save_new_reveal_from_attachment(attachment: discord.Attachment) -> dict:
    ok, kind, ext = is_allowed_media(attachment.filename, attachment.content_type)
    if not ok:
        raise RevealRejected("Please send an image (jpg/png/webp) or a video (mp4/mov/webm/mkv/avi/m4v).")
    if attachment.size > MAX_UPLOAD_BYTES:
        raise RevealRejected(
            f"That file is {attachment.size / 1048576:.0f} MB; the limit is {MAX_UPLOAD_BYTES / 1048576:.0f} MB."
        )
    tmp = TMP_DIR / f"upload_{secrets.token_hex(8)}{ext}"
    try:
        await attachment.save(tmp, use_cached=False)
        return await _save_reveal_file(tmp, attachment.filename, kind, ext)
    finally:
        tmp.unlink(missing_ok=True)


def _public_ip(host: str) -> bool:
    """Return True only if every resolved address is globally routable.

    This is deliberately conservative: a hostname that resolves to a mix of
    public and private/link-local addresses is rejected rather than risking SSRF.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addresses = {ipaddress.ip_address(item[4][0].split("%", 1)[0]) for item in infos}
        return bool(addresses) and all(ip.is_global for ip in addresses)
    except (socket.gaierror, ValueError):
        return False


async def download_media_url(url: str) -> tuple[Path, str, str, str]:
    current = url.strip().strip("<>")
    if not URL_RE.fullmatch(current):
        raise RevealRejected("Send one direct http(s) media URL.")

    timeout = aiohttp.ClientTimeout(total=180, connect=15, sock_read=45)
    connector = aiohttp.TCPConnector(limit=4, limit_per_host=2, ttl_dns_cache=30)
    path: Optional[Path] = None
    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            raise_for_status=False,
            headers={"User-Agent": HTTP_USER_AGENT},
        ) as session:
            for _ in range(5):
                parsed = urlparse(current)
                if (
                    parsed.scheme.lower() not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or not await asyncio.to_thread(_public_ip, parsed.hostname)
                ):
                    raise RevealRejected("That media URL is not allowed.")

                async with session.get(current, allow_redirects=False) as response:
                    if 300 <= response.status < 400 and response.headers.get("Location"):
                        current = urljoin(current, response.headers["Location"])
                        continue
                    if response.status != 200:
                        raise RevealRejected(f"The media URL returned HTTP {response.status}.")

                    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                    try:
                        size = int(response.headers.get("Content-Length", "0") or 0)
                    except ValueError:
                        size = 0
                    if size > MAX_URL_BYTES:
                        raise RevealRejected(
                            f"That URL points to {size / 1048576:.0f} MB; the limit is {MAX_URL_BYTES / 1048576:.0f} MB."
                        )

                    ext = Path(parsed.path).suffix.lower()
                    if ext not in IMAGE_EXTS | VIDEO_EXTS:
                        ext = mimetypes.guess_extension(content_type) or ""
                    fake_name = f"download{ext or '.bin'}"
                    ok, kind, normalized_ext = is_allowed_media(fake_name, content_type)
                    if not ok:
                        raise RevealRejected("The URL does not point to a supported image or video.")

                    path = TMP_DIR / f"url_{secrets.token_hex(8)}{normalized_ext}"
                    written = 0
                    with path.open("wb") as fh:
                        async for chunk in response.content.iter_chunked(1024 * 256):
                            written += len(chunk)
                            if written > MAX_URL_BYTES:
                                raise RevealRejected("The media URL is too large.")
                            fh.write(chunk)
                    return path, kind, normalized_ext, Path(parsed.path).name or fake_name
    except RevealRejected:
        if path:
            path.unlink(missing_ok=True)
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        if path:
            path.unlink(missing_ok=True)
        raise RevealRejected(f"I couldn't download that media URL ({type(exc).__name__}).") from exc
    finally:
        # The connector is owned by this short-lived session, so there is no
        # separate close operation needed here.
        pass
    raise RevealRejected("Too many redirects.")


async def save_new_reveal_from_url(url: str) -> dict:
    path: Optional[Path] = None
    try:
        path, kind, ext, filename = await download_media_url(url)
        return await _save_reveal_file(path, filename, kind, ext)
    finally:
        if path:
            path.unlink(missing_ok=True)


def _active_reveal(reveal_id: str, delta: int) -> None:
    with ACTIVE_REVEALS_LOCK:
        ACTIVE_REVEALS[reveal_id] += delta
        if ACTIVE_REVEALS[reveal_id] <= 0:
            ACTIVE_REVEALS.pop(reveal_id, None)


def prune_old_data(keep_reveal_id: str) -> None:
    try:
        cache_children = list(CACHE_DIR.iterdir())
    except OSError:
        cache_children = []

    for child in cache_children:
        if not child.is_dir() or child.name == keep_reveal_id:
            continue
        with ACTIVE_REVEALS_LOCK:
            active = child.name in ACTIVE_REVEALS
        if active:
            continue
        shutil.rmtree(child, ignore_errors=True)

    if AUTO_DELETE_OLD_ORIGINALS:
        try:
            reveal_children = list(REVEALS_DIR.iterdir())
        except OSError:
            reveal_children = []
        for directory in reveal_children:
            if not directory.is_dir() or directory.name == keep_reveal_id:
                continue
            with ACTIVE_REVEALS_LOCK:
                active = directory.name in ACTIVE_REVEALS
            if active:
                continue
            shutil.rmtree(directory, ignore_errors=True)

    cutoff = time.time() - 3600
    try:
        leftovers = list(TMP_DIR.iterdir())
    except OSError:
        leftovers = []
    for leftover in leftovers:
        try:
            if leftover.is_file() and leftover.stat().st_mtime < cutoff:
                leftover.unlink(missing_ok=True)
        except OSError:
            pass


def _select_video_preset(source: Path) -> str:
    """Choose an encoder preset that preserves more quality at the same size budget."""
    configured = VIDEO_PRESET or "auto"
    if configured != "auto":
        return configured

    try:
        info = wm.video_info(source, MAX_VIDEO_HEIGHT)
        duration = float(info.get("duration", 0.0) or 0.0)
    except Exception:
        duration = 0.0

    # Faster presets spend more CPU-saving shortcuts for encoding speed; slower
    # presets generally use the available bitrate more efficiently. Long videos
    # benefit most because they have fewer bits per second under the same target.
    if duration >= 180:
        return "slow"
    if duration >= 75:
        return "medium"
    return "fast"


def _build_sync(
    source: Path,
    kind: str,
    output: Path,
    user_id: int,
    reveal_id: str,
    target_bytes: Optional[int],
) -> dict:
    started = time.perf_counter()
    if kind == "image":
        info = wm.render_image_delivery(
            source,
            output,
            user_id,
            WM_KEY,
            reveal_id,
            amp=IMAGE_AMP,
            target_bytes=target_bytes,
        )
    else:
        preset = _select_video_preset(source)
        info = wm.embed_video(
            source,
            output,
            user_id,
            WM_KEY,
            reveal_id,
            amp=VIDEO_AMP,
            preset=preset,
            max_seconds=MAX_VIDEO_SECONDS,
            max_height=MAX_VIDEO_HEIGHT,
            target_bytes=target_bytes,
            audio_kbps=VIDEO_AUDIO_KBPS,
            max_bpp=VIDEO_MAX_BPP,
        )
        info["preset"] = preset
    info["seconds"] = round(time.perf_counter() - started, 1)
    return info


def _get_user_lock(user_id: int) -> asyncio.Lock:
    # Accessed on the bot event loop in normal operation; a tiny guard also makes
    # startup/tests that call this helper from different threads deterministic.
    with USER_LOCKS_GUARD:
        return USER_LOCKS.setdefault(user_id, asyncio.Lock())


def cache_path_for(reveal: dict, user_id: int, target_bytes: Optional[int]) -> Path:
    base = CACHE_DIR / reveal["reveal_id"]
    target_token = str(int(target_bytes or 0))
    if reveal["kind"] == "video":
        # Use the exact target, not rounded MBs. Two Discord limits that differ by
        # less than 1 MB must never share a cache entry because the larger file can
        # be rejected by the smaller limit.
        return base / f"user_{user_id}_{VIDEO_CACHE_VERSION}_{target_token}b.mp4"
    return base / f"user_{user_id}_{IMAGE_CACHE_VERSION}_{target_token}b.jpg"


def _ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


async def build_personalized_reveal(
    reveal: dict,
    user_id: int,
    *,
    video_target_bytes: Optional[int] = None,
) -> tuple[Path, bool]:
    # Mark the reveal active before taking the user lock. A new upload can arrive
    # while an existing interaction is queued behind another build; pruning must
    # not delete that interaction's source in the meantime.
    _active_reveal(reveal["reveal_id"], +1)
    try:
        source = Path(reveal["path"])
        if not source.is_file():
            raise RuntimeError("There is currently no valid reveal.")

        output = cache_path_for(reveal, user_id, video_target_bytes)
        if _ready(output):
            return output, True

        async with _get_user_lock(user_id):
            if _ready(output):
                return output, True
            async with PROCESS_SEMAPHORE:
                output.parent.mkdir(parents=True, exist_ok=True)
                tmp = output.with_name(f".{output.stem}.tmp{secrets.token_hex(4)}{output.suffix}")
                try:
                    info = await asyncio.to_thread(
                        _build_sync,
                        source,
                        reveal["kind"],
                        tmp,
                        user_id,
                        reveal["reveal_id"],
                        video_target_bytes,
                    )
                    if not _ready(tmp):
                        raise RuntimeError("The watermark encoder produced an empty file.")
                    tmp.replace(output)
                finally:
                    tmp.unlink(missing_ok=True)

        log.info("Built %s reveal=%s user=%s %s", reveal["kind"], reveal["reveal_id"], user_id, info)
        return output, False
    finally:
        _active_reveal(reveal["reveal_id"], -1)


def _components_v2_available() -> bool:
    """Return whether this discord.py build can render attached video in a MediaGallery."""
    return (
        hasattr(discord.ui, "LayoutView")
        and hasattr(discord.ui, "MediaGallery")
        and hasattr(discord.ui, "TextDisplay")
        and hasattr(discord, "MediaGalleryItem")
    )


def _build_media_gallery_view(filename: str, content: str):
    """Build a Components V2 media gallery around a locally uploaded attachment.

    Discord's current Components V2 MediaGallery supports video attachments via
    attachment://filename. This changes presentation only; the underlying MP4 is
    sent byte-for-byte from the existing personalized cache.
    """
    view = discord.ui.LayoutView(timeout=15 * 60)
    view.add_item(discord.ui.TextDisplay(content))
    gallery = discord.ui.MediaGallery()
    gallery.add_item(
        media=f"attachment://{filename}",
        description="Your private booster reveal",
        spoiler=False,
    )
    view.add_item(gallery)
    return view


async def send_personalized_reveal(interaction: discord.Interaction) -> None:
    """Build one personalized file and deliver it privately.

    Images and videos use a Components V2 MediaGallery when the installed discord.py
    supports it. This changes only the Discord presentation layer and does not
    re-encode the cached media. Older discord.py versions fall back to a normal
    attachment.
    """
    if not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message(
            "I couldn't verify your server membership.", ephemeral=True
        )
        return

    if not member_is_booster(interaction.user):
        await interaction.response.send_message(
            "âŒ This reveal is available to current server boosters only.",
            ephemeral=True,
        )
        return

    reveal = CURRENT_REVEAL
    if not reveal or not Path(reveal["path"]).is_file():
        await interaction.response.send_message(
            "There isn't a reveal uploaded right now.", ephemeral=True
        )
        return

    upload_limit = (
        getattr(interaction, "filesize_limit", None)
        or getattr(interaction, "attachment_size_limit", None)
        or (getattr(interaction.guild, "filesize_limit", None) if interaction.guild else None)
        or DEFAULT_UPLOAD_LIMIT
    )
    target = compute_video_target_bytes(upload_limit)

    await interaction.response.defer(ephemeral=True, thinking=True)
    started = time.perf_counter()

    async def fail(text: str) -> None:
        try:
            await interaction.edit_original_response(content=text, attachments=[], view=None)
        except discord.HTTPException:
            log.exception("Could not report reveal failure to user=%s", interaction.user.id)

    try:
        path, cached = await build_personalized_reveal(
            reveal,
            interaction.user.id,
            video_target_bytes=target,
        )
    except Exception:
        log.exception("Failed to build reveal for user=%s", interaction.user.id)
        await fail("âŒ I couldn't generate your personalized reveal. Please try again in a moment.")
        return

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes <= 0:
        log.error("Generated reveal is empty: %s", path)
        await fail("âŒ I couldn't generate your personalized reveal. Please try again in a moment.")
        return
    if size_bytes > upload_limit:
        log.error(
            "Reveal %s is %d bytes, over the %d byte limit",
            path,
            size_bytes,
            upload_limit,
        )
        await fail(
            f"âŒ The personalized file is too large to send here "
            f"({size_bytes / 1048576:.1f} MB vs a {upload_limit / 1048576:.1f} MB limit)."
        )
        return

    filename = "reveal.mp4" if reveal["kind"] == "video" else "reveal.jpg"
    content = "ðŸŽ¬ Your booster reveal:" if reveal["kind"] == "video" else "ðŸ–¼ï¸ Your booster reveal:"

    log.info(
        "Prepared reveal=%s user=%s kind=%s size=%d cached=%s build_seconds=%.2f delivery=%s",
        reveal["reveal_id"],
        interaction.user.id,
        reveal["kind"],
        size_bytes,
        cached,
        time.perf_counter() - started,
        "components-v2" if _components_v2_available() else "attachment",
    )

    delivered = False

    # discord.py 2.6+ can expose an attachment directly through a Components V2
    # MediaGallery. Use the same path for images and videos so mobile rendering
    # does not depend on the legacy attachment renderer.
    if _components_v2_available():
        try:
            reveal_file = discord.File(path, filename=filename, spoiler=False)
            try:
                # Edit the deferred original response rather than creating a follow-up.
                # A follow-up immediately after a deferred interaction can behave as
                # an edit of the original interaction response, so deleting the
                # "original" afterwards can accidentally delete the video itself.
                await interaction.edit_original_response(
                    content=None,
                    attachments=[reveal_file],
                    view=_build_media_gallery_view(filename, content),
                )
                delivered = True
            finally:
                reveal_file.close()
        except (discord.HTTPException, TypeError, AttributeError, OSError):
            log.exception(
                "Components V2 video delivery failed reveal=%s user=%s; falling back to attachment",
                reveal["reveal_id"],
                interaction.user.id,
            )
            delivered = False

    # Legacy attachment path. This remains the compatibility fallback for older
    # discord.py versions and any server/client/API combination that rejects V2.
    if not delivered:
        try:
            reveal_file = discord.File(path, filename=filename, spoiler=False)
            try:
                await interaction.edit_original_response(
                    content=content,
                    attachments=[reveal_file],
                    view=None,
                )
                delivered = True
            finally:
                reveal_file.close()
        except (discord.HTTPException, OSError, TypeError):
            log.exception(
                "edit_original_response failed reveal=%s user=%s",
                reveal["reveal_id"],
                interaction.user.id,
            )

    if not delivered:
        try:
            reveal_file = discord.File(path, filename=filename, spoiler=False)
            try:
                await interaction.followup.send(
                    content=content,
                    file=reveal_file,
                    ephemeral=True,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                delivered = True
            finally:
                reveal_file.close()
        except (discord.HTTPException, OSError, TypeError):
            log.exception(
                "followup.send failed reveal=%s user=%s",
                reveal["reveal_id"],
                interaction.user.id,
            )

    if not delivered:
        await fail("âŒ I couldn't deliver your personalized reveal. Please try again.")
        return

    await asyncio.to_thread(ledger_append, reveal["reveal_id"], interaction.user.id)


class RevealButtonView(discord.ui.View):
    def __init__(self) -> None:
        super().__init__(timeout=None)

    @discord.ui.button(
        label="View Reveal",
        style=discord.ButtonStyle.primary,
        emoji="ðŸ‘ï¸",
        custom_id="revealbot:view-reveal:v1",
    )
    async def view_reveal_button(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        del button
        await send_personalized_reveal(interaction)


class RevealBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents(guilds=True, messages=True, message_content=True)
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # Fixed custom_id + no timeout makes the button a persistent view across restarts.
        self.add_view(RevealButtonView())
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
        else:
            synced = await self.tree.sync()
        log.info("Synced %d command(s).", len(synced))

    async def on_message(self, message: discord.Message) -> None:
        _prune_pending_uploads()
        if message.author.bot or not message.guild:
            return
        key = (message.guild.id, message.channel.id, message.author.id)
        deadline = PENDING_UPLOADS.get(key)
        if not deadline:
            return
        if time.monotonic() > deadline:
            PENDING_UPLOADS.pop(key, None)
            return
        PENDING_UPLOADS.pop(key, None)

        if not isinstance(message.author, discord.Member) or not message.author.guild_permissions.manage_guild:
            return

        try:
            if len(message.attachments) > 1:
                await message.delete()
                await message.channel.send("âŒ Upload one file at a time.", delete_after=8)
                return
            if message.attachments:
                metadata = await save_new_reveal_from_attachment(message.attachments[0])
            else:
                match = URL_RE.fullmatch(message.content.strip())
                if not match:
                    await message.delete()
                    await message.channel.send(
                        "âŒ Send one direct media URL or one attached file.",
                        delete_after=8,
                    )
                    return
                metadata = await save_new_reveal_from_url(match.group(0))

            try:
                await message.delete()
            except discord.HTTPException:
                log.warning(
                    "Could not delete the admin's upload message; grant Manage Messages to keep originals hidden."
                )

            note = ""
            if metadata.get("truncated_to"):
                note = (
                    f"\nâš ï¸ The source is {metadata['duration']:.0f}s; viewers receive "
                    f"the first {metadata['truncated_to']}s."
                )
            await message.channel.send(
                f"âœ… Reveal `{metadata['reveal_id']}` is live ({metadata['kind']}). "
                f"Boosters can use `/view_reveal`.{note}",
                delete_after=12,
            )
        except RevealRejected as exc:
            try:
                await message.delete()
            except discord.HTTPException:
                pass
            await message.channel.send(f"âŒ {exc}", delete_after=10)
        except Exception:
            log.exception("Post-/upload ingestion failed.")
            try:
                await message.delete()
            except discord.HTTPException:
                pass
            await message.channel.send(
                "âŒ I couldn't ingest that media. Check the bot logs.",
                delete_after=10,
            )


bot = RevealBot()
bot_ready = threading.Event()


@bot.event
async def on_ready():
    bot_ready.set()
    log.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "?")


@bot.event
async def on_disconnect():
    bot_ready.clear()


@bot.event
async def on_resumed():
    bot_ready.set()


@bot.tree.command(name="upload", description="Admin: start an image/video reveal upload.")
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
async def upload(interaction: discord.Interaction):
    _prune_pending_uploads()
    key = (interaction.guild_id or 0, interaction.channel_id or 0, interaction.user.id)
    PENDING_UPLOADS[key] = time.monotonic() + UPLOAD_WAIT_SECONDS
    await interaction.response.send_message(
        "Send the reveal as the next message in this channel: attach the file or paste one direct media URL. "
        f"I'll remove that message after ingest. This expires in {UPLOAD_WAIT_SECONDS}s.",
        ephemeral=True,
    )


@bot.tree.command(name="view_reveal", description="Get your booster reveal.")
@app_commands.guild_only()
async def view_reveal(interaction: discord.Interaction):
    await send_personalized_reveal(interaction)


async def resolve_member(guild: discord.Guild, user_id: int) -> Optional[discord.Member]:
    member = guild.get_member(user_id)
    if member:
        return member
    try:
        return await guild.fetch_member(user_id)
    except discord.HTTPException:
        return None


@app_commands.context_menu(name="Add Reveal Button")
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
async def add_reveal_button(interaction: discord.Interaction, message: discord.Message):
    if not interaction.guild or not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message(
            "This app can only be used in a server.",
            ephemeral=True,
        )
        return
    if not interaction.user.guild_permissions.manage_guild:
        await interaction.response.send_message(
            "You need the Manage Server permission to add reveal buttons.",
            ephemeral=True,
        )
        return
    try:
        await message.reply(
            content="ðŸ‘ï¸ Booster reveal â€” tap the button below to open your private reveal.",
            view=RevealButtonView(),
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
    except discord.HTTPException:
        log.exception("Could not post reveal button reply to message %s", message.id)
        await interaction.response.send_message(
            "I couldn't reply with the reveal button. Check that I can view the channel and send messages there.",
            ephemeral=True,
        )
        return
    await interaction.response.send_message(
        "âœ… Posted a View Reveal button as a reply to the selected message.",
        ephemeral=True,
    )


bot.tree.add_command(add_reveal_button)


@bot.tree.command(name="trace", description="Admin: trace a leaked reveal to its Discord user.")
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
@app_commands.describe(
    file="Leaked image/video, or a frame from a video reveal",
    reveal_id="Optional reveal ID; omit to search every saved reveal",
    start_frame="Optional original frame hint for video tracing",
)
async def trace(
    interaction: discord.Interaction,
    file: discord.Attachment,
    reveal_id: Optional[str] = None,
    start_frame: app_commands.Range[int, 0, 200000] = 0,
):
    reveals = find_reveals(reveal_id)
    if not reveals:
        await interaction.response.send_message(
            "I couldn't find that reveal's original file.",
            ephemeral=True,
        )
        return

    for reveal in reveals:
        _active_reveal(reveal["reveal_id"], +1)

    if file.size > MAX_UPLOAD_BYTES:
        for reveal in reveals:
            _active_reveal(reveal["reveal_id"], -1)
        await interaction.response.send_message(
            f"That leak is {file.size / 1048576:.0f} MB; the limit is {MAX_UPLOAD_BYTES / 1048576:.0f} MB.",
            ephemeral=True,
        )
        return

    ok, leak_kind, ext = is_allowed_media(file.filename, file.content_type)
    if not ok:
        for reveal in reveals:
            _active_reveal(reveal["reveal_id"], -1)
        await interaction.response.send_message(
            "The leak must be an image or video.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    leak_path = TMP_DIR / f"leak_{secrets.token_hex(8)}{ext}"
    try:
        await file.save(leak_path, use_cached=False)
        trace_deadline = time.monotonic() + TRACE_TIME_BUDGET
        checked = 0

        async def trace_original(
            reveal: dict,
            *,
            delivery_height: Optional[int] = None,
            delivery_fps: Optional[float] = None,
        ) -> Optional[dict]:
            nonlocal checked

            if reveal["kind"] == "image":
                if leak_kind != "image":
                    return None
                trace_kind = "image"
            else:
                if leak_kind not in {"image", "video"}:
                    return None
                trace_kind = "video_frame" if leak_kind == "image" else "video"

            remaining = trace_deadline - time.monotonic()
            if remaining <= 0:
                return None
            kind_budget = TRACE_IMAGE_TIME_BUDGET if trace_kind == "image" or trace_kind == "video_frame" else TRACE_VIDEO_TIME_BUDGET
            kwargs = {
                "image_cell": IMAGE_CELL,
                "video_cell": VIDEO_CELL,
                "max_height": MAX_VIDEO_HEIGHT,
                "start_frame": int(start_frame),
                "time_budget": min(kind_budget, remaining),
            }
            if delivery_height is not None:
                kwargs["delivery_height"] = int(delivery_height)
            if delivery_fps is not None:
                kwargs["delivery_fps"] = float(delivery_fps)

            checked += 1
            try:
                async with PROCESS_SEMAPHORE:
                    result = await asyncio.to_thread(
                        wm.extract,
                        Path(reveal["path"]),
                        leak_path,
                        trace_kind,
                        WM_KEY,
                        reveal["reveal_id"],
                        **kwargs,
                    )
            except Exception as exc:
                log.info(
                    "Reveal %s did not decode with profile h=%s fps=%s: %s",
                    reveal["reveal_id"],
                    delivery_height,
                    delivery_fps,
                    exc,
                )
                return None

            if not result.get("valid"):
                return None

            return result

        for reveal in reveals:
            if time.monotonic() >= trace_deadline:
                break
            # Fast/accurate path for video: the personalized cache tells us the exact
            # delivery profile used for a served user, while extraction still compares
            # the leak against the unwatermarked reveal original.
            if reveal["kind"] == "video" and leak_kind in {"image", "video"}:
                served_ids = set(served_user_ids(reveal["reveal_id"]))
                profile_attempted: set[tuple[int, float]] = set()
                profile_paths_checked = 0
                for served_uid in served_ids:
                    if time.monotonic() >= trace_deadline or profile_paths_checked >= TRACE_CACHE_CANDIDATES:
                        break
                    for cached in cached_delivery_paths(reveal["reveal_id"], served_uid):
                        if time.monotonic() >= trace_deadline or profile_paths_checked >= TRACE_CACHE_CANDIDATES:
                            break
                        profile_paths_checked += 1
                        try:
                            profile = wm.video_info(cached, MAX_VIDEO_HEIGHT)
                            profile_key = (int(profile["height"]), round(float(profile["fps"]), 3))
                        except Exception as exc:
                            log.info("Could not read delivery profile %s: %s", cached, exc)
                            continue
                        if profile_key in profile_attempted:
                            continue
                        profile_attempted.add(profile_key)

                        result = await trace_original(
                            reveal,
                            delivery_height=profile_key[0],
                            delivery_fps=profile_key[1],
                        )
                        if result is None:
                            continue

                        uid = int(result["user_id"])
                        if uid not in served_ids:
                            log.info(
                                "Trace decoded unserved user=%s for reveal=%s profile=%sx%.3ffps",
                                uid,
                                reveal["reveal_id"],
                                profile_key[0],
                                profile_key[1],
                            )
                            continue

                        member = await resolve_member(interaction.guild, uid)
                        frame_text = f" â€¢ source frame `{result['frame']}`" if "frame" in result else ""
                        await interaction.followup.send(
                            f"ðŸ”Ž Watermark decoded (CRC verified): <@{uid}> (`{uid}`)"
                            f"{' â€” in this server' if member else ' â€” not currently in this server'}\n"
                            f"Reveal `{reveal['reveal_id']}`{frame_text} â€¢ served to this user: **yes**",
                            ephemeral=True,
                        )
                        log.warning(
                            "TRACE by %s: reveal=%s decoded_user=%s checked=%d source=original profile=%sx%.3ffps",
                            interaction.user.id,
                            reveal["reveal_id"],
                            uid,
                            checked,
                            profile_key[0],
                            profile_key[1],
                        )
                        return

            # Image delivery does not need a stored delivery profile: registration
            # already recovers resize/crop/translation against the original.
            if time.monotonic() >= trace_deadline:
                break
            result = await trace_original(reveal)
            if result is None:
                continue

            uid = int(result["user_id"])
            served = await asyncio.to_thread(ledger_served, reveal["reveal_id"], uid)
            member = await resolve_member(interaction.guild, uid)
            frame_text = f" â€¢ source frame `{result['frame']}`" if "frame" in result else ""
            await interaction.followup.send(
                f"ðŸ”Ž Watermark decoded (CRC verified): <@{uid}> (`{uid}`)"
                f"{' â€” in this server' if member else ' â€” not currently in this server'}\n"
                f"Reveal `{reveal['reveal_id']}`{frame_text} â€¢ served to this user: "
                f"{'**yes**' if served else '**no record**'}",
                ephemeral=True,
            )
            log.warning(
                "TRACE by %s: reveal=%s decoded_user=%s checked=%d source=original",
                interaction.user.id,
                reveal["reveal_id"],
                uid,
                checked,
            )
            return

        timed_out = time.monotonic() >= trace_deadline
        suffix = " The trace time budget was reached before every saved reveal could be checked." if timed_out else ""
        await interaction.followup.send(
            f"No valid watermark found across {checked} trace candidate(s). "
            f"The leak may be too cropped/edited/compressed or belong to a reveal whose original was deleted."
            f"{suffix}",
            ephemeral=True,
        )
    except Exception:
        log.exception("Trace failed.")
        await interaction.followup.send(
            "âŒ Trace failed. Check the bot logs.",
            ephemeral=True,
        )
    finally:
        leak_path.unlink(missing_ok=True)
        for reveal in reveals:
            _active_reveal(reveal["reveal_id"], -1)


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
):
    if isinstance(error, app_commands.MissingPermissions):
        message = "You need the **Manage Server** permission to use this command."
    else:
        log.error(
            "Slash command error: %r",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
        message = "Something went wrong while running that command."
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except discord.HTTPException:
        pass


async def main():
    health_server = start_health_server(bot_ready.is_set)
    log.info("Health server listening on 0.0.0.0:%s", health_server.server_port)
    try:
        async with bot:
            await bot.start(TOKEN)
    finally:
        health_server.shutdown()
        health_server.server_close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
