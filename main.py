from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import signal
import socket
import sys
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv
from PIL import Image

                                                                                
                                                                 
from health_server import start_health_server
from reveal_history import ArchiveServedUsersView, RevealHistoryError, RevealHistoryStore, make_history_command
from stresstest import install_stress_test
from status import StatusTracker

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

TOKEN = os.getenv("DISCORD_TOKEN")
SECRET = os.getenv("WATERMARK_SECRET")
if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing.")
if not SECRET or len(SECRET) < 16:
    raise RuntimeError("WATERMARK_SECRET is missing/too short.")
WM_KEY = SECRET.encode("utf-8")

GUILD_ID = os.getenv("GUILD_ID")
SYNC_COMMANDS = os.getenv("SYNC_COMMANDS", "true").lower() == "true"
BOOSTER_ROLE_ID = int(os.getenv("BOOSTER_ROLE_ID", "0") or 0)
STAFF_TEAM_ROLE_ID = int(os.getenv("STAFF_TEAM_ROLE_ID", "0") or 0)
REVEAL_ARCHIVE_CHANNEL_ID = int(os.getenv("REVEAL_ARCHIVE_CHANNEL_ID", "0") or 0)
if not BOOSTER_ROLE_ID or not STAFF_TEAM_ROLE_ID or not REVEAL_ARCHIVE_CHANNEL_ID:
    raise RuntimeError("BOOSTER_ROLE_ID, STAFF_TEAM_ROLE_ID and REVEAL_ARCHIVE_CHANNEL_ID must be configured.")

                                                                                     
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "180"))
MAX_VIDEO_HEIGHT = int(os.getenv("MAX_VIDEO_HEIGHT", "720"))
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "superfast").strip().lower() or "superfast"
VIDEO_AUDIO_KBPS = int(os.getenv("VIDEO_AUDIO_KBPS", "48"))
VIDEO_TARGET_MAX_MB = float(os.getenv("VIDEO_TARGET_MAX_MB", "48"))
DEFAULT_UPLOAD_LIMIT = 20 * 1048576
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "120")) * 1048576)
MAX_URL_BYTES = min(MAX_UPLOAD_BYTES, int(float(os.getenv("MAX_MEDIA_URL_MB", "120")) * 1048576))

                           
_TRACE_MP = float(os.getenv("TRACE_IMAGE_MAX_MEGAPIXELS", "18"))
_EMBED_MP = float(os.getenv("EMBED_IMAGE_MAX_MEGAPIXELS", "40"))
MAX_IMAGE_PIXELS = int(min(float(os.getenv("MAX_IMAGE_MEGAPIXELS", "40")), _TRACE_MP, _EMBED_MP) * 1_000_000)
TRACE_LEAK_MAX_PIXELS = int(_TRACE_MP * 1_000_000)
TRACE_TIME_BUDGET = max(10.0, float(os.getenv("TRACE_TIME_BUDGET", "30")))
TRACE_IMAGE_TIME_BUDGET = max(3.0, float(os.getenv("TRACE_IMAGE_TIME_BUDGET", "8")))
TRACE_VIDEO_TIME_BUDGET = max(5.0, float(os.getenv("TRACE_VIDEO_TIME_BUDGET", "20")))
TRACE_MAX_REVEALS = max(1, int(os.getenv("TRACE_MAX_REVEALS", "12")))

                     
IMAGE_BUILD_TIMEOUT = float(os.getenv("IMAGE_BUILD_TIMEOUT", "240"))
VIDEO_BUILD_TIMEOUT = float(os.getenv("VIDEO_BUILD_TIMEOUT", "600"))
CACHE_MAX_BYTES = int(float(os.getenv("CACHE_MAX_MB", "768")) * 1048576)
CACHE_TTL_SECONDS = max(300, int(os.getenv("CACHE_TTL_SECONDS", "3600")))
AUTO_DELETE_OLD_ORIGINALS = os.getenv("AUTO_DELETE_OLD_ORIGINALS", "true").lower() == "true"

IMAGE_PREVIEW_MAX_DIM = int(os.getenv("IMAGE_PREVIEW_MAX_DIM", "2048"))
IMAGE_PREVIEW_QUALITY = int(os.getenv("IMAGE_PREVIEW_QUALITY", "88"))
IMAGE_CACHE_VERSION = os.getenv("IMAGE_CACHE_VERSION", "fullres-jpeg-444-v2-free512")
VIDEO_CACHE_VERSION = f"{os.getenv('VIDEO_CACHE_VERSION', 'v23-free512-speed-v1')}-quality-first-crf"

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data" if Path("/var/data").is_dir() else "data"))
REVEALS_DIR = DATA_DIR / "reveals"
CACHE_DIR = DATA_DIR / "cache"
TMP_DIR = DATA_DIR / "tmp"
for d in (REVEALS_DIR, CACHE_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
REVEAL_ID_RE = re.compile(r"^\d+_[0-9a-f]{8}$")
URL_RE = re.compile(r"https?://[^\s<>]+", re.I)
HTTP_USER_AGENT = "RevealBot/1.1 (+https://discord.com/)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("revealbot")
STATUS = StatusTracker()


                                                                             
                   
                                                                             

_PROCESS_SEM: Optional[asyncio.Semaphore] = None


def process_sem() -> asyncio.Semaphore:
    global _PROCESS_SEM
    if _PROCESS_SEM is None:
        _PROCESS_SEM = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    return _PROCESS_SEM


USER_LOCKS: "weakref.WeakValueDictionary[int, asyncio.Lock]" = weakref.WeakValueDictionary()
REVEAL_RESTORE_LOCKS: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()
ACTIVE_REVEALS: defaultdict[str, int] = defaultdict(int)
ACTIVE_REVEALS_LOCK = threading.Lock()
_V2_OK = True
WORKER_PROCS: set = set()


@dataclass
class RevealBuildTicket:
    ticket_id: int
    user_id: int
    reveal_id: str
    kind: str
    queued_at: float
    initial_ahead: int = 0
    state: str = "queued"


_QUEUE_LOCK: Optional[asyncio.Lock] = None
_REVEAL_QUEUE: list[RevealBuildTicket] = []
_NEXT_QUEUE_ID = 0
_STATUS_CHANGED = asyncio.Event()
_STATUS_REVISION = 0
_RECENT_BUILD_SECONDS: dict[str, deque] = {
    "image": deque(maxlen=20),
    "video": deque(maxlen=20),
}


def _record_build_time(kind: str, seconds: float) -> None:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return
    if value > 0:
        _RECENT_BUILD_SECONDS.setdefault(str(kind), deque(maxlen=20)).append(value)


def _estimated_build_seconds(kind: str) -> float:
    values = _RECENT_BUILD_SECONDS.get(str(kind), ())
    if values:
        return max(0.5, min(60.0, sum(values) / len(values)))
    if str(kind) == "image":
        return 1.0
    return 8.0


def _notify_status_changed() -> None:
    global _STATUS_REVISION
    _STATUS_REVISION += 1
    _STATUS_CHANGED.set()


def _estimated_queue_wait(ticket: RevealBuildTicket, position: int) -> float:
    if ticket.state == "running":
        return 0.0
                                                                          
                                                                              
    return max(1.0, float(max(1, position)) * _estimated_build_seconds(ticket.kind))


def queue_lock() -> asyncio.Lock:
    global _QUEUE_LOCK
    if _QUEUE_LOCK is None:
        _QUEUE_LOCK = asyncio.Lock()
    return _QUEUE_LOCK


async def _queue_add(reveal: dict, user_id: int) -> RevealBuildTicket:
    global _NEXT_QUEUE_ID
    async with queue_lock():
        _NEXT_QUEUE_ID += 1
        active_before = [t for t in _REVEAL_QUEUE if t.state in {"queued", "running"}]
        ticket = RevealBuildTicket(
            ticket_id=_NEXT_QUEUE_ID,
            user_id=int(user_id),
            reveal_id=str(reveal["reveal_id"]),
            kind=str(reveal["kind"]),
            queued_at=time.perf_counter(),
            initial_ahead=len(active_before),
        )
        _REVEAL_QUEUE.append(ticket)
        _notify_status_changed()
        return ticket


async def _queue_position(ticket: RevealBuildTicket) -> int:
    async with queue_lock():
        ahead = 0
        for other in _REVEAL_QUEUE:
            if other is ticket:
                break
            if other.state in {"queued", "running"}:
                ahead += 1
        return ahead + 1


async def _queue_mark_running(ticket: RevealBuildTicket) -> None:
    async with queue_lock():
        ticket.state = "running"
        _notify_status_changed()


async def _queue_remove(ticket: RevealBuildTicket) -> None:
    async with queue_lock():
        ticket.state = "done"
        try:
            _REVEAL_QUEUE.remove(ticket)
        except ValueError:
            pass
        _notify_status_changed()


def _queue_percent(ticket: RevealBuildTicket, position: int) -> int:
                                                                                           
                                                                            
    if ticket.state == "running" or position <= 1:
        return 90
    initial = max(1, int(ticket.initial_ahead))
    completed_ahead = max(0, initial - max(0, position - 1))
    return max(1, min(89, 5 + round(84 * completed_ahead / initial)))


async def _send_queue_progress(progress_callback, percent: int, position: int, estimated_wait: float) -> None:
    if progress_callback is None:
        return
    try:
        await progress_callback(int(percent), int(position), float(estimated_wait))
    except TypeError:
                                                                              
        await progress_callback(int(percent))


async def _wait_for_build_slot(
    ticket: RevealBuildTicket,
    progress_callback=None,
) -> None:
    sem = process_sem()
    acquire_task = asyncio.create_task(sem.acquire())
    acquired = False
    last_percent = -1
    last_update = 0.0
    try:
        while not acquire_task.done():
            position = await _queue_position(ticket)
            percent = _queue_percent(ticket, position)
            now = time.perf_counter()
            if (progress_callback is not None and ticket.initial_ahead > 0
                    and now - last_update >= 0.9 and percent != last_percent):
                try:
                    await _send_queue_progress(
                        progress_callback,
                        percent,
                        position,
                        _estimated_queue_wait(ticket, position),
                    )
                except Exception:
                    log.debug("Queue progress update failed", exc_info=True)
                last_percent = percent
                last_update = now
            await asyncio.sleep(0.35)
        await acquire_task
        acquired = True
        await _queue_mark_running(ticket)
        if progress_callback is not None and ticket.initial_ahead > 0:
            try:
                await _send_queue_progress(
                    progress_callback,
                    90,
                    1,
                    0.0,
                )
            except Exception:
                log.debug("Final queue progress update failed", exc_info=True)
    except BaseException:
        if not acquire_task.done():
            acquire_task.cancel()
            try:
                await acquire_task
            except asyncio.CancelledError:
                pass
        elif acquired:
                                                                                 
                                                                                
            sem.release()
        raise


class RevealRejected(Exception):
    pass


class WorkerError(RuntimeError):
    pass


def _active_reveal(reveal_id: str, delta: int) -> None:
    with ACTIVE_REVEALS_LOCK:
        ACTIVE_REVEALS[reveal_id] += delta
        if ACTIVE_REVEALS[reveal_id] <= 0:
            ACTIVE_REVEALS.pop(reveal_id, None)


def _get_user_lock(user_id: int) -> asyncio.Lock:
    lock = USER_LOCKS.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        USER_LOCKS[user_id] = lock
    return lock


def _get_reveal_restore_lock(reveal_id: str) -> asyncio.Lock:
    lock = REVEAL_RESTORE_LOCKS.get(reveal_id)
    if lock is None:
        lock = asyncio.Lock()
        REVEAL_RESTORE_LOCKS[reveal_id] = lock
    return lock


                                                                             
                
                                                                             

_WORKER_SRC = r'''
import json, sys
from pathlib import Path
import watermark as wm

def main():
    req = json.loads(sys.stdin.read())
    op, key, a = req["op"], bytes.fromhex(req["key"]), req["args"]
    if op == "image":
        res = wm.render_image_delivery(
            Path(a["src"]), Path(a["dst"]), a["user_id"], key, a["reveal_id"],
            amp=a["amp"], target_bytes=a["target_bytes"])
    elif op == "video":
        res = wm.embed_video(
            Path(a["src"]), Path(a["dst"]), a["user_id"], key, a["reveal_id"],
            amp=a["amp"], preset=a["preset"], max_seconds=a["max_seconds"],
            max_height=a["max_height"], target_bytes=a["target_bytes"],
            audio_kbps=a["audio_kbps"])
    elif op == "extract":
        res = wm.extract(
            Path(a["orig"]), Path(a["leak"]), a["kind"], key, a["reveal_id"], **a["kwargs"])
    elif op == "image_check":
        w, h = wm._load_rgb(Path(a["path"])).size
        res = {"width": w, "height": h}
    elif op == "video_info":
        res = wm.video_info(Path(a["path"]), a["max_height"])
    else:
        raise ValueError("unknown op " + str(op))
    sys.stdout.write("\n" + json.dumps(res) + "\n")

main()
'''

_WORKER_ENV = {
    **{k: v for k, v in os.environ.items() if k not in {"DISCORD_TOKEN", "WATERMARK_SECRET"}},
    "MALLOC_ARENA_MAX": os.getenv("MALLOC_ARENA_MAX", "1"),
    "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "1"),
    "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS", "1"),
    "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", "1"),
    "OPENCV_OPENCL_RUNTIME": os.getenv("OPENCV_OPENCL_RUNTIME", "disabled"),
    "PYTHONUNBUFFERED": "1",
}


def _kill_group(proc) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


async def run_worker(op: str, timeout: float, **args) -> dict:
    payload = json.dumps({"op": op, "key": WM_KEY.hex(), "args": args}).encode()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _WORKER_SRC,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(BASE_DIR),
        env=_WORKER_ENV,
        start_new_session=True,
    )
    WORKER_PROCS.add(proc)
    try:
        out, err = await asyncio.wait_for(proc.communicate(payload), timeout)
    except asyncio.TimeoutError as exc:
        _kill_group(proc)
        await proc.wait()
        raise WorkerError(f"{op} timed out after {timeout:.0f}s") from exc
    except BaseException:
        _kill_group(proc)
        try:
            await proc.wait()
        except Exception:
            pass
        raise
    finally:
        WORKER_PROCS.discard(proc)

    if proc.returncode != 0:
        tail = err.decode("utf-8", "replace").strip()[-1000:]
        raise WorkerError(f"{op} failed (exit {proc.returncode}): {tail}")
    lines = [ln for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]
    if not lines:
        raise WorkerError(f"{op} produced no output")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise WorkerError(f"{op} returned unreadable output") from exc


                                                                             
                 
                                                                             

def safe_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            continue
    return sorted(out, key=lambda r: int(r.get("created_at", 0) or 0), reverse=True)


_local_reveals = all_reveals()
CURRENT_REVEAL = _local_reveals[0] if _local_reveals else None
PERSISTED_REVEALS: dict[str, dict] = {}


def find_reveals(reveal_id: Optional[str]) -> list[dict]:
    if reveal_id:
        if not REVEAL_ID_RE.fullmatch(reveal_id):
            return []
        row = PERSISTED_REVEALS.get(reveal_id)
        if row:
            return [dict(row)]
        directory = REVEALS_DIR / reveal_id
        row = _metadata_for(directory) if directory.is_dir() else None
        return [row] if row and Path(row["path"]).is_file() else []

    rows = list(PERSISTED_REVEALS.values())
    if not rows:
        rows = all_reveals()
    return sorted(rows, key=lambda r: int(r.get("created_at", 0) or 0), reverse=True)


def member_has_role(member: discord.Member, role_id: int) -> bool:
    return any(role.id == role_id for role in member.roles)


def member_is_booster(member: discord.Member) -> bool:
    return member_has_role(member, BOOSTER_ROLE_ID)


def is_booster(interaction: discord.Interaction) -> bool:
    return isinstance(interaction.user, discord.Member) and member_is_booster(interaction.user)


def is_staff(interaction: discord.Interaction) -> bool:
    return isinstance(interaction.user, discord.Member) and member_has_role(interaction.user, STAFF_TEAM_ROLE_ID)


                                                                             
                              
                                                                             

def compute_video_target_bytes(upload_limit: Optional[int]) -> int:
    configured = int(max(1.0, VIDEO_TARGET_MAX_MB) * 1048576)
    if not upload_limit or upload_limit <= 0:
        return configured
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


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as probe:
        return probe.size


async def validate_media(path: Path, kind: str) -> dict:
    if kind == "image":
        try:
            width, height = await asyncio.to_thread(_image_size, path)
        except Exception as exc:
            raise RevealRejected(f"I couldn't read that as an image ({type(exc).__name__}).") from exc
        pixels = width * height
        if pixels > MAX_IMAGE_PIXELS:
            raise RevealRejected(
                f"That image is {pixels / 1e6:.0f} MP; the limit is {MAX_IMAGE_PIXELS / 1e6:.0f} MP."
            )
        try:
            async with process_sem():
                await run_worker("image_check", 90, path=str(path))
        except Exception as exc:
            raise RevealRejected("I couldn't read that as an image (it may be corrupt).") from exc
        return {"width": width, "height": height}

    try:
        async with process_sem():
            info = await run_worker("video_info", 75, path=str(path), max_height=MAX_VIDEO_HEIGHT)
    except Exception as exc:
        raise RevealRejected("I couldn't decode that video. Try re-exporting it as H.264 MP4.") from exc
    if info["duration"] > MAX_VIDEO_SECONDS:
        info["truncated_to"] = MAX_VIDEO_SECONDS
    return info


def _commit_reveal(src: Path, directory: Path, destination: Path, metadata: dict) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(destination))
        safe_json_write(directory / "metadata.json", metadata)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def _save_reveal_file(src: Path, filename: str, kind: str, normalized_ext: str) -> dict:
    global CURRENT_REVEAL
    reveal_id = f"{int(time.time())}_{secrets.token_hex(4)}"
    directory = REVEALS_DIR / reveal_id
    destination = directory / f"original{normalized_ext}"
    info = await validate_media(src, kind)
    metadata = {
        "reveal_id": reveal_id,
        "kind": kind,
        "path": str(destination),
        "filename": filename,
        "created_at": int(time.time()),
        **info,
    }
    await asyncio.to_thread(_commit_reveal, src, directory, destination, metadata)
    CURRENT_REVEAL = metadata
    STATUS.reset(reveal_id)
    _notify_status_changed()
    await asyncio.to_thread(prune_old_data, reveal_id)
    return metadata


async def _stream_to_file(response: aiohttp.ClientResponse, dest: Path, limit: int) -> int:
    written = 0
    with dest.open("wb") as fh:
        async for chunk in response.content.iter_chunked(1 << 20):
            written += len(chunk)
            if written > limit:
                raise RevealRejected(f"The file is larger than the {limit / 1048576:.0f} MB limit.")
            fh.write(chunk)
    return written


async def fetch_attachment(attachment: discord.Attachment, dest: Path, limit: int = MAX_UPLOAD_BYTES) -> None:
    timeout = aiohttp.ClientTimeout(total=900, connect=15, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": HTTP_USER_AGENT}) as session:
        async with session.get(attachment.url) as response:
            if response.status != 200:
                raise RevealRejected(f"Discord returned HTTP {response.status} for that attachment.")
            await _stream_to_file(response, dest, limit)


async def save_new_reveal_from_attachment(
    attachment: discord.Attachment,
    upload_limit: int = MAX_UPLOAD_BYTES,
) -> dict:
    ok, kind, ext = is_allowed_media(attachment.filename, attachment.content_type)
    if not ok:
        raise RevealRejected("Please send an image (jpg/png/webp) or a video (mp4/mov/webm/mkv/avi/m4v).")
    if attachment.size > upload_limit:
        raise RevealRejected(
            f"That file is {attachment.size / 1048576:.1f} MB; the limit is {upload_limit / 1048576:.1f} MB."
        )
    tmp = TMP_DIR / f"upload_{secrets.token_hex(8)}{ext}"
    try:
        await fetch_attachment(attachment, tmp, upload_limit)
        return await _save_reveal_file(tmp, attachment.filename, kind, ext)
    finally:
        tmp.unlink(missing_ok=True)


class PublicOnlyResolver(aiohttp.abc.AbstractResolver):
    def __init__(self) -> None:
        self._inner = aiohttp.ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        infos = await self._inner.resolve(host, port, family)
        if not infos or any(not ipaddress.ip_address(i["host"].split("%", 1)[0]).is_global for i in infos):
            raise OSError(f"blocked non-public address for {host}")
        return infos

    async def close(self) -> None:
        await self._inner.close()


def _host_allowed(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip("[]").split("%", 1)[0]).is_global
    except ValueError:
        return True


async def download_media_url(url: str, limit: int = MAX_URL_BYTES) -> tuple[Path, str, str, str]:
    current = url.strip().strip("<>")
    if not URL_RE.fullmatch(current):
        raise RevealRejected("Send one direct http(s) media URL.")

    timeout = aiohttp.ClientTimeout(total=180, connect=15, sock_read=45)
    connector = aiohttp.TCPConnector(limit=2, limit_per_host=1, ttl_dns_cache=30, resolver=PublicOnlyResolver())
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
                    or parsed.port not in (None, 80, 443)
                    or not _host_allowed(parsed.hostname)
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
                    if size > limit:
                        raise RevealRejected(
                            f"That URL points to {size / 1048576:.1f} MB; the limit is {limit / 1048576:.1f} MB."
                        )

                    ext = Path(parsed.path).suffix.lower()
                    if ext not in IMAGE_EXTS | VIDEO_EXTS:
                        ext = mimetypes.guess_extension(content_type) or ""
                    fake_name = f"download{ext or '.bin'}"
                    ok, kind, normalized_ext = is_allowed_media(fake_name, content_type)
                    if not ok:
                        raise RevealRejected("The URL does not point to a supported image or video.")

                    path = TMP_DIR / f"url_{secrets.token_hex(8)}{normalized_ext}"
                    await _stream_to_file(response, path, limit)
                    return path, kind, normalized_ext, Path(parsed.path).name or fake_name
        raise RevealRejected("Too many redirects.")
    except aiohttp.ClientConnectorError as exc:
        if path:
            path.unlink(missing_ok=True)
        if "blocked" in str(exc):
            raise RevealRejected("That media URL is not allowed.") from exc
        raise RevealRejected(f"I couldn't download that media URL ({type(exc).__name__}).") from exc
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        if path:
            path.unlink(missing_ok=True)
        raise RevealRejected(f"I couldn't download that media URL ({type(exc).__name__}).") from exc
    except BaseException:
        if path:
            path.unlink(missing_ok=True)
        raise


async def save_new_reveal_from_url(url: str, upload_limit: int = MAX_UPLOAD_BYTES) -> dict:
    path: Optional[Path] = None
    try:
        path, kind, ext, filename = await download_media_url(url, min(MAX_URL_BYTES, upload_limit))
        return await _save_reveal_file(path, filename, kind, ext)
    finally:
        if path:
            path.unlink(missing_ok=True)


                                                                             
                 
                                                                             

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
        if not active:
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
            if not active:
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


def enforce_cache_limit(protect: Optional[Path] = None) -> None:
    now = time.time()
    files: list[tuple[float, int, Path]] = []
    total = 0
    try:
        for p in CACHE_DIR.glob("*/*"):
            try:
                st = p.stat()
            except OSError:
                continue
            if not p.is_file():
                continue
            if p.name.startswith(".") and now - st.st_mtime > 3600:
                p.unlink(missing_ok=True)
                continue
            if protect is not None and p == protect:
                files.append((st.st_mtime, st.st_size, p))
                total += st.st_size
                continue
            if now - st.st_mtime > CACHE_TTL_SECONDS:
                p.unlink(missing_ok=True)
                continue
            total += st.st_size
            files.append((st.st_mtime, st.st_size, p))
    except OSError:
        return

    if CACHE_MAX_BYTES <= 0 or total <= CACHE_MAX_BYTES:
        return
    files.sort(key=lambda t: t[0])
    for mtime, size, p in files:
        if total <= CACHE_MAX_BYTES:
            break
        if protect is not None and p == protect:
            continue
        p.unlink(missing_ok=True)
        total -= size


def startup_cleanup() -> None:
    try:
        for leftover in TMP_DIR.iterdir():
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
    except OSError:
        pass
    enforce_cache_limit()


                                                                             
                                              
                                                                             

bot: "RevealBot"
HISTORY: "RevealHistoryStore"
CURRENT_REVEAL: Optional[dict]


def _history_to_metadata(record: dict) -> dict:
    filename = record.get("archive_filename") or record.get("filename") or (
        "original.mp4" if record.get("kind") == "video" else "original.jpg"
    )
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTS | VIDEO_EXTS:
        suffix = ".mp4" if record.get("kind") == "video" else ".jpg"
    local_path = REVEALS_DIR / record["reveal_id"] / f"original{suffix}"
    out = dict(record)
    out["path"] = str(local_path)
    out["filename"] = record.get("filename") or filename
    return out


async def hydrate_reveal_history() -> None:
    global CURRENT_REVEAL, PERSISTED_REVEALS
    if not HISTORY.enabled:
        log.warning("REVEAL_ARCHIVE_CHANNEL_ID is not configured; restart persistence is disabled.")
        return
    rows = await HISTORY.records(force_refresh=True)
    PERSISTED_REVEALS = {row["reveal_id"]: _history_to_metadata(row) for row in rows}
    CURRENT_REVEAL = PERSISTED_REVEALS[rows[0]["reveal_id"]] if rows else None
    if CURRENT_REVEAL:
        STATUS.reset(CURRENT_REVEAL["reveal_id"])
        _notify_status_changed()


async def archive_new_reveal(metadata: dict) -> dict:
    global CURRENT_REVEAL
    if not HISTORY.enabled:
        raise RevealHistoryError("REVEAL_ARCHIVE_CHANNEL_ID is required for restart-safe reveal storage.")
    rid = metadata["reveal_id"]
    try:
        record = await HISTORY.archive(metadata)
    except Exception:
        shutil.rmtree(REVEALS_DIR / rid, ignore_errors=True)
        raise
    metadata = {**metadata, **record}
    safe_json_write(REVEALS_DIR / rid / "metadata.json", metadata)
    PERSISTED_REVEALS[rid] = metadata
    CURRENT_REVEAL = metadata
    return metadata


async def ensure_reveal_source(reveal: dict) -> dict:
    source = Path(reveal.get("path", ""))
    if source.is_file() and source.stat().st_size > 0:
        return reveal
    if not HISTORY.enabled:
        raise RuntimeError("Reveal source is not present locally and persistent archive is disabled.")

                                                                                   
                                                                                      
                                                                            
    async with _get_reveal_restore_lock(reveal["reveal_id"]):
        source = Path(reveal.get("path", ""))
        if source.is_file() and source.stat().st_size > 0:
            return reveal

        filename = reveal.get("archive_filename") or reveal.get("filename") or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTS | VIDEO_EXTS:
            suffix = ".mp4" if reveal.get("kind") == "video" else ".jpg"
        destination = REVEALS_DIR / reveal["reveal_id"] / f"original{suffix}"
        await HISTORY.ensure_local_copy(reveal, destination, max_bytes=MAX_UPLOAD_BYTES)

        restored = dict(reveal)
        restored["path"] = str(destination)
        restored["filename"] = reveal.get("filename") or destination.name
        restored["kind"] = reveal.get("kind") or ("video" if suffix in VIDEO_EXTS else "image")
        safe_json_write(destination.parent / "metadata.json", restored)
        PERSISTED_REVEALS[restored["reveal_id"]] = restored
        return restored



async def cleanup_deleted_reveal(reveal_id: str) -> None:
    global CURRENT_REVEAL
    PERSISTED_REVEALS.pop(reveal_id, None)
    shutil.rmtree(REVEALS_DIR / reveal_id, ignore_errors=True)
    shutil.rmtree(CACHE_DIR / reveal_id, ignore_errors=True)
    if CURRENT_REVEAL and CURRENT_REVEAL.get("reveal_id") == reveal_id:
        latest = await HISTORY.latest()
        CURRENT_REVEAL = _history_to_metadata(latest) if latest else None
        if CURRENT_REVEAL:
            PERSISTED_REVEALS[CURRENT_REVEAL["reveal_id"]] = CURRENT_REVEAL


async def before_delete_reveal(reveal_id: str) -> None:
    with ACTIVE_REVEALS_LOCK:
        if reveal_id in ACTIVE_REVEALS:
            raise RevealHistoryError("That reveal is being used right now. Try again after the operation finishes.")


                                                                             
                      
                                                                             

async def probe_profile(path: Path) -> Optional[tuple[int, float]]:
    """Read a delivered video's exact height/FPS without loading it into the bot process."""
    try:
        async with process_sem():
            info = await run_worker(
                "video_info",
                60,
                path=str(path),
                max_height=MAX_VIDEO_HEIGHT,
            )
        return int(info["height"]), round(float(info["fps"]), 3)
    except Exception as exc:
        log.info("Could not read delivery profile %s: %s", path, exc)
        return None


                                                                             
                     
                                                                             

def cache_path_for(reveal: dict, user_id: int, target_bytes: Optional[int]) -> Path:
    base = CACHE_DIR / reveal["reveal_id"]
    target_token = str(int(target_bytes or 0))
    if reveal["kind"] == "video":
        return base / f"user_{user_id}_{VIDEO_CACHE_VERSION}_{target_token}b.mp4"
    return base / f"user_{user_id}_{IMAGE_CACHE_VERSION}_{target_token}b.jpg"


def _ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _cleanup_tmp_family(tmp: Path) -> None:
    try:
        for p in tmp.parent.glob(tmp.stem + "*"):
            p.unlink(missing_ok=True)
    except OSError:
        pass


async def build_personalized_reveal(
    reveal: dict,
    user_id: int,
    *,
    video_target_bytes: Optional[int] = None,
    progress_callback=None,
    track_status: bool = True,
) -> tuple[Path, Optional[dict]]:
    reveal = await ensure_reveal_source(reveal)
    if track_status:
        STATUS.request(int(user_id))
        _notify_status_changed()
    _active_reveal(reveal["reveal_id"], +1)
    try:
        source = Path(reveal["path"])
        if not source.is_file():
            raise RuntimeError("There is currently no valid reveal.")
        output = cache_path_for(reveal, user_id, video_target_bytes)
        if _ready(output):
            try:
                output.touch()
            except OSError:
                pass
            if track_status:
                STATUS.success(int(user_id))
                _notify_status_changed()
            return output, None

        ticket = await _queue_add(reveal, user_id)
        try:
            async with _get_user_lock(user_id):
                if _ready(output):
                    try:
                        output.touch()
                    except OSError:
                        pass
                    if track_status:
                        STATUS.success(int(user_id))
                        _notify_status_changed()
                    return output, None

                await _wait_for_build_slot(ticket, progress_callback)
                try:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    tmp = output.with_name(f".{output.stem}.tmp{secrets.token_hex(4)}{output.suffix}")
                    started = time.perf_counter()
                    try:
                        if reveal["kind"] == "image":
                            info = await run_worker(
                                "image", IMAGE_BUILD_TIMEOUT,
                                src=str(source), dst=str(tmp), user_id=user_id,
                                reveal_id=reveal["reveal_id"], amp=float(os.getenv("WM_IMAGE_AMP", "3.0")),
                                target_bytes=video_target_bytes,
                            )
                        else:
                            info = await run_worker(
                                "video", VIDEO_BUILD_TIMEOUT,
                                src=str(source), dst=str(tmp), user_id=user_id,
                                reveal_id=reveal["reveal_id"], amp=float(os.getenv("WM_VIDEO_AMP", "3.0")),
                                preset=VIDEO_PRESET, max_seconds=MAX_VIDEO_SECONDS,
                                max_height=MAX_VIDEO_HEIGHT, target_bytes=video_target_bytes,
                                audio_kbps=VIDEO_AUDIO_KBPS,
                            )
                        if not _ready(tmp):
                            raise RuntimeError("The watermark encoder produced an empty file.")
                        tmp.replace(output)
                    finally:
                        _cleanup_tmp_family(tmp)
                    info["seconds"] = round(time.perf_counter() - started, 1)
                    _record_build_time(reveal["kind"], info["seconds"])
                finally:
                    process_sem().release()
        finally:
            await _queue_remove(ticket)

        log.info("Built %s reveal=%s user=%s %s", reveal["kind"], reveal["reveal_id"], user_id, info)
        await asyncio.to_thread(enforce_cache_limit, output)
        if track_status:
            STATUS.success(int(user_id))
            _notify_status_changed()
        return output, info
    finally:
        _active_reveal(reveal["reveal_id"], -1)


                                                                             
          
                                                                             

def _components_v2_available() -> bool:
    return (
        _V2_OK
        and hasattr(discord.ui, "LayoutView")
        and hasattr(discord.ui, "MediaGallery")
        and hasattr(discord.ui, "TextDisplay")
        and hasattr(discord, "MediaGalleryItem")
    )


def _build_media_gallery_view(filename: str, content: str):
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


async def _fail_response(interaction: discord.Interaction, text: str) -> None:
    try:
        await interaction.edit_original_response(content=text, attachments=[], view=None)
    except discord.HTTPException:
        log.exception("Could not report reveal failure to user=%s", interaction.user.id)


async def _edit_with_file(
    interaction: discord.Interaction, path: Path, filename: str, content: str, *, v2: bool
) -> None:
    reveal_file = discord.File(path, filename=filename, spoiler=False)
    try:
        if v2:
            await interaction.edit_original_response(
                content=None,
                attachments=[reveal_file],
                view=_build_media_gallery_view(filename, content),
            )
        else:
            await interaction.edit_original_response(content=content, attachments=[reveal_file], view=None)
    finally:
        reveal_file.close()


async def send_personalized_reveal(interaction: discord.Interaction) -> None:
    global _V2_OK

    if not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message("I couldn't verify your server membership.", ephemeral=True)
        return
    if not member_is_booster(interaction.user):
        await interaction.response.send_message(
            "❌ This reveal is available to current server boosters only.", ephemeral=True
        )
        return

    reveal = CURRENT_REVEAL
    if not reveal:
        await interaction.response.send_message("There isn't a reveal uploaded right now.", ephemeral=True)
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

    last_progress_edit = 0.0

    async def reveal_progress(percent: int, position: int = 1, estimated_wait: float = 0.0) -> None:
        nonlocal last_progress_edit
        now = time.perf_counter()
        if percent < 90 and now - last_progress_edit < 0.9:
            return
        last_progress_edit = now
        if estimated_wait > 0.5 and int(position) > 1:
            wait_text = f"Queue: {int(position)} • Estimated wait: {max(1, round(estimated_wait))}s"
        elif percent >= 90:
            wait_text = "Generating your personalized reveal now"
        else:
            wait_text = "Joining the reveal queue"
        await interaction.edit_original_response(
            content=f"Reveal loading: {int(percent)}% • {wait_text}",
            attachments=[],
            view=None,
        )

    try:
        path, build_info = await build_personalized_reveal(
            reveal, interaction.user.id, video_target_bytes=target, progress_callback=reveal_progress
        )
    except Exception:
        log.exception("Failed to build reveal for user=%s", interaction.user.id)
        await _fail_response(interaction, "❌ I couldn't generate your personalized reveal. Please try again in a moment.")
        return

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes <= 0:
        await _fail_response(interaction, "❌ I couldn't generate your personalized reveal. Please try again in a moment.")
        return
    if size_bytes > upload_limit:
        await _fail_response(
            interaction,
            f"❌ The personalized file is too large to send here ({size_bytes / 1048576:.1f} MB vs a {upload_limit / 1048576:.1f} MB limit).",
        )
        return

    try:
        path.touch()
    except OSError:
        pass

    filename = "reveal.mp4" if reveal["kind"] == "video" else "reveal.jpg"
    content = (
        "Do NOT share this video to anyone else, doing so could result in moderation."
        if reveal["kind"] == "video"
        else "Do NOT share this image to anyone else, doing so could result in moderation."
    )
    delivered = False

    if _components_v2_available():
        try:
            await _edit_with_file(interaction, path, filename, content, v2=True)
            delivered = True
        except (discord.HTTPException, TypeError, AttributeError, OSError) as exc:
            if isinstance(exc, discord.HTTPException) and exc.status == 400:
                _V2_OK = False
            log.exception("Components V2 delivery failed; falling back to attachment")

    if not delivered:
        try:
            await _edit_with_file(interaction, path, filename, content, v2=False)
            delivered = True
        except (discord.HTTPException, OSError, TypeError):
            log.exception("edit_original_response failed")

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
            log.exception("followup.send failed")

    if not delivered:
        await _fail_response(interaction, "❌ I couldn't deliver your personalized reveal. Please try again.")
        return

                                                                                   
    try:
        profile: Optional[tuple[int, float]] = None
        if reveal["kind"] == "video":
            if build_info and build_info.get("height") and build_info.get("fps"):
                profile = (int(build_info["height"]), round(float(build_info["fps"]), 3))
            else:
                record = PERSISTED_REVEALS.get(reveal["reveal_id"], {})
                known = {int(uid) for uid in record.get("served_user_ids", [])}
                if interaction.user.id not in known:
                    profile = await probe_profile(path)
        await HISTORY.record_delivery(reveal["reveal_id"], interaction.user.id, profile)
        refreshed = await HISTORY.get(reveal["reveal_id"])
        if refreshed:
            PERSISTED_REVEALS[reveal["reveal_id"]] = _history_to_metadata(refreshed)
    except Exception:
        log.exception("Could not persist delivery history for user=%s", interaction.user.id)

    log.info(
        "Delivered reveal=%s user=%s kind=%s size=%d cached=%s elapsed=%.1fs",
        reveal["reveal_id"], interaction.user.id, reveal["kind"],
        size_bytes, build_info is None, time.perf_counter() - started,
    )


class RevealButtonView(discord.ui.View):
    def __init__(self) -> None:
        super().__init__(timeout=None)

    @discord.ui.button(
        label="View Reveal",
        style=discord.ButtonStyle.primary,
        emoji="👁️",
        custom_id="revealbot:view-reveal:v1",
    )
    async def view_reveal_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        await send_personalized_reveal(interaction)


                                                                             
     
                                                                             

class RevealBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents(guilds=True)
        super().__init__(
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
            max_messages=0,
            member_cache_flags=discord.MemberCacheFlags.none(),
            chunk_guilds_at_startup=False,
        )
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        await asyncio.to_thread(startup_cleanup)
        self.add_view(RevealButtonView())
        self.add_view(ArchiveServedUsersView(HISTORY, STAFF_TEAM_ROLE_ID))
        if SYNC_COMMANDS:
            if GUILD_ID:
                guild = discord.Object(id=int(GUILD_ID))
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
            else:
                synced = await self.tree.sync()
            log.info("Synced %d command(s).", len(synced))


bot = RevealBot()
HISTORY = RevealHistoryStore(bot, REVEAL_ARCHIVE_CHANNEL_ID, STAFF_TEAM_ROLE_ID)
_history_hydrated = False


@bot.event
async def on_ready():
    global _history_hydrated
    bot_ready.set()
    if not _history_hydrated:
        try:
            await hydrate_reveal_history()
            _history_hydrated = True
            log.info("Hydrated %d persisted reveal(s).", len(PERSISTED_REVEALS))
        except Exception:
            log.exception("Could not hydrate reveal history")
    log.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "?")


@bot.event
async def on_disconnect():
    bot_ready.clear()


@bot.event
async def on_resumed():
    bot_ready.set()


_STATUS_TASKS: dict[int, asyncio.Task] = {}


def _status_embed() -> discord.Embed:
    snapshot = STATUS.snapshot()
    waiting = sum(1 for ticket in _REVEAL_QUEUE if ticket.state == "queued")
    running = sum(1 for ticket in _REVEAL_QUEUE if ticket.state == "running")
    kind = CURRENT_REVEAL["kind"] if CURRENT_REVEAL else "video"
    build_estimate = _estimated_build_seconds(kind)
    queue_wait = max(0.0, float(waiting) * build_estimate)
    status_name = snapshot.bot_state(waiting, running, queue_wait, bot_ready.is_set())
    total_active = snapshot.successful_users + waiting + running
    if total_active <= 0:
        progress = "0\n[" + "░" * 20 + "] 0%"
    else:
        ratio = min(1.0, snapshot.successful_users / total_active)
        filled = min(20, int(ratio * 20))
        progress = (
            f"{snapshot.successful_users} / {total_active} completed\n"
            f"[{'█' * filled}{'░' * (20 - filled)}] {ratio * 100:.0f}%"
        )

    embed = discord.Embed(title="Bot Status")
    embed.add_field(name="Progress", value=progress, inline=False)
    embed.add_field(name="Generating", value=str(running), inline=True)
    embed.add_field(name="Waiting", value=str(waiting), inline=True)
    embed.add_field(
        name="Estimated wait",
        value=f"~{queue_wait:.0f}s" if waiting else "0s",
        inline=True,
    )
    embed.add_field(name="Status", value=status_name, inline=True)
    embed.add_field(name="Requested", value=str(snapshot.requested_users), inline=True)
    embed.set_footer(text="Live queue dashboard")
    return embed


async def _status_watch(interaction: discord.Interaction) -> None:
    task = asyncio.current_task()
    last_revision = -1
    last_edit = 0.0
    try:
        while True:
            current_revision = _STATUS_REVISION
            if current_revision == last_revision:
                try:
                    await asyncio.wait_for(_STATUS_CHANGED.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                continue

            delay = max(0.0, 0.75 - (time.monotonic() - last_edit))
            if delay:
                await asyncio.sleep(delay)

            try:
                current_revision = _STATUS_REVISION
                await interaction.edit_original_response(embed=_status_embed())
                last_revision = current_revision
                last_edit = time.monotonic()
            except discord.NotFound:
                return
            except discord.HTTPException as exc:
                if exc.status in {401, 404}:
                    return
                await asyncio.sleep(1.5)
                continue

            if _STATUS_REVISION == last_revision:
                _STATUS_CHANGED.clear()
    except asyncio.CancelledError:
        raise
    finally:
        if task is not None and _STATUS_TASKS.get(interaction.user.id) is task:
            _STATUS_TASKS.pop(interaction.user.id, None)


@bot.tree.command(name="status", description="Staff: show live reveal generation status.")
@app_commands.guild_only()
@app_commands.check(is_staff)
async def status(interaction: discord.Interaction):
    old_task = _STATUS_TASKS.get(interaction.user.id)
    if old_task is not None and not old_task.done():
        old_task.cancel()
    await interaction.response.send_message(embed=_status_embed(), ephemeral=True)
    _STATUS_CHANGED.set()
    task = asyncio.create_task(_status_watch(interaction))
    _STATUS_TASKS[interaction.user.id] = task


                                                                             
                          
                                                                             

@bot.tree.command(name="upload", description="Admin: upload an image/video reveal.")
@app_commands.guild_only()
@app_commands.check(is_staff)
@app_commands.describe(
    file="The reveal image/video",
    url="A direct http(s) media URL (use this instead of file)",
)
async def upload(
    interaction: discord.Interaction,
    file: Optional[discord.Attachment] = None,
    url: Optional[str] = None,
):
    if (file is None) == (url is None):
        await interaction.response.send_message("Provide exactly one of `file` or `url`.", ephemeral=True)
        return

    upload_limit = min(
        MAX_UPLOAD_BYTES,
        int(getattr(interaction, "filesize_limit", 0) or MAX_UPLOAD_BYTES),
    )
    await interaction.response.defer(ephemeral=True, thinking=True)

    try:
        if file is not None:
            if file.size > upload_limit:
                raise RevealRejected(
                    f"That file is {file.size / 1048576:.1f} MB; this server allows {upload_limit / 1048576:.1f} MB per upload."
                )
            metadata = await save_new_reveal_from_attachment(file, upload_limit)
        else:
            current = (url or "").strip().strip("<>")
            if not URL_RE.fullmatch(current):
                raise RevealRejected("Send one direct http(s) media URL.")
            metadata = await save_new_reveal_from_url(current, upload_limit)

        metadata = await archive_new_reveal(metadata)
        note = ""
        if metadata.get("truncated_to"):
            note = (
                f" The source is {metadata['duration']:.0f}s; viewers receive the first "
                f"{metadata['truncated_to']}s."
            )
        await interaction.followup.send(
            f"✅ Reveal `{metadata['reveal_id']}` is live ({metadata['kind']}). Boosters can use `/view_reveal`." + note,
            ephemeral=True,
        )
    except RevealRejected as exc:
        await interaction.followup.send(f"❌ {exc}", ephemeral=True)
    except RevealHistoryError as exc:
        log.exception("Reveal archive failed")
        await interaction.followup.send(f"❌ {exc}", ephemeral=True)
    except Exception:
        log.exception("Upload failed")
        await interaction.followup.send("❌ I couldn't ingest/archive that media. Check the bot logs.", ephemeral=True)


@bot.tree.command(name="view_reveal", description="Get your booster reveal.")
@app_commands.guild_only()
@app_commands.check(is_booster)
async def view_reveal(interaction: discord.Interaction):
    await send_personalized_reveal(interaction)


async def resolve_member(guild: discord.Guild, user_id: int) -> Optional[discord.Member]:
    try:
        member = guild.get_member(user_id)
        if member:
            return member
    except Exception:
        pass
    try:
        return await guild.fetch_member(user_id)
    except discord.HTTPException:
        return None


@app_commands.context_menu(name="Add Reveal Button")
@app_commands.check(is_staff)
async def add_reveal_button(interaction: discord.Interaction, message: discord.Message):
    if not interaction.guild or not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message("This app can only be used in a server.", ephemeral=True)
        return
    try:
        await message.reply(
            content="Booster reveal - tap the button below to open the reveal.",
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
    await interaction.response.send_message("✅ Posted a View Reveal button.", ephemeral=True)


bot.tree.add_command(add_reveal_button)


                                                                             
        
                                                                             

async def _announce_trace(
    interaction: discord.Interaction,
    reveal: dict,
    result: dict,
    *,
    served: bool,
    checked: int,
    profile: Optional[tuple[int, float]] = None,
) -> None:
    uid = int(result["user_id"])
    member = await resolve_member(interaction.guild, uid)
    frame_text = f" • source frame `{result['frame']}`" if "frame" in result else ""
    served_text = "**yes**" if served else "**no record — treat as unverified, possibly a false match**"
    await interaction.followup.send(
        f"🔎 Watermark decoded (CRC verified): <@{uid}> (`{uid}`)"
        f"{' — in this server' if member else ' — not currently in this server'}\n"
        f"Reveal `{reveal['reveal_id']}`{frame_text} • served to this user: {served_text}",
        ephemeral=True,
    )
    log.warning(
        "TRACE by %s: reveal=%s decoded_user=%s served=%s checked=%d profile=%s",
        interaction.user.id, reveal["reveal_id"], uid, served, checked, profile,
    )


@bot.tree.command(name="trace", description="Admin: trace a leaked reveal to its Discord user.")
@app_commands.guild_only()
@app_commands.check(is_staff)
@app_commands.describe(
    file="Leaked image/video, or a frame from a video reveal",
    reveal_id="Optional reveal ID; omit to search the newest saved reveals",
    start_frame="Optional original frame hint for video tracing",
)
async def trace(
    interaction: discord.Interaction,
    file: discord.Attachment,
    reveal_id: Optional[str] = None,
    start_frame: app_commands.Range[int, 0, 200000] = 0,
):
    if file.size > MAX_UPLOAD_BYTES:
        await interaction.response.send_message(
            f"That leak is {file.size / 1048576:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1048576:.1f} MB.",
            ephemeral=True,
        )
        return

    ok, leak_kind, ext = is_allowed_media(file.filename, file.content_type)
    if not ok:
        await interaction.response.send_message("The leak must be an image or video.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    reveals = find_reveals(reveal_id)
    if not reveals:
        await interaction.followup.send("I couldn't find that reveal's persistent record.", ephemeral=True)
        return
    if reveal_id is None and len(reveals) > TRACE_MAX_REVEALS:
        reveals = reveals[:TRACE_MAX_REVEALS]

    leak_path = TMP_DIR / f"leak_{secrets.token_hex(8)}{ext}"
    try:
        await fetch_attachment(file, leak_path, MAX_UPLOAD_BYTES)
        if leak_kind == "image":
            try:
                lw, lh = await asyncio.to_thread(_image_size, leak_path)
            except Exception:
                await interaction.followup.send("I couldn't read the leak as an image.", ephemeral=True)
                return
            if lw * lh > TRACE_LEAK_MAX_PIXELS:
                await interaction.followup.send(
                    f"That leak is {lw * lh / 1e6:.0f} MP; tracing is limited to {TRACE_LEAK_MAX_PIXELS / 1e6:.0f} MP here. Downscaling it first is fine.",
                    ephemeral=True,
                )
                return

        trace_deadline = time.monotonic() + TRACE_TIME_BUDGET
        checked = 0

        async def attempt(
            reveal: dict,
            delivery_height: Optional[int] = None,
            delivery_fps: Optional[float] = None,
        ) -> Optional[dict]:
            nonlocal checked
            if reveal["kind"] == "image":
                if leak_kind != "image":
                    return None
                trace_kind = "image"
                kind_budget = TRACE_IMAGE_TIME_BUDGET
            else:
                trace_kind = "video_frame" if leak_kind == "image" else "video"
                kind_budget = TRACE_VIDEO_TIME_BUDGET

            remaining = trace_deadline - time.monotonic()
            if remaining <= 0:
                return None
            budget = min(kind_budget, remaining)
            kwargs = {
                "max_height": delivery_height or MAX_VIDEO_HEIGHT,
                "start_frame": int(start_frame),
                "time_budget": budget,
            }
            if delivery_height is not None:
                kwargs["delivery_height"] = int(delivery_height)
            if delivery_fps is not None:
                kwargs["delivery_fps"] = float(delivery_fps)

            checked += 1
            try:
                async with process_sem():
                    result = await run_worker(
                        "extract", budget + 30,
                        orig=reveal["path"],
                        leak=str(leak_path),
                        kind=trace_kind,
                        reveal_id=reveal["reveal_id"],
                        kwargs=kwargs,
                    )
            except Exception as exc:
                log.info(
                    "Reveal %s trace failed h=%s fps=%s: %s",
                    reveal["reveal_id"], delivery_height, delivery_fps, exc,
                )
                return None
            return result if result.get("valid") else None

        for reveal_row in reveals:
            if time.monotonic() >= trace_deadline:
                break
            rid = reveal_row["reveal_id"]
            _active_reveal(rid, +1)
            try:
                try:
                    reveal = await ensure_reveal_source(reveal_row)
                except Exception:
                    log.exception("Could not restore reveal %s for tracing", rid)
                    continue

                persisted = PERSISTED_REVEALS.get(rid, reveal)
                served_ids = {int(uid) for uid in persisted.get("served_user_ids", [])}

                if reveal["kind"] == "video":
                    profiles = [
                        (int(h), round(float(fps), 3))
                        for h, fps in persisted.get("profiles", [])
                    ]
                    if not profiles:
                                                                                         
                                                                                             
                                                      
                        profiles = []

                    for profile in profiles[:4]:
                        if time.monotonic() >= trace_deadline:
                            break
                        result = await attempt(reveal, profile[0], profile[1])
                        if result is None:
                            continue
                        uid = int(result["user_id"])
                        if uid not in served_ids:
                            log.info("Trace decoded unserved user=%s reveal=%s", uid, rid)
                            continue
                        await _announce_trace(
                            interaction, reveal, result, served=True, checked=checked, profile=profile
                        )
                        return

                if time.monotonic() >= trace_deadline:
                    break
                result = await attempt(reveal)
                if result is None:
                    continue
                uid = int(result["user_id"])
                served = uid in served_ids
                await _announce_trace(
                    interaction, reveal, result, served=served, checked=checked
                )
                return
            finally:
                _active_reveal(rid, -1)

        timed_out = time.monotonic() >= trace_deadline
        suffix = " The trace time budget was reached before every candidate could be checked." if timed_out else ""
        await interaction.followup.send(
            f"No valid watermark found across {checked} trace candidate(s). The leak may be too cropped/edited/compressed or belong to a reveal whose original was deleted.{suffix}",
            ephemeral=True,
        )
    except RevealRejected as exc:
        await interaction.followup.send(f"❌ {exc}", ephemeral=True)
    except Exception:
        log.exception("Trace failed")
        await interaction.followup.send("❌ Trace failed. Check the bot logs.", ephemeral=True)
    finally:
        leak_path.unlink(missing_ok=True)


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        message = "You don't have permission to use this command."
    else:
        log.error("Slash command error: %r", error, exc_info=(type(error), error, error.__traceback__))
        message = "Something went wrong while running that command."
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except discord.HTTPException:
        pass


                                                                             
bot.tree.add_command(
    make_history_command(
        HISTORY,
        cleanup_callback=cleanup_deleted_reveal,
        before_delete_callback=before_delete_reveal,
        staff_role_id=STAFF_TEAM_ROLE_ID,
    )
)

install_stress_test(
    bot,
    get_current_reveal=lambda: CURRENT_REVEAL,
    ensure_reveal_source=ensure_reveal_source,
    build_personalized_reveal=build_personalized_reveal,
    compute_video_target_bytes=compute_video_target_bytes,
    run_worker=run_worker,
    process_sem=process_sem,
    get_worker_procs=lambda: list(WORKER_PROCS),
    cache_dir=CACHE_DIR,
    tmp_dir=TMP_DIR,
    max_video_height=MAX_VIDEO_HEIGHT,
    trace_image_budget=TRACE_IMAGE_TIME_BUDGET,
    trace_video_budget=TRACE_VIDEO_TIME_BUDGET,
    cache_path_for=cache_path_for,
    history_store=HISTORY,
    record_test_users=HISTORY.record_stress_test_users,
    staff_role_id=STAFF_TEAM_ROLE_ID,
)


                                                                             
      
                                                                             

bot_ready = threading.Event()


async def main():
    health_server = start_health_server(bot_ready.is_set)
    log.info("Health server listening on 0.0.0.0:%s", health_server.server_port)
    try:
        async with bot:
            await bot.start(TOKEN)
    finally:
        for proc in list(WORKER_PROCS):
            _kill_group(proc)
        health_server.shutdown()
        health_server.server_close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
