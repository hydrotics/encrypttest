import asyncio
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
import mimetypes

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv
from PIL import Image

import watermark as wm

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
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "veryfast")
VIDEO_AUDIO_KBPS = int(os.getenv("VIDEO_AUDIO_KBPS", "96"))
VIDEO_TARGET_MAX_MB = float(os.getenv("VIDEO_TARGET_MAX_MB", "19.5"))
VIDEO_MAX_BPP = float(os.getenv("VIDEO_MAX_BPP", str(wm.MAX_BITS_PER_PIXEL)))
VIDEO_CACHE_VERSION = f"{os.getenv('VIDEO_CACHE_VERSION', 'v15')}-h264-level-auto-v1"
IMAGE_AMP = float(os.getenv("WM_IMAGE_AMP", "3.0"))
IMAGE_CELL = int(os.getenv("WM_IMAGE_CELL", "4"))
VIDEO_AMP = float(os.getenv("WM_VIDEO_AMP", "3.0"))
VIDEO_CELL = int(os.getenv("WM_VIDEO_CELL", "8"))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "250")) * 1048576)
MAX_IMAGE_PIXELS = int(float(os.getenv("MAX_IMAGE_MEGAPIXELS", "100")) * 1_000_000)
UPLOAD_WAIT_SECONDS = int(os.getenv("UPLOAD_WAIT_SECONDS", "90"))
MAX_URL_BYTES = min(MAX_UPLOAD_BYTES, int(float(os.getenv("MAX_MEDIA_URL_MB", "250")) * 1048576))
AUTO_DELETE_OLD_ORIGINALS = os.getenv("AUTO_DELETE_OLD_ORIGINALS", "false").lower() == "true"

IMAGE_PREVIEW_MAX_DIM = int(os.getenv("IMAGE_PREVIEW_MAX_DIM", "2048"))
IMAGE_PREVIEW_QUALITY = int(os.getenv("IMAGE_PREVIEW_QUALITY", "92"))

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("revealbot")

PROCESS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
USER_LOCKS: dict[tuple[str, int], asyncio.Lock] = defaultdict(asyncio.Lock)
PENDING_UPLOADS: dict[tuple[int, int, int], float] = {}


class RevealRejected(Exception):
    pass


def safe_json_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _original_in(directory: Path) -> Optional[Path]:
    for f in directory.glob("original.*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            return f
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
    out = []
    for directory in REVEALS_DIR.iterdir():
        if not directory.is_dir() or not REVEAL_ID_RE.fullmatch(directory.name):
            continue
        row = _metadata_for(directory)
        if row and Path(row["path"]).exists():
            out.append(row)
    return sorted(out, key=lambda r: int(r.get("created_at", 0)), reverse=True)


def load_current_reveal() -> Optional[dict]:
    try:
        if META_FILE.exists():
            row = json.loads(META_FILE.read_text(encoding="utf-8"))
            if Path(row.get("path", "")).exists():
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
        return [row] if row and Path(row["path"]).exists() else []
    return all_reveals()


def member_is_booster(member: discord.Member) -> bool:
    return bool(BOOSTER_ROLE_ID and any(r.id == BOOSTER_ROLE_ID for r in member.roles)) or member.premium_since is not None


def ledger_append(reveal_id: str, user_id: int) -> None:
    try:
        with LEDGER_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": int(time.time()), "reveal_id": reveal_id, "user_id": int(user_id)}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        log.exception("Could not write ledger.")


def ledger_served(reveal_id: str, user_id: int) -> bool:
    if not LEDGER_FILE.exists():
        return False
    try:
        with LEDGER_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("reveal_id") == reveal_id and int(row.get("user_id", -1)) == int(user_id):
                    return True
    except OSError:
        log.exception("Could not read ledger.")
    return False


def compute_video_target_bytes(upload_limit: Optional[int]) -> int:
    configured = int(VIDEO_TARGET_MAX_MB * 1048576)
    if not upload_limit:
        return configured
    margin = min(512 * 1024, max(128 * 1024, int(upload_limit * 0.025)))
    return min(configured, max(4 * 1048576, int(upload_limit) - margin))


def is_allowed_media(filename: str, content_type: Optional[str]) -> tuple[bool, Optional[str], Optional[str]]:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS or (content_type and content_type.startswith("image/") and content_type != "image/gif"):
        return True, "image", ext if ext in IMAGE_EXTS else ".png"
    if ext in VIDEO_EXTS or (content_type and content_type.startswith("video/")):
        return True, "video", ext if ext in VIDEO_EXTS else ".mp4"
    return False, None, None


def validate_media(path: Path, kind: str) -> dict:
    if kind == "image":
        try:
            with Image.open(path) as probe:
                width, height = probe.size
            if width * height > MAX_IMAGE_PIXELS:
                raise RevealRejected(f"That image is {width * height / 1e6:.0f} MP; the limit is {MAX_IMAGE_PIXELS / 1e6:.0f} MP.")
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
        for key in list(USER_LOCKS):
            if key[0] != reveal_id:
                USER_LOCKS.pop(key, None)
        return metadata
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def save_new_reveal_from_attachment(attachment: discord.Attachment) -> dict:
    ok, kind, ext = is_allowed_media(attachment.filename, attachment.content_type)
    if not ok:
        raise RevealRejected("Please send an image (jpg/png/webp) or a video (mp4/mov/webm/mkv/avi/m4v).")
    if attachment.size > MAX_UPLOAD_BYTES:
        raise RevealRejected(f"That file is {attachment.size / 1048576:.0f} MB; the limit is {MAX_UPLOAD_BYTES / 1048576:.0f} MB.")
    tmp = TMP_DIR / f"upload_{secrets.token_hex(8)}{ext}"
    try:
        await attachment.save(tmp, use_cached=False)
        return await _save_reveal_file(tmp, attachment.filename, kind, ext)
    finally:
        tmp.unlink(missing_ok=True)


def _public_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        for item in infos:
            ip = ipaddress.ip_address(item[4][0].split("%", 1)[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                return False
        return True
    except (socket.gaierror, ValueError):
        return False


async def download_media_url(url: str) -> tuple[Path, str, str, str]:
    current = url.strip().strip("<>")
    if not URL_RE.fullmatch(current):
        raise RevealRejected("Send one direct http(s) media URL.")
    timeout = aiohttp.ClientTimeout(total=180, connect=15, sock_read=45)
    path: Optional[Path] = None
    try:
        async with aiohttp.ClientSession(timeout=timeout, raise_for_status=False) as session:
            for _ in range(5):
                parsed = urlparse(current)
                if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or not _public_ip(parsed.hostname):
                    raise RevealRejected("That media URL is not allowed.")
                async with session.get(current, allow_redirects=False) as response:
                    if 300 <= response.status < 400 and response.headers.get("Location"):
                        current = urljoin(current, response.headers["Location"])
                        continue
                    if response.status != 200:
                        raise RevealRejected(f"The media URL returned HTTP {response.status}.")
                    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
                    size = int(response.headers.get("Content-Length", "0") or 0)
                    if size > MAX_URL_BYTES:
                        raise RevealRejected(f"That URL points to {size / 1048576:.0f} MB; the limit is {MAX_URL_BYTES / 1048576:.0f} MB.")
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
    raise RevealRejected("Too many redirects.")


async def save_new_reveal_from_url(url: str) -> dict:
    path = None
    try:
        path, kind, ext, filename = await download_media_url(url)
        return await _save_reveal_file(path, filename, kind, ext)
    finally:
        if path:
            path.unlink(missing_ok=True)


def prune_old_data(keep_reveal_id: str) -> None:
    for child in CACHE_DIR.iterdir():
        if child.is_dir() and child.name != keep_reveal_id:
            shutil.rmtree(child, ignore_errors=True)
    if AUTO_DELETE_OLD_ORIGINALS:
        for directory in REVEALS_DIR.iterdir():
            if directory.is_dir() and directory.name != keep_reveal_id:
                shutil.rmtree(directory, ignore_errors=True)
    cutoff = time.time() - 3600
    for leftover in TMP_DIR.iterdir():
        try:
            if leftover.is_file() and leftover.stat().st_mtime < cutoff:
                leftover.unlink(missing_ok=True)
        except OSError:
            pass


def _build_sync(source: Path, kind: str, output: Path, user_id: int, reveal_id: str,
                target_bytes: Optional[int]) -> dict:
    started = time.perf_counter()
    if kind == "image":
        wm.render_image_preview(
            source, output, user_id, WM_KEY, reveal_id,
            amp=IMAGE_AMP, max_dim=IMAGE_PREVIEW_MAX_DIM, quality=IMAGE_PREVIEW_QUALITY,
        )
        info = {}
    else:
        info = wm.embed_video(
            source, output, user_id, WM_KEY, reveal_id,
            amp=VIDEO_AMP, preset=VIDEO_PRESET, max_seconds=MAX_VIDEO_SECONDS,
            max_height=MAX_VIDEO_HEIGHT, target_bytes=target_bytes,
            audio_kbps=VIDEO_AUDIO_KBPS, max_bpp=VIDEO_MAX_BPP,
        )
    info["seconds"] = round(time.perf_counter() - started, 1)
    return info


def cache_path_for(reveal: dict, user_id: int, target_bytes: Optional[int]) -> Path:
    base = CACHE_DIR / reveal["reveal_id"]
    if reveal["kind"] == "video":
        target_mb = int(round((target_bytes or 0) / 1048576))
        return base / f"user_{user_id}_{VIDEO_CACHE_VERSION}_{target_mb}mb.mp4"
    return base / f"user_{user_id}_mobile.jpg"


def _ready(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


async def build_personalized_reveal(reveal: dict, user_id: int, *, video_target_bytes: Optional[int] = None) -> tuple[Path, bool]:
    source = Path(reveal["path"])
    if not source.exists():
        raise RuntimeError("There is currently no valid reveal.")
    output = cache_path_for(reveal, user_id, video_target_bytes)
    if _ready(output):
        return output, True
    async with USER_LOCKS[(reveal["reveal_id"], user_id)]:
        if _ready(output):
            return output, True
        async with PROCESS_SEMAPHORE:
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_name(f"{output.stem}.tmp{output.suffix}")
            try:
                info = await asyncio.to_thread(
                    _build_sync, source, reveal["kind"], tmp, user_id,
                    reveal["reveal_id"], video_target_bytes
                )
                tmp.replace(output)
            finally:
                tmp.unlink(missing_ok=True)
    log.info("Built %s reveal=%s user=%s %s", reveal["kind"], reveal["reveal_id"], user_id, info)
    return output, False




async def send_personalized_reveal(interaction: discord.Interaction) -> None:
    """Generate one personalized file and deliver it as one ephemeral Discord attachment."""
    if not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message(
            "I couldn't verify your server membership.", ephemeral=True
        )
        return

    if not member_is_booster(interaction.user):
        await interaction.response.send_message(
            "❌ This reveal is available to current server boosters only.",
            ephemeral=True,
        )
        return

    reveal = CURRENT_REVEAL
    if not reveal or not Path(reveal["path"]).exists():
        await interaction.response.send_message(
            "There isn't a reveal uploaded right now.", ephemeral=True
        )
        return

    upload_limit = getattr(interaction, "filesize_limit", None)
    if upload_limit is None and interaction.guild is not None:
        upload_limit = getattr(interaction.guild, "filesize_limit", None)
    target = compute_video_target_bytes(upload_limit) if reveal["kind"] == "video" else None

    # Discord interactions must be acknowledged quickly. Defer once, do the build,
    # then replace that same ephemeral response with the actual attachment.
    await interaction.response.defer(ephemeral=True, thinking=True)
    started = time.perf_counter()
    try:
        path, cached = await build_personalized_reveal(
            reveal, interaction.user.id, video_target_bytes=target
        )
    except Exception as exc:
        log.exception("Failed to build reveal for user=%s", interaction.user.id)
        await interaction.edit_original_response(
            content=f"❌ I couldn't generate your personalized reveal: {exc}",
            attachments=[],
        )
        return

    size_bytes = path.stat().st_size
    if reveal["kind"] == "video" and upload_limit and size_bytes > upload_limit:
        await interaction.edit_original_response(
            content=(
                f"❌ The personalized video is too large ({size_bytes / 1048576:.1f} MB vs "
                f"the {upload_limit / 1048576:.1f} MB attachment limit)."
            ),
            attachments=[],
        )
        return

    filename = "reveal.mp4" if reveal["kind"] == "video" else "reveal.jpg"
    content = (
        "🎬 Booster reveal attached below."
        if reveal["kind"] == "video"
        else "🖼️ Booster reveal attached below."
    )

    log.info(
        "Prepared reveal=%s user=%s kind=%s size=%d cached=%s build_seconds=%.2f",
        reveal["reveal_id"], interaction.user.id, reveal["kind"], size_bytes,
        cached, time.perf_counter() - started,
    )

    try:
        await interaction.edit_original_response(
            content=content,
            attachments=[discord.File(path, filename=filename, spoiler=False)],
        )
    except Exception:
        log.exception(
            "Failed to deliver reveal attachment reveal=%s user=%s",
            reveal["reveal_id"], interaction.user.id,
        )
        try:
            await interaction.edit_original_response(
                content="❌ I couldn't deliver your personalized reveal. Please try again.",
                attachments=[],
            )
        except discord.HTTPException:
            log.exception("Could not update failed reveal response for user=%s", interaction.user.id)
        return

    await asyncio.to_thread(ledger_append, reveal["reveal_id"], interaction.user.id)


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
        intents = discord.Intents(guilds=True, messages=True, message_content=True)
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # The button has a fixed custom_id and no timeout, so it keeps working after restarts.
        self.add_view(RevealButtonView())
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
        else:
            synced = await self.tree.sync()
        log.info("Synced %d command(s).", len(synced))

    async def on_message(self, message: discord.Message) -> None:
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
                await message.channel.send("❌ Upload one file at a time.", delete_after=8)
                return
            if message.attachments:
                metadata = await save_new_reveal_from_attachment(message.attachments[0])
            else:
                match = URL_RE.fullmatch(message.content.strip())
                if not match:
                    await message.delete()
                    await message.channel.send("❌ Send one direct media URL or one attached file.", delete_after=8)
                    return
                metadata = await save_new_reveal_from_url(match.group(0))

            try:
                await message.delete()
            except discord.HTTPException:
                log.warning("Could not delete the admin's upload message; grant Manage Messages to keep originals hidden.")

            note = ""
            if metadata.get("truncated_to"):
                note = f"\n⚠️ The source is {metadata['duration']:.0f}s; viewers receive the first {metadata['truncated_to']}s."
            await message.channel.send(
                f"✅ Reveal `{metadata['reveal_id']}` is live ({metadata['kind']}). Boosters can use `/view_reveal`.{note}",
                delete_after=12,
            )
        except RevealRejected as exc:
            try:
                await message.delete()
            except discord.HTTPException:
                pass
            await message.channel.send(f"❌ {exc}", delete_after=10)
        except Exception:
            log.exception("Post-/upload ingestion failed.")
            try:
                await message.delete()
            except discord.HTTPException:
                pass
            await message.channel.send("❌ I couldn't ingest that media. Check the bot logs.", delete_after=10)


bot = RevealBot()


@bot.event
async def on_ready():
    log.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "?")


@bot.tree.command(name="upload", description="Admin: start an image/video reveal upload.")
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
async def upload(interaction: discord.Interaction):
    key = (interaction.guild_id or 0, interaction.channel_id or 0, interaction.user.id)
    PENDING_UPLOADS[key] = time.monotonic() + UPLOAD_WAIT_SECONDS
    await interaction.response.send_message(
        f"Send the reveal as the next message in this channel: attach the file or paste one direct media URL. "
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
        await interaction.response.send_message("This app can only be used in a server.", ephemeral=True)
        return
    if not interaction.user.guild_permissions.manage_guild:
        await interaction.response.send_message("You need the Manage Server permission to add reveal buttons.", ephemeral=True)
        return
    try:
        await message.reply(
            content="👁️ Booster reveal — tap the button below to open your private reveal.",
            view=RevealButtonView(),
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
    except discord.HTTPException as exc:
        log.exception("Could not post reveal button reply to message %s", message.id)
        await interaction.response.send_message(
            "I couldn't reply with the reveal button. Check that I can view the channel and send messages there.",
            ephemeral=True,
        )
        return
    await interaction.response.send_message(
        "✅ Posted a View Reveal button as a reply to the selected message.",
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
        await interaction.response.send_message("I couldn't find that reveal's original file.", ephemeral=True)
        return

    ok, leak_kind, ext = is_allowed_media(file.filename, file.content_type)
    if not ok:
        await interaction.response.send_message("The leak must be an image or video.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    leak_path = TMP_DIR / f"leak_{secrets.token_hex(8)}{ext}"
    try:
        await file.save(leak_path, use_cached=False)
        checked = 0
        for reveal in reveals:
            if reveal["kind"] == "image" and leak_kind != "image":
                continue
            if reveal["kind"] == "video" and leak_kind not in {"image", "video"}:
                continue
            checked += 1
            try:
                trace_kind = "video_frame" if reveal["kind"] == "video" and leak_kind == "image" else leak_kind
                async with PROCESS_SEMAPHORE:
                    result = await asyncio.to_thread(
                        wm.extract,
                        Path(reveal["path"]), leak_path, trace_kind, WM_KEY, reveal["reveal_id"],
                        image_cell=IMAGE_CELL, video_cell=VIDEO_CELL,
                        max_height=MAX_VIDEO_HEIGHT, start_frame=int(start_frame),
                    )
            except Exception as exc:
                log.info("Reveal %s did not decode: %s", reveal["reveal_id"], exc)
                continue
            if not result.get("valid"):
                continue

            uid = int(result["user_id"])
            served = await asyncio.to_thread(ledger_served, reveal["reveal_id"], uid)
            member = await resolve_member(interaction.guild, uid)
            frame_text = f" • source frame `{result['frame']}`" if "frame" in result else ""
            await interaction.followup.send(
                f"🔎 Watermark decoded (CRC verified): <@{uid}> (`{uid}`)"
                f"{' — in this server' if member else ' — not currently in this server'}\n"
                f"Reveal `{reveal['reveal_id']}`{frame_text} • served to this user: "
                f"{'**yes**' if served else '**no record**'}",
                ephemeral=True,
            )
            log.warning(
                "TRACE by %s: reveal=%s decoded_user=%s checked=%d",
                interaction.user.id, reveal["reveal_id"], uid, checked,
            )
            return

        await interaction.followup.send(
            f"No valid watermark found across {checked} saved reveal(s). "
            f"The leak may be too cropped/edited/compressed or belong to a reveal whose original was deleted.",
            ephemeral=True,
        )
    except Exception:
        log.exception("Trace failed.")
        await interaction.followup.send("❌ Trace failed. Check the bot logs.", ephemeral=True)
    finally:
        leak_path.unlink(missing_ok=True)


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        message = "You need the **Manage Server** permission to use this command."
    else:
        log.exception("Slash command error", exc_info=error)
        message = "Something went wrong while running that command."
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except discord.HTTPException:
        pass


async def main():
    async with bot:
        await bot.start(TOKEN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
