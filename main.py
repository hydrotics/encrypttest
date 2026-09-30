import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from aiohttp import web
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from PIL import Image, ImageOps
import watermark as wm

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing from .env/environment variables.")
_SECRET = os.getenv("WATERMARK_SECRET")
if not _SECRET or len(_SECRET) < 16:
    raise RuntimeError("WATERMARK_SECRET is missing/too short (use 32+ random chars).")
WM_KEY = _SECRET.encode("utf-8")

GUILD_ID = os.getenv("GUILD_ID")
BOOSTER_ROLE_ID = int(os.getenv("BOOSTER_ROLE_ID", "0") or 0)
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
PENDING_UPLOAD_TIMEOUT = int(os.getenv("PENDING_UPLOAD_TIMEOUT", "300"))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "300"))
MAX_VIDEO_HEIGHT = int(os.getenv("MAX_VIDEO_HEIGHT", "720"))
VIDEO_CRF = int(os.getenv("VIDEO_CRF", "19"))
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "veryfast")
VIDEO_AUDIO_KBPS = int(os.getenv("VIDEO_AUDIO_KBPS", "96"))
VIDEO_TARGET_MAX_MB = float(os.getenv("VIDEO_TARGET_MAX_MB", "18.0"))
VIDEO_CACHE_VERSION = os.getenv("VIDEO_CACHE_VERSION", "v4")
DELETE_UPLOAD_MESSAGE = os.getenv("DELETE_UPLOAD_MESSAGE", "true").lower() == "true"
IMAGE_AMP = float(os.getenv("WM_IMAGE_AMP", "3.0"))
IMAGE_CELL = int(os.getenv("WM_IMAGE_CELL", "4"))
VIDEO_AMP = float(os.getenv("WM_VIDEO_AMP", "3.0"))
VIDEO_CELL = int(os.getenv("WM_VIDEO_CELL", "8"))
KEEP_ORIGINALS = os.getenv("KEEP_ORIGINALS", "true").lower() == "true"

# Render Web Service / health endpoint.
# Render supplies PORT and RENDER_EXTERNAL_URL automatically for web services.
HTTP_HOST = "0.0.0.0"
HTTP_PORT = int(os.getenv("PORT", "10000"))
PUBLIC_URL = (os.getenv("PUBLIC_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").rstrip("/")

# The Discord client benefits from a smaller, baseline JPEG for the mobile preview.
# The original personalized image remains on disk for the bot's internal reveal/trace workflow.
IMAGE_PREVIEW_MAX_DIM = int(os.getenv("IMAGE_PREVIEW_MAX_DIM", "2048"))
IMAGE_PREVIEW_QUALITY = int(os.getenv("IMAGE_PREVIEW_QUALITY", "88"))

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
REVEALS_DIR = DATA_DIR / "reveals"
CACHE_DIR = DATA_DIR / "cache"
META_FILE = DATA_DIR / "current_reveal.json"
TMP_DIR = DATA_DIR / "tmp"
LEDGER_FILE = DATA_DIR / "served.jsonl"
for directory in (REVEALS_DIR, CACHE_DIR, TMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
REVEAL_ID_RE = re.compile(r"^\d+_[0-9a-f]{8}$")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("revealbot")
PROCESS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
USER_LOCKS: dict[tuple[str, int], asyncio.Lock] = defaultdict(asyncio.Lock)
PENDING_UPLOADS: dict[tuple[int, int, int], float] = {}
PENDING_SWEEP_INTERVAL = 60


def safe_json_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_current_reveal() -> Optional[dict]:
    if not META_FILE.exists():
        return None
    try:
        payload = json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Could not read %s", META_FILE)
        return None
    if not Path(payload.get("path", "")).exists():
        log.warning("Reveal source file is missing: %s", payload.get("path"))
        return None
    return payload


CURRENT_REVEAL = load_current_reveal()


def is_allowed_media(filename: str, content_type: Optional[str]) -> tuple[bool, Optional[str], Optional[str]]:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS or (content_type and content_type.startswith("image/") and content_type != "image/gif"):
        return True, "image", ext if ext in IMAGE_EXTS else ".png"
    if ext in VIDEO_EXTS or (content_type and content_type.startswith("video/")):
        return True, "video", ext if ext in VIDEO_EXTS else ".mp4"
    return False, None, None


def find_reveal(reveal_id: Optional[str]) -> Optional[dict]:
    if not reveal_id:
        return CURRENT_REVEAL
    reveal_dir = REVEALS_DIR / reveal_id
    if not reveal_dir.is_dir():
        return None
    for f in reveal_dir.glob("original.*"):
        kind = "image" if f.suffix.lower() in IMAGE_EXTS else "video"
        return {"reveal_id": reveal_id, "kind": kind, "path": str(f)}
    return None


def member_is_booster(member: discord.Member) -> bool:
    return bool(BOOSTER_ROLE_ID and any(r.id == BOOSTER_ROLE_ID for r in member.roles)) or member.premium_since is not None


def ledger_append(reveal_id: str, user_id: int) -> None:
    try:
        with LEDGER_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": int(time.time()), "reveal_id": reveal_id, "user_id": user_id}) + "\n")
    except OSError:
        log.exception("Could not write ledger.")


def ledger_served(reveal_id: str, user_id: int) -> bool:
    if not LEDGER_FILE.exists():
        return False
    try:
        with LEDGER_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("reveal_id") == reveal_id and row.get("user_id") == user_id:
                    return True
    except OSError:
        log.exception("Could not read ledger.")
    return False


def _build_sync(source: Path, kind: str, output: Path, user_id: int, reveal_id: str,
                target_bytes: Optional[int] = None) -> None:
    if kind == "image":
        wm.embed_image(source, output, user_id, WM_KEY, reveal_id, amp=IMAGE_AMP, cell=IMAGE_CELL)
    else:
        wm.embed_video(
            source, output, user_id, WM_KEY, reveal_id, amp=VIDEO_AMP, cell=VIDEO_CELL,
            crf=VIDEO_CRF, preset=VIDEO_PRESET, max_seconds=MAX_VIDEO_SECONDS,
            max_height=MAX_VIDEO_HEIGHT, target_bytes=target_bytes,
            audio_kbps=VIDEO_AUDIO_KBPS,
        )


def _build_mobile_preview_sync(personalized: Path, preview: Path) -> None:
    """Create a small baseline JPEG used for fast Discord/mobile display."""
    with Image.open(personalized) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((IMAGE_PREVIEW_MAX_DIM, IMAGE_PREVIEW_MAX_DIM), Image.Resampling.LANCZOS)
        preview.parent.mkdir(parents=True, exist_ok=True)
        im.save(
            preview,
            "JPEG",
            quality=max(70, min(95, IMAGE_PREVIEW_QUALITY)),
            optimize=True,
            progressive=False,
            subsampling=2,
        )


def media_token(reveal_id: str, user_id: int) -> str:
    payload = f"media:{reveal_id}:{int(user_id)}".encode("utf-8")
    return hmac.new(WM_KEY, payload, hashlib.sha256).hexdigest()


def _media_url(reveal_id: str, user_id: int, variant: str = "preview") -> Optional[str]:
    if not PUBLIC_URL:
        return None
    token = media_token(reveal_id, user_id)
    return f"{PUBLIC_URL}/media/{reveal_id}/{int(user_id)}/{token}?v={variant}"


async def build_personalized_reveal(user_id: int, *, video_target_bytes: Optional[int] = None) -> tuple[Path, Optional[Path]]:
    reveal = CURRENT_REVEAL
    if not reveal or not Path(reveal["path"]).exists():
        raise RuntimeError("There is currently no valid reveal.")
    source = Path(reveal["path"])
    kind = reveal["kind"]
    reveal_id = reveal["reveal_id"]
    cache_dir = CACHE_DIR / reveal_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".mp4" if kind == "video" else (source.suffix.lower() if source.suffix.lower() in IMAGE_EXTS else ".png")
    if kind == "video":
        target_mb = int(round((video_target_bytes or (VIDEO_TARGET_MAX_MB * 1048576)) / 1048576))
        output = cache_dir / f"user_{user_id}_{VIDEO_CACHE_VERSION}_{target_mb}mb.mp4"
    else:
        output = cache_dir / f"user_{user_id}{suffix}"
    preview = cache_dir / f"user_{user_id}_mobile.jpg" if kind == "image" else None

    needs_output = not output.exists() or output.stat().st_size <= 0
    needs_preview = preview is not None and (not preview.exists() or preview.stat().st_size <= 0)

    if not needs_output and not needs_preview:
        return output, preview

    async with USER_LOCKS[(reveal_id, user_id)]:
        needs_output = not output.exists() or output.stat().st_size <= 0
        needs_preview = preview is not None and (not preview.exists() or preview.stat().st_size <= 0)
        if not needs_output and not needs_preview:
            return output, preview

        async with PROCESS_SEMAPHORE:
            if needs_output:
                temp_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
                try:
                    await asyncio.to_thread(
                        _build_sync, source, kind, temp_output, user_id, reveal_id, video_target_bytes
                    )
                    temp_output.replace(output)
                finally:
                    temp_output.unlink(missing_ok=True)

            if preview is not None and (not preview.exists() or preview.stat().st_size <= 0):
                temp_preview = preview.with_name(f"{preview.stem}.tmp{preview.suffix}")
                try:
                    await asyncio.to_thread(_build_mobile_preview_sync, output, temp_preview)
                    temp_preview.replace(preview)
                finally:
                    temp_preview.unlink(missing_ok=True)

    return output, preview


async def pending_upload_sweeper() -> None:
    while True:
        await asyncio.sleep(PENDING_SWEEP_INTERVAL)
        now = time.monotonic()
        for key, expiry in list(PENDING_UPLOADS.items()):
            if expiry <= now:
                PENDING_UPLOADS.pop(key, None)


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "discord_ready": bot.is_ready() if "bot" in globals() else False,
        "service": "revealbot",
    })


async def media_handler(request: web.Request) -> web.StreamResponse:
    reveal_id = request.match_info["reveal_id"]
    raw_user_id = request.match_info["user_id"]
    token = request.match_info["token"]

    if not REVEAL_ID_RE.fullmatch(reveal_id):
        raise web.HTTPNotFound()
    try:
        user_id = int(raw_user_id)
    except ValueError:
        raise web.HTTPNotFound()

    expected = media_token(reveal_id, user_id)
    if not hmac.compare_digest(token, expected):
        raise web.HTTPNotFound()

    reveal = find_reveal(reveal_id)
    if not reveal or reveal["kind"] != "image":
        raise web.HTTPNotFound()

    source_suffix = Path(reveal["path"]).suffix.lower()
    if source_suffix not in IMAGE_EXTS:
        source_suffix = ".png"

    original = CACHE_DIR / reveal_id / f"user_{user_id}{source_suffix}"
    preview = CACHE_DIR / reveal_id / f"user_{user_id}_mobile.jpg"
    variant = request.query.get("v", "preview").lower()
    if variant not in {"preview", "original"}:
        raise web.HTTPNotFound()

    path = preview if variant == "preview" else original
    if not path.exists() or path.stat().st_size <= 0:
        raise web.HTTPNotFound()

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    response = web.FileResponse(path, headers={
        "Content-Type": content_type,
        "Content-Disposition": f'inline; filename="{path.name}"',
        "Cache-Control": "public, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
    })
    return response


async def start_http_server() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/media/{reveal_id}/{user_id}/{token}", media_handler)

    runner = web.AppRunner(app, access_log=log)
    await runner.setup()
    site = web.TCPSite(runner, HTTP_HOST, HTTP_PORT)
    await site.start()
    log.info("HTTP server listening on %s:%d (public URL: %s)", HTTP_HOST, HTTP_PORT,
             PUBLIC_URL or "not configured")
    return runner


def clear_old_cache(keep_reveal_id: str) -> None:
    for child in CACHE_DIR.iterdir():
        if child.is_dir() and child.name != keep_reveal_id:
            shutil.rmtree(child, ignore_errors=True)
    for key in [k for k in USER_LOCKS if k[0] != keep_reveal_id]:
        USER_LOCKS.pop(key, None)


async def save_new_reveal(attachment: discord.Attachment, kind: str, normalized_ext: str) -> dict:
    global CURRENT_REVEAL
    reveal_id = f"{int(time.time())}_{secrets.token_hex(4)}"
    reveal_dir = REVEALS_DIR / reveal_id
    reveal_dir.mkdir(parents=True, exist_ok=True)
    temp_path = TMP_DIR / f"upload_{secrets.token_hex(8)}{normalized_ext}"
    source_path = reveal_dir / f"original{normalized_ext}"
    try:
        await attachment.save(temp_path, use_cached=False)
        temp_path.replace(source_path)
        metadata = {"reveal_id": reveal_id, "kind": kind, "path": str(source_path),
                    "filename": attachment.filename, "created_at": int(time.time())}
        safe_json_write(META_FILE, metadata)
        CURRENT_REVEAL = metadata
        await asyncio.to_thread(clear_old_cache, reveal_id)
        if not KEEP_ORIGINALS:
            for old_dir in REVEALS_DIR.iterdir():
                if old_dir.is_dir() and old_dir.name != reveal_id:
                    shutil.rmtree(old_dir, ignore_errors=True)
        return metadata
    finally:
        temp_path.unlink(missing_ok=True)


intents = discord.Intents.default()
intents.message_content = True


class RevealBot(commands.Bot):
    async def setup_hook(self) -> None:
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
        else:
            synced = await self.tree.sync()
        log.info("Synced %d command(s).", len(synced))


bot = RevealBot(command_prefix="!", intents=intents,
                allowed_mentions=discord.AllowedMentions.none())


@bot.event
async def on_ready():
    log.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "?")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return
    key = (message.guild.id, message.channel.id, message.author.id)
    expiry = PENDING_UPLOADS.get(key)
    if expiry is None:
        return
    if time.monotonic() > expiry:
        PENDING_UPLOADS.pop(key, None)
        return
    attachment = allowed_attachment = None
    kind = ext = None
    if len(message.attachments) == 1:
        attachment = message.attachments[0]
        ok, kind, ext = is_allowed_media(attachment.filename, attachment.content_type)
        if ok:
            allowed_attachment = attachment
    if not allowed_attachment:
        try:
            await message.reply("Please send exactly one image or video attachment.", mention_author=False, delete_after=8)
        except discord.HTTPException:
            pass
        return
    PENDING_UPLOADS.pop(key, None)
    try:
        metadata = await save_new_reveal(allowed_attachment, kind, ext)
    except Exception:
        log.exception("Reveal upload failed.")
        await message.channel.send(f"❌ I couldn't save that {kind}. Check the bot logs.", delete_after=10)
        return
    if DELETE_UPLOAD_MESSAGE:
        try:
            await message.delete()
        except discord.HTTPException:
            pass
    await message.channel.send(
        f"✅ Reveal uploaded ({metadata['kind']}, id `{metadata['reveal_id']}`). Boosters can now use `/view_reveal`.",
        delete_after=15,
    )


@bot.tree.command(name="upload", description="Start an admin-only reveal upload.")
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
async def upload(interaction: discord.Interaction):
    key = (interaction.guild.id, interaction.channel.id, interaction.user.id)
    PENDING_UPLOADS[key] = time.monotonic() + PENDING_UPLOAD_TIMEOUT
    await interaction.response.send_message(
        f"Send **one image or video** in this channel within {PENDING_UPLOAD_TIMEOUT // 60} minutes. "
        "The next valid attachment you send will become the current reveal.", ephemeral=True)


@bot.tree.command(name="view_reveal", description="Get your booster reveal.")
@app_commands.guild_only()
async def view_reveal(interaction: discord.Interaction):
    member = interaction.user
    if not isinstance(member, discord.Member):
        await interaction.response.send_message("I couldn't verify your server membership.", ephemeral=True)
        return
    if not member_is_booster(member):
        await interaction.response.send_message("❌ This reveal is available to current server boosters only.", ephemeral=True)
        return
    if not CURRENT_REVEAL or not Path(CURRENT_REVEAL["path"]).exists():
        await interaction.response.send_message("There isn't a reveal uploaded right now.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        interaction_limit = getattr(interaction, "filesize_limit", None)
        guild_limit = getattr(interaction.guild, "filesize_limit", None)
        upload_limit = interaction_limit or guild_limit
        video_target_bytes = None
        if CURRENT_REVEAL["kind"] == "video":
            configured_target = int(VIDEO_TARGET_MAX_MB * 1048576)
            safe_limit = int(upload_limit * 0.90) if upload_limit else configured_target
            video_target_bytes = max(2 * 1048576, min(configured_target, safe_limit))
        personalized, preview = await build_personalized_reveal(
            member.id, video_target_bytes=video_target_bytes
        )
        size_bytes = personalized.stat().st_size
        reveal_id = CURRENT_REVEAL["reveal_id"]

        # Images use a single Discord embed backed by the fast mobile preview.
        # There is intentionally no separate full-resolution link.
        preview_url = _media_url(reveal_id, member.id, "preview")
        if preview is not None and preview_url:
            embed = discord.Embed()
            embed.set_image(url=preview_url)
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            # Videos are sent as a normal MP4 attachment so Discord's native mobile
            # player handles playback. The personalized video is encoded as H.264/AAC
            # with fast-start metadata and a lower CRF for cleaner playback.
            if upload_limit and size_bytes > upload_limit:
                await interaction.followup.send(
                    f"❌ The personalized file is too large for this server's upload limit "
                    f"({size_bytes / 1048576:.1f} MB vs {upload_limit / 1048576:.1f} MB).",
                    ephemeral=True,
                )
                return
            await interaction.followup.send(
                "Here is your personalized reveal. It is uniquely marked to you — please don't share it.",
                file=discord.File(str(personalized), filename="personalized-reveal.mp4"), ephemeral=True)

        ledger_append(reveal_id, member.id)
        log.info("Served reveal_id=%s to user_id=%s size=%d", reveal_id, member.id, size_bytes)
    except Exception:
        log.exception("Failed to build/send personalized reveal.")
        await interaction.followup.send("❌ I couldn't generate your personalized reveal.", ephemeral=True)


@bot.tree.command(name="trace", description="Admin: identify which user a leaked reveal came from.")
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
@app_commands.describe(
    file="The leaked image, video, or a frame from a video",
    reveal_id="Reveal ID to test against (defaults to the current reveal)",
    start_frame="Optional original video frame hint; the tracer searches around it",
)
async def trace(interaction: discord.Interaction, file: discord.Attachment,
                reveal_id: Optional[str] = None,
                start_frame: app_commands.Range[int, 0, 200000] = 0):
    reveal = find_reveal(reveal_id)
    if not reveal:
        await interaction.response.send_message("I couldn't find that reveal's original file.", ephemeral=True)
        return
    ok, leak_kind, ext = is_allowed_media(file.filename, file.content_type)
    if not ok or (reveal["kind"] == "image" and leak_kind != "image") or (reveal["kind"] == "video" and leak_kind not in {"image", "video"}):
        await interaction.response.send_message(
            f"The leak must be a {reveal['kind']} or, for a video reveal, a frame image.", ephemeral=True)
        return

    trace_kind = "video_frame" if reveal["kind"] == "video" and leak_kind == "image" else leak_kind
    await interaction.response.defer(ephemeral=True, thinking=True)
    leak_path = TMP_DIR / f"leak_{secrets.token_hex(8)}{ext}"
    try:
        await file.save(leak_path, use_cached=False)
        async with PROCESS_SEMAPHORE:
            result = await asyncio.to_thread(
                wm.extract, Path(reveal["path"]), leak_path, trace_kind, WM_KEY, reveal["reveal_id"],
                image_cell=IMAGE_CELL, video_cell=VIDEO_CELL, max_height=MAX_VIDEO_HEIGHT,
                start_frame=int(start_frame))
        if not result["valid"]:
            await interaction.followup.send(
                "No valid watermark found. The leak may contain too little of the original, be heavily edited, "
                "or belong to another reveal.", ephemeral=True)
            return
        uid = result["user_id"]
        served = ledger_served(reveal["reveal_id"], uid)
        member = interaction.guild.get_member(uid)
        frame_text = f" • source frame `{result['frame']}`" if "frame" in result else ""
        await interaction.followup.send(
            f"🔎 Watermark decoded (CRC verified): <@{uid}> (`{uid}`)"
            f"{' — in this server' if member else ' — not currently in this server'}\n"
            f"Reveal `{reveal['reveal_id']}`{frame_text} • served to this user: {'**yes**' if served else '**no record**'}",
            ephemeral=True)
        log.warning("TRACE by %s: reveal=%s decoded_user=%s", interaction.user.id, reveal["reveal_id"], uid)
    except Exception:
        log.exception("Trace failed.")
        await interaction.followup.send("❌ Trace failed. Check the bot logs.", ephemeral=True)
    finally:
        leak_path.unlink(missing_ok=True)


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.errors.MissingPermissions):
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
    http_runner = await start_http_server()
    sweeper = asyncio.create_task(pending_upload_sweeper(), name="pending-upload-sweeper")
    try:
        async with bot:
            await bot.start(TOKEN)
    finally:
        sweeper.cancel()
        await asyncio.gather(sweeper, return_exceptions=True)
        await http_runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
