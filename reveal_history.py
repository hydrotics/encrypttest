from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import zlib
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiohttp
import discord
from discord import app_commands

log = logging.getLogger("revealbot.history")

ARCHIVE_MARKER = "revealbot-archive"
LEGACY_ARCHIVE_MARKER = "revealbot-archive:v2"
ARCHIVE_DATA_PREFIX = "revealbot-data:"
ZW_MARKER = "\u2063\u2063\u2063\u2062"
ZW_CHARS = "\u200b\u200c\u200d\u2060"
ZW_REVERSE = {ch: i for i, ch in enumerate(ZW_CHARS)}
PAGE_SIZE = 8
FIELD_CHUNK = 900
DOWNLOAD_CHUNK = 1 << 20
DEFAULT_DOWNLOAD_LIMIT = 250 * 1048576


class RevealHistoryError(RuntimeError):
    pass


class RevealHistoryStore:
    """Discord-backed reveal registry.

    Render Free has an ephemeral local filesystem, so the archive channel is the
    persistence layer. Each reveal is one bot-authored message containing the
    original media attachment plus a compact metadata embed. The message ID is
    the durable reveal record.

    Keep this channel private to staff and the bot.
    """

    def __init__(self, bot: discord.Client, archive_channel_id: int = 0, staff_role_id: int = 0) -> None:
        self.bot = bot
        self.archive_channel_id = int(archive_channel_id or 0)
        self.staff_role_id = int(staff_role_id or 0)
        self._edit_lock = asyncio.Lock()
        self._cache: dict[str, dict] = {}
        self._loaded = False

    @property
    def enabled(self) -> bool:
        return bool(self.archive_channel_id)

    async def _channel(self) -> discord.TextChannel:
        if not self.archive_channel_id:
            raise RevealHistoryError(
                "REVEAL_ARCHIVE_CHANNEL_ID is not configured; persistent reveal storage is disabled."
            )
        channel = self.bot.get_channel(self.archive_channel_id)
        if isinstance(channel, discord.TextChannel):
            return channel
        fetched = await self.bot.fetch_channel(self.archive_channel_id)
        if not isinstance(fetched, discord.TextChannel):
            raise RevealHistoryError("REVEAL_ARCHIVE_CHANNEL_ID is not a text channel.")
        return fetched

    @staticmethod
    def _normalise_record(record: dict) -> dict:
        return {
            "reveal_id": str(record["reveal_id"]),
            "kind": str(record.get("kind", "unknown")),
            "path": str(record.get("path", "")),
            "filename": str(record.get("filename", "original")),
            "created_at": int(record.get("created_at", 0) or 0),
            "archive_channel_id": int(record.get("archive_channel_id", 0) or 0),
            "archive_message_id": int(record.get("archive_message_id", 0) or 0),
            "archive_attachment_id": int(record.get("archive_attachment_id", 0) or 0),
            "archive_filename": str(record.get("archive_filename", record.get("filename", "original"))),
            "served_user_ids": sorted({int(x) for x in record.get("served_user_ids", [])}),
            "stress_test_user_labels": sorted({str(x) for x in record.get("stress_test_user_labels", []) if str(x).strip()}),
            "profiles": sorted(
                {(int(p[0]), round(float(p[1]), 3)) for p in record.get("profiles", []) if len(p) >= 2},
                reverse=True,
            ),
            "width": int(record.get("width", 0) or 0),
            "height": int(record.get("height", 0) or 0),
            "fps": round(float(record.get("fps", 0.0) or 0.0), 3),
            "duration": float(record.get("duration", 0.0) or 0.0),
        }

    @staticmethod
    def _encode_archive_data(record: dict) -> str:
        payload = {
            "u": [int(x) for x in record.get("served_user_ids", [])],
            "s": [str(x) for x in record.get("stress_test_user_labels", [])],
        }
        raw = zlib.compress(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            9,
        )
        encoded = "".join(ZW_CHARS[(byte >> shift) & 0x03] for byte in raw for shift in (6, 4, 2, 0))
        return ZW_MARKER + encoded

    @staticmethod
    def _decode_archive_data(content: str) -> tuple[set[int], set[str]]:
        encoded = ""
        if content.startswith(ZW_MARKER):
            encoded = content[len(ZW_MARKER):]
            if any(ch not in ZW_REVERSE for ch in encoded):
                encoded = ""
            elif len(encoded) % 4:
                encoded = ""
            else:
                try:
                    raw = bytes(
                        (ZW_REVERSE[encoded[i]] << 6)
                        | (ZW_REVERSE[encoded[i + 1]] << 4)
                        | (ZW_REVERSE[encoded[i + 2]] << 2)
                        | ZW_REVERSE[encoded[i + 3]]
                        for i in range(0, len(encoded), 4)
                    )
                    payload = json.loads(zlib.decompress(raw).decode("utf-8"))
                except Exception:
                    payload = None
                if payload is not None:
                    users = set()
                    labels = set()
                    for value in payload.get("u", []):
                        try:
                            users.add(int(value))
                        except (TypeError, ValueError):
                            pass
                    for value in payload.get("s", []):
                        label = str(value).strip()
                        if label:
                            labels.add(label)
                    return users, labels

        prefix = ARCHIVE_DATA_PREFIX
        if prefix not in content:
            return set(), set()
        encoded = content.split(prefix, 1)[1].strip()
        try:
            payload = json.loads(zlib.decompress(base64.urlsafe_b64decode(encoded)).decode("utf-8"))
        except Exception:
            return set(), set()
        users = set()
        labels = set()
        for value in payload.get("u", []):
            try:
                users.add(int(value))
            except (TypeError, ValueError):
                pass
        for value in payload.get("s", []):
            label = str(value).strip()
            if label:
                labels.add(label)
        return users, labels

    def _served_view(self) -> "ArchiveServedUsersView":
        return ArchiveServedUsersView(self, self.staff_role_id)

    def _embed_for(self, record: dict) -> discord.Embed:
        kind = record["kind"]
        embed = discord.Embed(title="Reveal")
        embed.add_field(name="Reveal ID", value=f"`{record['reveal_id']}`", inline=True)
        embed.add_field(name="Type", value=kind, inline=True)
        embed.add_field(name="Created", value=f"<t:{record['created_at']}:F>" if record["created_at"] else "unknown", inline=True)
        if record.get("filename"):
            embed.add_field(name="Original name", value=record["filename"][:1024], inline=False)
        if record.get("width") and record.get("height"):
            media = f"{record['width']}×{record['height']}"
            if record.get("fps"):
                media += f" @ {record['fps']:.3f} fps"
            if record.get("duration"):
                media += f" • {record['duration']:.1f}s"
            embed.add_field(name="Media", value=media[:1024], inline=False)
        server_count = len(record.get("served_user_ids", []))
        embed.add_field(name="Server users", value=str(server_count), inline=True)
        stress_count = len(record.get("stress_test_user_labels", []))
        if stress_count:
            embed.add_field(name="Stress test users", value=str(stress_count), inline=True)
        profiles = [f"{int(h)}h/{float(fps):.3f}fps" for h, fps in record.get("profiles", [])]
        if profiles:
            embed.add_field(name="Delivery profiles", value=", ".join(profiles)[:1024], inline=False)
        return embed

    @staticmethod
    def _is_archive_message(message: discord.Message) -> bool:
        if not message.author.bot or not message.embeds:
            return False
        return (
            message.content.startswith((ARCHIVE_MARKER, LEGACY_ARCHIVE_MARKER))
            or message.content.startswith(ZW_MARKER)
        )

    @classmethod
    def _record_from_message(cls, message: discord.Message) -> Optional[dict]:
        if not cls._is_archive_message(message):
            return None
        embed = message.embeds[0]
        fields = {field.name: field.value for field in embed.fields}
        rid = fields.get("Reveal ID", "").strip("`")
        kind = fields.get("Type", "").strip()
        if not rid or not kind:
            return None
        created = 0
        for field_name, value in fields.items():
            if field_name == "Created" and value.startswith("<t:"):
                try:
                    created = int(value.split(":", 2)[1])
                except (ValueError, IndexError):
                    pass
        served, stress_test_users = RevealHistoryStore._decode_archive_data(message.content)
        if not served and not stress_test_users:
            for field_name, value in fields.items():
                if not field_name.startswith("Served users"):
                    continue
                raw = value.replace("`", "")
                for item in raw.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    try:
                        served.add(int(item))
                    except ValueError:
                                                                               
                                                                                             
                        if item.lower().startswith("test"):
                            stress_test_users.add(item)
        profiles: set[tuple[int, float]] = set()
        for item in fields.get("Delivery profiles", "").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                h_text, fps_text = item.split("h/", 1)
                profiles.add((int(h_text), float(fps_text.removesuffix("fps"))))
            except (ValueError, IndexError):
                continue

        filename = message.attachments[0].filename if message.attachments else "original"
        attachment_id = int(message.attachments[0].id) if message.attachments else 0
        return cls._normalise_record(
            {
                "reveal_id": rid,
                "kind": kind,
                "filename": filename,
                "created_at": created,
                "archive_channel_id": message.channel.id,
                "archive_message_id": message.id,
                "archive_attachment_id": attachment_id,
                "archive_filename": filename,
                "served_user_ids": sorted(served),
                "stress_test_user_labels": sorted(stress_test_users),
                "profiles": sorted(profiles, reverse=True),
            }
        )

    async def load(self) -> list[dict]:
        """Load the durable registry from the archive channel."""
        if not self.enabled:
            self._loaded = True
            self._cache.clear()
            return []
        channel = await self._channel()
        found: list[dict] = []
        try:
            async for message in channel.history(limit=None, oldest_first=False):
                record = self._record_from_message(message)
                if record:
                    found.append(record)
                    try:
                        kwargs = {"view": self._served_view()}
                        if message.content.startswith((ARCHIVE_MARKER, LEGACY_ARCHIVE_MARKER)):
                            kwargs["content"] = self._encode_archive_data(record)
                        if not message.components or "content" in kwargs:
                            await message.edit(**kwargs)
                    except discord.HTTPException:
                        log.debug("Could not migrate archive message %s", message.id)
        except discord.HTTPException as exc:
            raise RevealHistoryError(f"Could not read reveal archive channel: {exc}") from exc
        found.sort(key=lambda r: int(r.get("created_at", 0) or 0), reverse=True)
        self._cache = {r["reveal_id"]: r for r in found}
        self._loaded = True
        return found

    async def records(self, *, force_refresh: bool = False) -> list[dict]:
        if force_refresh or not self._loaded:
            await self.load()
        rows = list(self._cache.values())
        rows.sort(key=lambda r: int(r.get("created_at", 0) or 0), reverse=True)
        return rows

    async def get(self, reveal_id: str, *, force_refresh: bool = False) -> Optional[dict]:
        if not self._loaded or force_refresh:
            await self.load()
        return self._cache.get(reveal_id)

    async def by_archive_message_id(self, message_id: int) -> Optional[dict]:
        rows = await self.records()
        target = int(message_id)
        for record in rows:
            if int(record.get("archive_message_id", 0) or 0) == target:
                return record
        return None

    async def latest(self) -> Optional[dict]:
        rows = await self.records()
        return rows[0] if rows else None

    async def archive(self, reveal: dict) -> dict:
        if not self.enabled:
            raise RevealHistoryError("Reveal archive is not configured.")
        source = Path(reveal["path"])
        if not source.is_file():
            raise RevealHistoryError("The local reveal source disappeared before it could be archived.")
        channel = await self._channel()
        limit = int(getattr(channel.guild, "filesize_limit", 0) or 0)
        size = source.stat().st_size
        if limit and size > limit:
            raise RevealHistoryError(
                f"The source is {size / 1048576:.1f} MiB, above this server's attachment limit "
                f"of {limit / 1048576:.1f} MiB; it cannot be made restart-safe on Discord-only storage."
            )

        record = self._normalise_record({**reveal, "served_user_ids": [], "profiles": []})
        discord_file = discord.File(source, filename=source.name, spoiler=False)
        try:
            message = await channel.send(
                content=self._encode_archive_data(record),
                embed=self._embed_for(record),
                file=discord_file,
                view=self._served_view(),
                allowed_mentions=discord.AllowedMentions.none(),
            )
        finally:
            discord_file.close()

        record.update(
            {
                "archive_channel_id": message.channel.id,
                "archive_message_id": message.id,
                "archive_attachment_id": int(message.attachments[0].id) if message.attachments else 0,
                "archive_filename": message.attachments[0].filename if message.attachments else source.name,
            }
        )
                                                                                             
                                                                                                        
        self._cache[record["reveal_id"]] = record
        self._loaded = True
        return record

    async def _fetch_message(self, record: dict) -> discord.Message:
        channel = await self._channel()
        if int(record.get("archive_channel_id", channel.id)) != channel.id:
            raise RevealHistoryError("Reveal archive channel mismatch.")
        try:
            return await channel.fetch_message(int(record["archive_message_id"]))
        except discord.HTTPException as exc:
            raise RevealHistoryError(f"Could not fetch archived reveal {record['reveal_id']}: {exc}") from exc

    async def ensure_local_copy(self, record: dict, destination: Path, max_bytes: int = DEFAULT_DOWNLOAD_LIMIT) -> Path:
        """Download the archived original without buffering the whole attachment in RAM."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        message = await self._fetch_message(record)
        attachment = None
        target_id = int(record.get("archive_attachment_id", 0) or 0)
        for item in message.attachments:
            if target_id and int(item.id) == target_id:
                attachment = item
                break
        if attachment is None and message.attachments:
            attachment = message.attachments[0]
        if attachment is None:
            raise RevealHistoryError(f"Archived reveal {record['reveal_id']} has no source attachment.")
        if attachment.size > max_bytes:
            raise RevealHistoryError(
                f"Archived reveal {record['reveal_id']} is {attachment.size / 1048576:.1f} MiB, above the local restore limit."
            )

        partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        timeout = aiohttp.ClientTimeout(total=900, connect=15, sock_read=60)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "RevealBot/1.0"}) as session:
                async with session.get(attachment.url) as response:
                    if response.status != 200:
                        raise RevealHistoryError(f"Discord CDN returned HTTP {response.status} while restoring a reveal.")
                    written = 0
                    with partial.open("wb") as fh:
                        async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK):
                            written += len(chunk)
                            if written > max_bytes:
                                raise RevealHistoryError("Archived reveal exceeded the local restore size limit.")
                            fh.write(chunk)
                            if written and written % (8 * DOWNLOAD_CHUNK) == 0:
                                fh.flush()
                    if written <= 0:
                        raise RevealHistoryError("Discord returned an empty reveal attachment.")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        return destination

    async def record_delivery(
        self,
        reveal_id: str,
        user_id: int,
        profile: Optional[tuple[int, float]] = None,
    ) -> None:
        """Persist served-user and video-profile data into the archive message."""
        async with self._edit_lock:
            record = await self.get(reveal_id, force_refresh=False)
            if record is None:
                return
            uid = int(user_id)
            record.setdefault("served_user_ids", [])
            served = set(int(x) for x in record["served_user_ids"])
            already = uid in served
            served.add(uid)
            record["served_user_ids"] = sorted(served)
            if profile is not None:
                p = (int(profile[0]), round(float(profile[1]), 3))
                record.setdefault("profiles", [])
                if p not in record["profiles"]:
                    record["profiles"].append(p)
                    record["profiles"].sort(reverse=True)
            if already and profile is None:
                return

            message = await self._fetch_message(record)
            try:
                await message.edit(
                    content=self._encode_archive_data(record),
                    embeds=[self._embed_for(record)],
                    attachments=list(message.attachments),
                    view=self._served_view(),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except discord.HTTPException:
                log.exception("Could not persist delivery info for reveal=%s", reveal_id)
                raise

            self._cache[reveal_id] = record

    async def record_stress_test_users(self, reveal_id: str, labels: list[str]) -> None:
        """Persist clearly-labelled synthetic test users without adding fake IDs to tracing data."""
        clean = sorted({str(label).strip() for label in labels if str(label).strip()})
        if not clean:
            return
        async with self._edit_lock:
            record = await self.get(reveal_id, force_refresh=False)
            if record is None:
                return
            record.setdefault("stress_test_user_labels", [])
            before = set(record["stress_test_user_labels"])
            record["stress_test_user_labels"] = sorted(before.union(clean))
            if record["stress_test_user_labels"] == sorted(before):
                return

            message = await self._fetch_message(record)
            try:
                await message.edit(
                    content=self._encode_archive_data(record),
                    embeds=[self._embed_for(record)],
                    attachments=list(message.attachments),
                    view=self._served_view(),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except discord.HTTPException:
                log.exception("Could not persist stress-test users for reveal=%s", reveal_id)
                raise
            self._cache[reveal_id] = record

    async def delete(self, reveal_id: str) -> bool:
        record = await self.get(reveal_id, force_refresh=True)
        if record is None:
            return False
        message = await self._fetch_message(record)
        try:
            await message.delete()
        except discord.NotFound:
            pass
        except discord.HTTPException as exc:
            raise RevealHistoryError(f"Could not delete archived reveal {reveal_id}: {exc}") from exc
        self._cache.pop(reveal_id, None)
        return True


class ArchiveServedUsersView(discord.ui.View):
    def __init__(self, store: RevealHistoryStore, staff_role_id: int) -> None:
        super().__init__(timeout=None)
        self.store = store
        self.staff_role_id = int(staff_role_id)
        button = discord.ui.Button(
            label="View served users",
            style=discord.ButtonStyle.secondary,
            custom_id="revealbot:archive-served-users:v1",
        )
        button.callback = self._callback
        self.add_item(button)

    async def _callback(self, interaction: discord.Interaction) -> None:
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        if member is None or not any(role.id == self.staff_role_id for role in member.roles):
            await interaction.response.send_message("You don't have permission to use this button.", ephemeral=True)
            return
        record = await self.store.by_archive_message_id(interaction.message.id if interaction.message else 0)
        if record is None:
            await interaction.response.send_message("That reveal record could not be found.", ephemeral=True)
            return
        mentions = [f"<@{int(uid)}>" for uid in record.get("served_user_ids", [])]
        mentions.extend(str(label) for label in record.get("stress_test_user_labels", []))
        if not mentions:
            await interaction.response.send_message("No users have been served yet.", ephemeral=True)
            return
        chunks = []
        current = ""
        for index, item in enumerate(mentions, 1):
            line = f"{index}. {item}"
            candidate = line if not current else f"{current}\n{line}"
            if len(candidate) > 1800:
                chunks.append(current)
                current = line
            else:
                current = candidate
        if current:
            chunks.append(current)
        await interaction.response.send_message(
            f"Served users: {len(mentions)}\n{chunks[0]}",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
        )
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True, allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False))


def make_history_command(
    store: RevealHistoryStore,
    cleanup_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    before_delete_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    staff_role_id: int = 0,
) -> app_commands.Command:
    class HistoryView(discord.ui.View):
        def __init__(self, owner_id: int, rows: list[dict], page: int = 0) -> None:
            super().__init__(timeout=10 * 60)
            self.owner_id = owner_id
            self.rows = rows
            self.page = page
            start = page * PAGE_SIZE
            end = start + PAGE_SIZE
            visible = rows[start:end]
            for record in visible:
                short = record["reveal_id"][-8:]
                button = discord.ui.Button(
                    label=f"Delete {short}",
                    style=discord.ButtonStyle.danger,
                    custom_id=f"reveal_history:delete:{record['reveal_id']}",
                )
                button.callback = self._make_delete_callback(record["reveal_id"])
                self.add_item(button)

            total_pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
            if total_pages > 1:
                prev = discord.ui.Button(
                    label="Previous", style=discord.ButtonStyle.secondary,
                    disabled=page <= 0,
                    custom_id=f"reveal_history:prev:{page}",
                )
                prev.callback = self._previous
                nxt = discord.ui.Button(
                    label="Next", style=discord.ButtonStyle.secondary,
                    disabled=page >= total_pages - 1,
                    custom_id=f"reveal_history:next:{page}",
                )
                nxt.callback = self._next
                self.add_item(prev)
                self.add_item(nxt)

        def _make_delete_callback(self, reveal_id: str):
            async def callback(interaction: discord.Interaction) -> None:
                member = interaction.user if isinstance(interaction.user, discord.Member) else None
                if member is None or not any(role.id == int(staff_role_id) for role in member.roles):
                    await interaction.response.send_message("You don't have permission to use these buttons.", ephemeral=True)
                    return
                await interaction.response.defer(ephemeral=True)
                try:
                    if before_delete_callback:
                        await before_delete_callback(reveal_id)
                    removed = await store.delete(reveal_id)
                    if cleanup_callback:
                        await cleanup_callback(reveal_id)
                    self.rows = await store.records(force_refresh=False)
                    self.page = min(self.page, max(0, (len(self.rows) - 1) // PAGE_SIZE))
                    embed = _history_embed(self.rows, self.page)
                    replacement = HistoryView(self.owner_id, self.rows, self.page)
                    await interaction.edit_original_response(embed=embed, view=replacement)
                    if not removed:
                        await interaction.followup.send("That reveal was already deleted.", ephemeral=True)
                except Exception:
                    log.exception("Reveal history delete failed for %s", reveal_id)
                    await interaction.followup.send("I couldn't delete that reveal. Check the bot logs.", ephemeral=True)
            return callback

        async def _previous(self, interaction: discord.Interaction) -> None:
            member = interaction.user if isinstance(interaction.user, discord.Member) else None
            if member is None or not any(role.id == int(staff_role_id) for role in member.roles):
                await interaction.response.send_message("You don't have permission to use these buttons.", ephemeral=True)
                return
            self.page = max(0, self.page - 1)
            await interaction.response.edit_message(embed=_history_embed(self.rows, self.page), view=HistoryView(self.owner_id, self.rows, self.page))

        async def _next(self, interaction: discord.Interaction) -> None:
            member = interaction.user if isinstance(interaction.user, discord.Member) else None
            if member is None or not any(role.id == int(staff_role_id) for role in member.roles):
                await interaction.response.send_message("You don't have permission to use these buttons.", ephemeral=True)
                return
            total_pages = max(1, (len(self.rows) + PAGE_SIZE - 1) // PAGE_SIZE)
            self.page = min(total_pages - 1, self.page + 1)
            await interaction.response.edit_message(embed=_history_embed(self.rows, self.page), view=HistoryView(self.owner_id, self.rows, self.page))

    def _history_embed(rows: list[dict], page: int) -> discord.Embed:
        total_pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
        embed = discord.Embed(title="Reveal History")
        if not rows:
            embed.add_field(name="No reveals", value="The archive is empty.", inline=False)
            return embed
        start = page * PAGE_SIZE
        for record in rows[start:start + PAGE_SIZE]:
            served_count = len(record.get("served_user_ids", []))
            embed.add_field(
                name=f"{record['kind'].title()} • `{record['reveal_id']}`",
                value=f"Created: <t:{int(record.get('created_at', 0) or 0)}:R> • served to {served_count} server user(s)\nUse the matching red button below to delete it.",
                inline=False,
            )
        embed.set_footer(text=f"Page {page + 1}/{total_pages} • {len(rows)} reveal(s)")
        return embed

    @app_commands.command(name="reveal_history", description="Staff: show and delete archived past reveals.")
    @app_commands.guild_only()
    @app_commands.check(lambda interaction: isinstance(interaction.user, discord.Member) and any(role.id == int(staff_role_id) for role in interaction.user.roles))
    async def reveal_history(interaction: discord.Interaction) -> None:
        rows = await store.records(force_refresh=True)
        await interaction.response.send_message(
            embed=_history_embed(rows, 0),
            view=HistoryView(interaction.user.id, rows, 0),
            ephemeral=True,
        )

    return reveal_history
