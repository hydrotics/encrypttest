from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import math
import os
import platform
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import discord
from discord import app_commands

log = logging.getLogger("revealbot.stresstest")
                                                                                                                
def _linux_rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass
    return 0


def _windows_rss(pid: int) -> int:
    """
    Windows fallback without importing the Unix-only resource module.
    Returns bytes from PROCESS_MEMORY_COUNTERS.WorkingSetSize.
    """
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
            False,
            int(pid),
        )
        if not handle:
            return 0
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ok = psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.WorkingSetSize) if ok else 0
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, ValueError):
        return 0


def _rss(pid: int) -> int:
    system = platform.system().lower()
    if system == "windows":
        return _windows_rss(pid)
    return _linux_rss(pid)


def _parent_rss() -> int:
    return _rss(os.getpid())


def _worker_rss(worker_procs: list[Any]) -> int:
    total = 0
    for proc in worker_procs:
        try:
            pid = int(proc.pid)
        except (AttributeError, TypeError, ValueError):
            continue
        total += _rss(pid)
    return total


def _mb(value: int) -> str:
    return f"{value / 1048576:.1f} MiB"


                                                                             
            
                                                                             

@dataclass(slots=True)
class StressResult:
    user_id: int
    ok: bool
    cached: bool = False
    request_seconds: float = 0.0
    generation_seconds: Optional[float] = None
    file_bytes: int = 0
    path: Optional[Path] = None
    build_info: Optional[dict] = None
    error: Optional[str] = None


@dataclass(slots=True)
class StressStats:
    planned: int = 0
    started: int = 0
    completed: int = 0
    failed: int = 0
    cancelled_by_deadline: int = 0
    cache_hits: int = 0
    real_builds: int = 0
    max_inflight: int = 0
    archive_test_users_recorded: int = 0

    trace_planned: int = 0
    trace_completed: int = 0
    trace_valid: int = 0
    trace_failed: int = 0
    trace_errors: list[str] = field(default_factory=list)

    errors: Counter[str] = field(default_factory=Counter)

    wall_seconds: float = 0.0

    peak_parent_rss: int = 0
    peak_worker_rss: int = 0
    peak_combined_rss: int = 0
    samples: int = 0

    peak_cache_bytes: int = 0
    peak_created_file_bytes: int = 0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * fraction
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "average": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "average": round(statistics_mean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


                                                                             
                          
                                                                             

async def _read_file_streaming(
    path: Path,
    *,
    chunk_size: int = 1 << 20,
) -> int:
    """
    Stream the entire generated file without loading it into Python RAM.
    This approximates the local file-read cost of creating/sending a Discord File.
    """
    def _read() -> int:
        total = 0
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
        return total

    return await asyncio.to_thread(_read)


def _cache_size(cache_dir: Path) -> int:
    total = 0
    try:
        for path in cache_dir.glob("*/*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


                                                                             
                                       
                                                                             

def _fake_user_ids(count: int, seed: int) -> list[int]:
    """
    Deterministic non-existent IDs. They are used ONLY as watermark payloads.
    No Discord requests are made for them.
    """
    base = 10_000_000_000_000_000 + ((seed & 0xFFFF) * 100_000)
    return [base + i for i in range(count)]


def _test_user_label(index: int) -> str:
    return f"test{int(index) + 1}"


def _realistic_schedule(
    *,
    members: int,
    viewers: int,
    repeat_ratio: float,
    stagger_seconds: float,
    rng: random.Random,
    population: Optional[list[int]] = None,
) -> list[tuple[float, int]]:
    members = max(1, min(int(members), 150))
    viewers = max(1, min(int(viewers), 150, members + max(0, int(viewers * 0.3))))

    population = list(population or _fake_user_ids(members, rng.randrange(1 << 30)))[:members]

                                                                    
    unique_count = min(members, viewers)
    unique_users = rng.sample(population, unique_count)

    repeat_count = 0
    if viewers > unique_count:
        repeat_count = viewers - unique_count
    else:
        desired_repeat_count = min(
            max(0, viewers - 1),
            int(round(viewers * max(0.0, min(repeat_ratio, 0.35)))),
        )
        unique_count = max(1, viewers - desired_repeat_count)
        unique_users = unique_users[:unique_count]
        repeat_count = viewers - unique_count

    sequence = list(unique_users)
    for _ in range(repeat_count):
        sequence.append(rng.choice(unique_users))
    rng.shuffle(sequence)

    if stagger_seconds <= 0:
        return [(0.0, uid) for uid in sequence]

                                                                                  
                                       
    mean_gap = stagger_seconds / max(len(sequence) * 0.55, 1.0)
    cursor = 0.0
    schedule: list[tuple[float, int]] = []

    for uid in sequence:
        gap = rng.expovariate(1.0 / max(mean_gap, 0.01))
        gap *= rng.uniform(0.65, 1.35)
        cursor = min(stagger_seconds, cursor + gap)
        schedule.append((cursor, uid))

    schedule.sort(key=lambda item: item[0])
    return schedule


async def _wait_until(
    target_offset: float,
    wall_start: float,
    deadline: float,
) -> bool:
    remaining = min(
        target_offset - (time.perf_counter() - wall_start),
        deadline - time.monotonic(),
    )
    if remaining <= 0:
        return time.monotonic() < deadline
    await asyncio.sleep(remaining)
    return time.monotonic() < deadline


                                                                             
                      
                                                                             

def install_stress_test(
    bot: discord.Client,
    *,
    get_current_reveal: Callable[[], Optional[dict]],
    ensure_reveal_source: Callable[[dict], Awaitable[dict]],
    build_personalized_reveal: Callable[..., Awaitable[tuple[Path, Optional[dict]]]],
    compute_video_target_bytes: Callable[[Optional[int]], int],
    run_worker: Callable[..., Awaitable[dict]],
    process_sem: Callable[[], asyncio.Semaphore],
    get_worker_procs: Callable[[], list[Any]],
    cache_dir: Path,
    tmp_dir: Path,
    max_video_height: int,
    trace_image_budget: float,
    trace_video_budget: float,
    cache_path_for: Optional[Callable[[dict, int, Optional[int]], Path]] = None,
    history_store: Optional[Any] = None,
    record_test_users: Optional[Callable[[str, list[str]], Awaitable[None]]] = None,
    staff_role_id: int = 0,
) -> None:

    @bot.tree.command(
        name="start_test",
        description="Staff: realistically stress-test reveal generation.",
    )
    @app_commands.guild_only()
    @app_commands.check(lambda interaction: isinstance(interaction.user, discord.Member) and any(role.id == int(staff_role_id) for role in interaction.user.roles))
    @app_commands.describe(
        members="Simulated booster population, 5-150.",
        viewers="How many simulated reveal views to run.",
        stagger_seconds="Spread requests over this many seconds.",
        repeat_ratio="Approximate repeat-click ratio when possible.",
        trace_jobs="Number of generated files to trace after the load wave.",
        duration_minutes="Hard maximum duration for the test.",
        cleanup="Delete only files created by this test after the report.",
        archive_test_users="Record simulated test users in the private reveal archive.",
    )
    async def start_test(
        interaction: discord.Interaction,
        members: app_commands.Range[int, 5, 150] = 70,
        viewers: app_commands.Range[int, 1, 150] = 60,
        stagger_seconds: app_commands.Range[float, 0.0, 180.0] = 45.0,
        repeat_ratio: app_commands.Range[float, 0.0, 0.35] = 0.12,
        trace_jobs: app_commands.Range[int, 0, 3] = 1,
        duration_minutes: app_commands.Range[int, 1, 15] = 8,
        cleanup: bool = True,
        archive_test_users: bool = True,
    ) -> None:
        if not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message(
                "This command can only be used inside the server.",
                ephemeral=True,
            )
            return

        reveal = get_current_reveal()
        if not reveal:
            await interaction.response.send_message(
                "There is no current reveal to stress test.",
                ephemeral=True,
            )
            return

                                                                                   
                                                                             
        members = int(max(5, min(members, 150)))
        viewers = int(max(1, min(viewers, 150)))

        await interaction.response.defer(ephemeral=True, thinking=True)

        hard_deadline = time.monotonic() + float(duration_minutes) * 60.0

        try:
            reveal = await ensure_reveal_source(reveal)
        except Exception:
            log.exception("Stress test could not restore current reveal.")
            await interaction.followup.send(
                "I couldn't restore the current reveal, so the stress test did not start.",
                ephemeral=True,
            )
            return

        upload_limit = (
            int(getattr(interaction, "filesize_limit", 0) or 0)
            or int(getattr(interaction.guild, "filesize_limit", 0) or 0)
            or 20 * 1048576
        )
        target_bytes = compute_video_target_bytes(upload_limit)

        seed = (
            int(time.time_ns())
            ^ int(interaction.user.id)
            ^ hash(reveal["reveal_id"])
        ) & 0xFFFFFFFF
        rng = random.Random(seed)
        population = _fake_user_ids(members, seed ^ 0x51A7)

        schedule = _realistic_schedule(
            members=members,
            viewers=viewers,
            repeat_ratio=float(repeat_ratio),
            stagger_seconds=float(stagger_seconds),
            rng=rng,
            population=population,
        )

        stats = StressStats(planned=len(schedule))
        results: list[StressResult] = []
        created_paths: set[Path] = set()
        result_lock = asyncio.Lock()

                                                                               
                                                                                 
                                                                                 
                                                                      
        test_user_labels = {
            int(uid): _test_user_label(index)
            for index, uid in enumerate(population)
        }

        archive_labels_pending: list[str] = []
        archive_wakeup = asyncio.Event()
        archive_flush_lock = asyncio.Lock()
        archive_stop = asyncio.Event()

        async def _flush_archive_test_users(force: bool = False) -> None:
            nonlocal archive_labels_pending
            async with archive_flush_lock:
                if not archive_labels_pending:
                    return
                if not force and len(archive_labels_pending) < 5:
                    return
                labels = list(dict.fromkeys(archive_labels_pending))
                archive_labels_pending.clear()
            try:
                if record_test_users is not None:
                    await record_test_users(str(reveal["reveal_id"]), labels)
                    async with result_lock:
                        stats.archive_test_users_recorded += len(labels)
                elif history_store is not None:
                                                                                   
                                                                                    
                                                                                    
                                                     
                    reverse = {label: uid for uid, label in test_user_labels.items()}
                    for label in labels:
                        uid = reverse.get(label)
                        if uid is not None and hasattr(history_store, "record_delivery"):
                            await history_store.record_delivery(
                                str(reveal["reveal_id"]), int(uid), None
                            )
                            async with result_lock:
                                stats.archive_test_users_recorded += 1
            except Exception:
                log.exception("Could not persist stress-test users to reveal archive")

        async def archive_worker() -> None:
            while not archive_stop.is_set():
                try:
                    await asyncio.wait_for(archive_wakeup.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                archive_wakeup.clear()
                await _flush_archive_test_users(force=False)
            await _flush_archive_test_users(force=True)

        archive_worker_task = asyncio.create_task(archive_worker())

        inflight = 0
        wall_start = time.perf_counter()

        async def sample_memory(stop: asyncio.Event) -> None:
            while not stop.is_set():
                parent = _parent_rss()
                workers = _worker_rss(get_worker_procs())
                combined = parent + workers

                async with result_lock:
                    stats.peak_parent_rss = max(stats.peak_parent_rss, parent)
                    stats.peak_worker_rss = max(stats.peak_worker_rss, workers)
                    stats.peak_combined_rss = max(stats.peak_combined_rss, combined)
                    stats.samples += 1
                    stats.peak_cache_bytes = max(
                        stats.peak_cache_bytes,
                        _cache_size(cache_dir),
                    )

                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.10)
                except asyncio.TimeoutError:
                    pass

        stop_sampling = asyncio.Event()
        sampler_task = asyncio.create_task(sample_memory(stop_sampling))

        async def run_view(arrival_delay: float, user_id: int) -> None:
            nonlocal inflight

            if not await _wait_until(arrival_delay, wall_start, hard_deadline):
                return

            request_start = time.perf_counter()

            async with result_lock:
                stats.started += 1
                inflight += 1
                stats.max_inflight = max(stats.max_inflight, inflight)

            try:
                                                                                  
                                                                                    
                                                                                   
                                                     
                preexisting = False
                if cache_path_for is not None:
                    try:
                        expected = cache_path_for(reveal, int(user_id), target_bytes)
                        preexisting = expected.is_file() and expected.stat().st_size > 0
                    except (OSError, TypeError, ValueError):
                        preexisting = False

                build_started = time.perf_counter()
                path, build_info = await build_personalized_reveal(
                    reveal,
                    int(user_id),
                    video_target_bytes=target_bytes,
                    track_status=False,
                )
                build_elapsed = time.perf_counter() - build_started

                path = Path(path)
                if not path.is_file():
                    raise RuntimeError("Personalized build returned a missing file.")

                file_size = path.stat().st_size
                if file_size <= 0:
                    raise RuntimeError("Personalized build returned an empty file.")

                                                                                  
                                                                                 
                bytes_read = await _read_file_streaming(path)
                if bytes_read != file_size:
                    raise RuntimeError(
                        f"Generated file changed while reading "
                        f"({bytes_read} read vs {file_size} expected)."
                    )

                request_elapsed = time.perf_counter() - request_start
                cached = build_info is None or preexisting

                async with result_lock:
                    stats.completed += 1
                    if cached:
                        stats.cache_hits += 1
                        generation_seconds = None
                    else:
                        stats.real_builds += 1
                        generation_seconds = (
                            float(build_info.get("seconds"))
                            if build_info and build_info.get("seconds") is not None
                            else build_elapsed
                        )

                    results.append(
                        StressResult(
                            user_id=int(user_id),
                            ok=True,
                            cached=cached,
                            request_seconds=request_elapsed,
                            generation_seconds=generation_seconds,
                            file_bytes=file_size,
                            path=path,
                            build_info=build_info,
                        )
                    )
                    created_paths.add(path)
                    stats.peak_created_file_bytes += file_size

                if archive_test_users:
                    label = test_user_labels.get(int(user_id))
                    if label:
                        archive_labels_pending.append(label)
                        archive_wakeup.set()
                        if len(archive_labels_pending) >= 5:
                            await _flush_archive_test_users(force=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with result_lock:
                    stats.failed += 1
                    stats.errors[type(exc).__name__] += 1
                    results.append(
                        StressResult(
                            user_id=int(user_id),
                            ok=False,
                            request_seconds=time.perf_counter() - request_start,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            finally:
                async with result_lock:
                    inflight = max(0, inflight - 1)

                                                                                  
        progress_stop = asyncio.Event()

        async def progress_loop() -> None:
            while not progress_stop.is_set():
                async with result_lock:
                    done = stats.completed + stats.failed
                    text = (
                        f"Stress test running — {done}/{stats.planned} finished; "
                        f"{stats.real_builds} builds; {stats.cache_hits} cache hits; "
                        f"{stats.failed} failures; peak RSS "
                        f"{_mb(stats.peak_combined_rss)}; "
                        f"peak cache {_mb(stats.peak_cache_bytes)}. "
                        f"Real /reveal requests share this same queue."
                    )
                try:
                    await interaction.edit_original_response(content=text)
                except discord.HTTPException:
                    pass

                try:
                    await asyncio.wait_for(progress_stop.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

        progress_task = asyncio.create_task(progress_loop())

        view_tasks = [
            asyncio.create_task(run_view(delay, user_id))
            for delay, user_id in schedule
        ]

        try:
            remaining = max(0.1, hard_deadline - time.monotonic())
            await asyncio.wait_for(
                asyncio.gather(*view_tasks),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            cancelled = 0
            for task in view_tasks:
                if not task.done():
                    cancelled += 1
                    task.cancel()
            stats.cancelled_by_deadline += cancelled
            await asyncio.gather(*view_tasks, return_exceptions=True)

                                                                           
        archive_stop.set()
        archive_wakeup.set()
        try:
            await asyncio.wait_for(archive_worker_task, timeout=10.0)
        except asyncio.TimeoutError:
            archive_worker_task.cancel()
            await asyncio.gather(archive_worker_task, return_exceptions=True)

                                                                         
        trace_candidates = [
            result
            for result in results
            if result.ok and result.path and result.path.is_file()
        ]
                                                                                          
        trace_candidates.sort(
            key=lambda result: (
                result.cached,
                -(result.generation_seconds or 0.0),
            )
        )
        trace_candidates = trace_candidates[:int(trace_jobs)]
        stats.trace_planned = len(trace_candidates)

        async def run_trace(result: StressResult) -> None:
            if time.monotonic() >= hard_deadline or not result.path:
                return

            try:
                if reveal["kind"] == "image":
                    kind = "image"
                    kwargs = {
                        "max_height": int(max_video_height),
                        "start_frame": 0,
                        "time_budget": float(trace_image_budget),
                    }
                else:
                    if not result.build_info:
                        return
                    height = result.build_info.get("height")
                    fps = result.build_info.get("fps")
                    if not height or not fps:
                        return
                                                                                     
                                                                                     
                                                                                      
                                                                                       
                                    
                    kind = "video"
                    kwargs = {
                        "max_height": int(height),
                        "start_frame": 0,
                        "delivery_height": int(height),
                        "delivery_fps": float(fps),
                        "time_budget": float(trace_video_budget),
                    }

                async with process_sem():
                    decoded = await run_worker(
                        "extract",
                        float(kwargs["time_budget"]) + 30.0,
                        orig=str(reveal["path"]),
                        leak=str(result.path),
                        kind=kind,
                        reveal_id=reveal["reveal_id"],
                        kwargs=kwargs,
                    )

                async with result_lock:
                    stats.trace_completed += 1
                    if decoded.get("valid"):
                        stats.trace_valid += 1
                    else:
                        stats.trace_failed += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with result_lock:
                    stats.trace_completed += 1
                    stats.trace_failed += 1
                    stats.trace_errors.append(
                        f"{type(exc).__name__}: {exc}"
                    )

        trace_tasks = [
            asyncio.create_task(run_trace(result))
            for result in trace_candidates
        ]
        if trace_tasks:
            try:
                remaining = max(0.1, hard_deadline - time.monotonic())
                await asyncio.wait_for(
                    asyncio.gather(*trace_tasks),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                for task in trace_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*trace_tasks, return_exceptions=True)

        stats.wall_seconds = time.perf_counter() - wall_start

        progress_stop.set()
        stop_sampling.set()
        progress_task.cancel()
        sampler_task.cancel()
        await asyncio.gather(
            progress_task,
            sampler_task,
            return_exceptions=True,
        )

                                                                       
        successful = [result for result in results if result.ok]
        load_times = [result.request_seconds for result in successful]
        build_times = [
            result.generation_seconds
            for result in successful
            if result.generation_seconds is not None
        ]
        file_sizes = [result.file_bytes for result in successful]

        total_created_bytes = sum(file_sizes)
        average_file_bytes = (
            total_created_bytes / len(file_sizes)
            if file_sizes else 0.0
        )

        report = {
            "status": (
                "PASS"
                if stats.failed == 0 and stats.cancelled_by_deadline == 0
                else "DEGRADED"
            ),
            "reveal_id": reveal["reveal_id"],
            "kind": reveal["kind"],
            "population": {
                "simulated_members": members,
                "simulated_viewers": viewers,
                "test_user_label_format": "test1, test2, ...",
                "stagger_seconds": float(stagger_seconds),
                "repeat_ratio": float(repeat_ratio),
                "archive_test_users": bool(archive_test_users),
            },
            "requests": {
                "planned": stats.planned,
                "started": stats.started,
                "completed": stats.completed,
                "failed": stats.failed,
                "cancelled_by_deadline": stats.cancelled_by_deadline,
                "real_builds": stats.real_builds,
                "cache_hits": stats.cache_hits,
                "max_inflight_coroutines": stats.max_inflight,
                "archive_test_users_recorded": stats.archive_test_users_recorded,
            },
            "load_time_seconds_per_user": {
                "average": round(statistics_mean(load_times), 3),
                "p50": round(_percentile(load_times, 0.50), 3),
                "p95": round(_percentile(load_times, 0.95), 3),
                "min": round(min(load_times), 3) if load_times else 0.0,
                "max": round(max(load_times), 3) if load_times else 0.0,
            },
            "generation_time_seconds_real_builds": {
                "average": round(statistics_mean(build_times), 3),
                "p50": round(_percentile(build_times, 0.50), 3),
                "p95": round(_percentile(build_times, 0.95), 3),
                "min": round(min(build_times), 3) if build_times else 0.0,
                "max": round(max(build_times), 3) if build_times else 0.0,
            },
            "generated_files": {
                "successful_outputs": len(successful),
                "real_build_outputs": stats.real_builds,
                "total_bytes_read": total_created_bytes,
                "total_mib_read": round(total_created_bytes / 1048576, 2),
                "average_file_bytes": round(average_file_bytes),
                "average_file_mib": round(average_file_bytes / 1048576, 2),
                "largest_file_bytes": max(file_sizes) if file_sizes else 0,
                "largest_file_mib": round(
                    max(file_sizes) / 1048576, 2
                ) if file_sizes else 0.0,
            },
            "disk": {
                "peak_cache_bytes": stats.peak_cache_bytes,
                "peak_cache_mib": round(stats.peak_cache_bytes / 1048576, 2),
                "peak_created_file_bytes": stats.peak_created_file_bytes,
            },
            "memory": {
                "peak_parent_rss": stats.peak_parent_rss,
                "peak_parent_mib": round(stats.peak_parent_rss / 1048576, 2),
                "peak_worker_rss": stats.peak_worker_rss,
                "peak_worker_mib": round(stats.peak_worker_rss / 1048576, 2),
                "peak_combined_rss": stats.peak_combined_rss,
                "peak_combined_mib": round(stats.peak_combined_rss / 1048576, 2),
                "samples": stats.samples,
            },
            "trace": {
                "planned": stats.trace_planned,
                "completed": stats.trace_completed,
                "valid": stats.trace_valid,
                "failed": stats.trace_failed,
                "errors": stats.trace_errors[:10],
            },
            "wall_seconds": round(stats.wall_seconds, 3),
            "errors": dict(stats.errors),
            "seed": seed,
            "cleanup_requested": bool(cleanup),
        }

        report_path = tmp_dir / (
            f"stress_test_{int(time.time())}_{interaction.user.id}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            report_path = None

                                                                                         
        cleaned_files = 0
        cleaned_bytes = 0
        if cleanup:
            for path in created_paths:
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        path.unlink()
                        cleaned_files += 1
                        cleaned_bytes += size
                except OSError:
                    log.exception(
                        "Could not clean stress-test output %s",
                        path,
                    )

            try:
                reveal_cache = cache_dir / reveal["reveal_id"]
                if reveal_cache.is_dir() and not any(reveal_cache.iterdir()):
                    reveal_cache.rmdir()
            except OSError:
                pass

        report["cleanup"] = {
            "enabled": bool(cleanup),
            "files_removed": cleaned_files,
            "bytes_removed": cleaned_bytes,
            "mib_removed": round(cleaned_bytes / 1048576, 2),
        }

        if stats.peak_combined_rss >= 450 * 1048576:
            report["status"] = "MEMORY RISK"
        elif stats.wall_seconds >= float(duration_minutes) * 60.0 - 1.0:
            report["status"] = "DEADLINE HIT"

        if report_path:
            try:
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except OSError:
                pass

        status = str(report["status"])
        summary = (
            f"**Stress test: {status}**\n"
            f"Reveal: `{reveal['reveal_id']}` ({reveal['kind']})\n"
            f"Requests: {stats.completed}/{stats.planned}; "
            f"{stats.real_builds} real builds; {stats.cache_hits} cache hits; "
            f"{stats.failed} failures; {stats.cancelled_by_deadline} cancelled by deadline.\n"
            f"Average user load time: "
            f"{statistics_mean(load_times):.2f}s "
            f"(p95 { _percentile(load_times, 0.95):.2f}s).\n"
            f"Average real generation time: "
            f"{statistics_mean(build_times):.2f}s "
            f"(p95 { _percentile(build_times, 0.95):.2f}s).\n"
            f"Generated/read: {total_created_bytes / 1048576:.1f} MiB across "
            f"{len(successful)} completed outputs.\n"
            f"Peak cache: {_mb(stats.peak_cache_bytes)}.\n"
            f"Peak RSS: parent {_mb(stats.peak_parent_rss)} + "
            f"workers {_mb(stats.peak_worker_rss)} = "
            f"{_mb(stats.peak_combined_rss)}.\n"
            f"Elapsed: {stats.wall_seconds:.1f}s."
            f"\nArchive test users recorded: {stats.archive_test_users_recorded}."
        )

        if stats.trace_planned:
            summary += (
                f"\nTrace: {stats.trace_valid}/{stats.trace_completed} "
                f"generated-output traces decoded successfully."
            )

        if stats.errors:
            summary += "\nErrors: " + ", ".join(
                f"{name}={count}"
                for name, count in stats.errors.most_common()
            )

        try:
            if report_path and report_path.is_file():
                report_file = discord.File(
                    report_path,
                    filename=report_path.name,
                    spoiler=False,
                )
                try:
                    await interaction.followup.send(
                        content=summary,
                        file=report_file,
                        ephemeral=True,
                    )
                finally:
                    report_file.close()
                try:
                    report_path.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                await interaction.followup.send(
                    content=summary,
                    ephemeral=True,
                )
        except discord.HTTPException:
            log.exception("Could not deliver stress-test report.")
            if report_path:
                try:
                    report_path.unlink(missing_ok=True)
                except OSError:
                    pass
