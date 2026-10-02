from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import watermark as wm

load_dotenv()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decode a RevealBot keyed watermark from an image, video frame, or video."
    )
    p.add_argument("--original", required=True, type=Path, help="Saved original reveal media")
    p.add_argument("--leak", required=True, type=Path, help="Leaked image/frame/video")
    p.add_argument("--reveal-id", required=True, help="Reveal ID used during embedding")
    p.add_argument(
        "--kind",
        choices=("auto", "image", "video_frame", "video"),
        default="auto",
        help="Override automatic leak type detection",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--image-cell", type=int, default=int(os.getenv("WM_IMAGE_CELL", "4")))
    p.add_argument("--video-cell", type=int, default=int(os.getenv("WM_VIDEO_CELL", "8")))
    p.add_argument("--max-height", type=int, default=int(os.getenv("MAX_VIDEO_HEIGHT", "1080")))
    p.add_argument(
        "--delivery-height",
        type=int,
        default=None,
        help="Height of the delivered video (the 'height' returned by embed_video). "
             "Strongly recommended for video / video_frame traces.",
    )
    p.add_argument(
        "--delivery-fps",
        type=float,
        default=None,
        help="FPS of the delivered video (the 'fps' returned by embed_video).",
    )
    p.add_argument(
        "--time-budget",
        type=float,
        default=float(os.getenv("TRACE_TIME_BUDGET", str(wm.DEFAULT_TIME_BUDGET))),
        help="Max seconds to spend on a video trace",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Log which stage fails")
    p.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    return p


def infer_kind(original: Path, leak: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    orig_is_image = original.suffix.lower() in IMAGE_EXTS
    leak_is_image = leak.suffix.lower() in IMAGE_EXTS
    if orig_is_image:
        return "image"
    if leak_is_image:
        return "video_frame"
    if leak.suffix.lower() in VIDEO_EXTS:
        return "video"
    raise SystemExit("Leak must be an image or video.")


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # keep stdout clean for --json
    )

    secret = os.getenv("WATERMARK_SECRET")
    if not secret or len(secret) < 16:
        raise SystemExit("WATERMARK_SECRET is missing or too short.")

    if not args.original.is_file():
        raise SystemExit(f"Original file does not exist: {args.original}")
    if not args.leak.is_file():
        raise SystemExit(f"Leak file does not exist: {args.leak}")
    if args.start_frame < 0:
        raise SystemExit("--start-frame must be >= 0")
    if args.max_height < 144:
        raise SystemExit("--max-height must be >= 144")
    if args.delivery_height is not None and args.delivery_height < 144:
        raise SystemExit("--delivery-height must be >= 144")
    if args.delivery_fps is not None and args.delivery_fps <= 0:
        raise SystemExit("--delivery-fps must be > 0")
    if args.time_budget <= 0:
        raise SystemExit("--time-budget must be > 0")

    kind = infer_kind(args.original, args.leak, args.kind)

    if kind in ("video", "video_frame") and args.delivery_height is None:
        print(
            "Warning: no --delivery-height given; trying every height in the ladder (slower, "
            "less reliable). Pass the height/fps stored when the video was embedded.",
            file=sys.stderr,
        )

    try:
        res = wm.extract(
            args.original,
            args.leak,
            kind,
            secret.encode("utf-8"),
            args.reveal_id,
            image_cell=args.image_cell,
            video_cell=args.video_cell,
            max_height=args.max_height,
            start_frame=args.start_frame,
            delivery_height=args.delivery_height,
            delivery_fps=args.delivery_fps,
            time_budget=args.time_budget,
        )
    except Exception as exc:
        if args.as_json:
            print(json.dumps({"valid": False, "error": type(exc).__name__, "message": str(exc)}))
        else:
            print(f"Trace failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    result = dict(res)
    result["valid"] = bool(result.get("valid", result.get("ok", False)))

    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    elif result["valid"]:
        print(
            f"MATCH: Discord user ID {result['user_id']} "
            f"(CRC verified) {result}"
        )
    else:
        print(f"No valid watermark found. {result}")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
