"""Isolated subprocess entry point for CPU/memory-heavy watermark operations.

The Discord bot sends one JSON request on stdin. Results are emitted as JSON on
stdout; diagnostics and tracebacks remain on stderr for the parent process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import watermark as wm


def main() -> None:
    request = json.loads(sys.stdin.read())
    operation = request["op"]
    key = bytes.fromhex(request["key"])
    args = request["args"]

    if operation == "image":
        result = wm.render_image_delivery(
            Path(args["src"]), Path(args["dst"]), args["user_id"], key,
            args["reveal_id"], amp=args["amp"], target_bytes=args["target_bytes"],
        )
    elif operation == "video":
        result = wm.embed_video(
            Path(args["src"]), Path(args["dst"]), args["user_id"], key,
            args["reveal_id"], amp=args["amp"], preset=args["preset"],
            max_seconds=args["max_seconds"], max_height=args["max_height"],
            target_bytes=args["target_bytes"], audio_kbps=args["audio_kbps"],
        )
    elif operation == "extract":
        result = wm.extract(
            Path(args["orig"]), Path(args["leak"]), args["kind"], key,
            args["reveal_id"], **args["kwargs"],
        )
    elif operation == "image_check":
        width, height = wm._load_rgb(Path(args["path"])).size
        result = {"width": width, "height": height}
    elif operation == "video_info":
        result = wm.video_info(Path(args["path"]), args["max_height"])
    else:
        raise ValueError(f"Unknown worker operation: {operation!r}")

    # Keep a leading newline to separate JSON from any incidental library stdout.
    sys.stdout.write("\n" + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
