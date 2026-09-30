import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import watermark as wm

load_dotenv()

p = argparse.ArgumentParser()
p.add_argument("--original", required=True, type=Path)
p.add_argument("--leak", required=True, type=Path)
p.add_argument("--reveal-id", required=True)
p.add_argument("--start-frame", type=int, default=0)
p.add_argument("--image-cell", type=int, default=int(os.getenv("WM_IMAGE_CELL", "4")))
p.add_argument("--video-cell", type=int, default=int(os.getenv("WM_VIDEO_CELL", "8")))
p.add_argument("--max-height", type=int, default=int(os.getenv("MAX_VIDEO_HEIGHT", "720")))
args = p.parse_args()

secret = os.getenv("WATERMARK_SECRET")
if not secret:
    raise SystemExit("WATERMARK_SECRET is not set.")

image_exts = {".jpg", ".jpeg", ".png", ".webp"}
video_exts = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
orig_is_image = args.original.suffix.lower() in image_exts
leak_is_image = args.leak.suffix.lower() in image_exts

if orig_is_image:
    kind = "image"
elif leak_is_image:
    kind = "video_frame"
elif args.leak.suffix.lower() in video_exts:
    kind = "video"
else:
    raise SystemExit("Leak must be an image or video.")

res = wm.extract(
    args.original, args.leak, kind, secret.encode(), args.reveal_id,
    image_cell=args.image_cell, video_cell=args.video_cell,
    max_height=args.max_height, start_frame=args.start_frame,
)

if res["valid"]:
    print(f"MATCH: Discord user ID {res['user_id']} (CRC verified) {res}")
else:
    print(f"No valid watermark found. {res}")
