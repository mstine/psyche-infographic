#!/usr/bin/env python3
"""
Gemini Nano Banana 2 image generator for the Psyche sanctuary build.

Reads scripts/sanctuary-prompts.json and generates JPEG outputs to
images/sanctuary/<id>-v<n>.jpg for each prompt variant. Skips files that
already exist (idempotent — safe to re-run).

Requires GEMINI_API_KEY env var. Run from the psyche-infographic repo root:

    python3 scripts/generate-images.py

Cost: roughly $0.039 per image × 24 = ~$0.94 for a full sanctuary build.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "gemini-3.1-flash-image-preview"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = REPO_ROOT / "scripts" / "sanctuary-prompts.json"
OUTPUT_DIR = REPO_ROOT / "images" / "sanctuary"

RATE_LIMIT_DELAY_SECONDS = 2


def load_reference_image(path: Path) -> dict:
    """Read an image file and return it as a Gemini API inlineData part."""
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"inlineData": {"mimeType": mime, "data": b64}}


def generate_one(parts: list, output_path: Path, api_key: str) -> tuple[int, str]:
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}") from e

    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"API error {err.get('code')} {err.get('status')}: {err.get('message','')[:300]}")

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected response shape: {json.dumps(data)[:500]}") from e

    for p in parts:
        if "inlineData" in p:
            img_bytes = base64.b64decode(p["inlineData"]["data"])
            output_path.write_bytes(img_bytes)
            return len(img_bytes), p["inlineData"].get("mimeType", "unknown")

    raise RuntimeError(f"no inlineData in response parts: {json.dumps(parts)[:500]}")


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
        return 1

    if not PROMPTS_FILE.exists():
        print(f"ERROR: prompts file not found at {PROMPTS_FILE}", file=sys.stderr)
        return 1

    with PROMPTS_FILE.open() as f:
        prompts = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = sum(len(item["variants"]) for item in prompts)
    done = 0
    skipped = 0
    failed = 0

    print(f"Generating {total} images to {OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print()

    for item in prompts:
        image_id = item["id"]
        title = item.get("title", image_id)
        print(f"== {image_id}: {title} ==")

        ref_part = None
        ref_image_rel = item.get("reference_image")
        if ref_image_rel:
            ref_image_path = REPO_ROOT / ref_image_rel
            if not ref_image_path.exists():
                print(f"  WARNING: reference_image not found at {ref_image_rel}, skipping station")
                failed += len(item["variants"])
                continue
            ref_part = load_reference_image(ref_image_path)
            print(f"  using reference: {ref_image_rel}")

        for idx, prompt in enumerate(item["variants"], start=1):
            output_path = OUTPUT_DIR / f"{image_id}-v{idx}.jpg"

            if output_path.exists():
                print(f"  v{idx}: SKIP (exists at {output_path.name})")
                skipped += 1
                continue

            parts = []
            if ref_part is not None:
                parts.append(ref_part)
            parts.append({"text": prompt})

            print(f"  v{idx}: generating... ", end="", flush=True)
            try:
                size, mime = generate_one(parts, output_path, api_key)
                print(f"OK ({size:,} bytes, {mime})")
                done += 1
                time.sleep(RATE_LIMIT_DELAY_SECONDS)
            except Exception as e:
                print(f"FAIL: {e}")
                failed += 1

    print()
    print(f"Done: {done} generated, {skipped} skipped, {failed} failed, {total} total")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
