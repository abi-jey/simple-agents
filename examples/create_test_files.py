#!/usr/bin/env python3
"""
Utility script to create test image files for examples.

Audio files should be created using Audacity:
1. Open Audacity and record or import audio
2. Export as WAV (for direct audio to GPT-4o) or OGG (for STT with Gemini)
3. Save to examples/test_audio.wav or examples/test_audio.ogg

Usage:
------
Run this script to generate test image:
    poetry run python examples/create_test_files.py

Then run the image example:
    poetry run python examples/image_input_azure_v1.py
"""

from pathlib import Path

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

console = None
try:
    from rich.console import Console

    console = Console()
except ImportError:
    pass


def log(message: str, style: str = "") -> None:
    """Print message with optional style."""
    if console:
        if style:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            console.print(message)
    else:
        print(message)


def create_test_image_png(output_path: Path, size: tuple[int, int] = (400, 300)) -> None:
    """Create a simple test image with shapes and text."""
    if not HAS_PIL:
        log("PIL/Pillow not installed. Creating minimal PNG...", "yellow")
        create_minimal_png(output_path)
        return

    img = PILImage.new("RGB", size, color=(73, 109, 137))
    draw = ImageDraw.Draw(img)

    draw.rectangle([50, 50, 150, 150], fill=(255, 100, 100), outline=(200, 50, 50))
    draw.ellipse([200, 50, 350, 200], fill=(100, 255, 100), outline=(50, 200, 50))
    draw.polygon([(100, 200), (50, 280), (150, 280)], fill=(100, 100, 255))

    try:
        draw.text((120, 100), "Test Image", fill=(255, 255, 255))
    except Exception:
        pass

    img.save(output_path)
    log(f"Created: {output_path}", "green")


def create_minimal_png(output_path: Path) -> None:
    """Create a minimal valid PNG file without PIL."""
    import math
    import struct
    import zlib

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk + crc.to_bytes(4, "big")

    width, height = 100, 100

    signature = b"\x89PNG\r\n\x1a\n"

    ihdr_data = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    ihdr = png_chunk(b"IHDR", ihdr_data)

    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"
        raw_data += b"\xff\x00\x00" * width

    compressed = zlib.compress(raw_data)
    idat = png_chunk(b"IDAT", compressed)

    iend = png_chunk(b"IEND", b"")

    output_path.write_bytes(signature + ihdr + idat + iend)
    log(f"Created minimal PNG: {output_path}", "green")


def main() -> None:
    """Create test files."""
    log("=" * 60, "bold blue")
    log("Creating Test Image for nagents Examples", "bold blue")
    log("=" * 60, "bold blue")
    log("")

    output_dir = Path("examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Creating image file...", "bold")
    create_test_image_png(output_dir / "test_image.png")

    log("")
    log("=" * 60, "bold green")
    log("Done! Test image created.", "bold green")
    log("=" * 60, "bold green")
    log("")
    log("For audio files, use Audacity:", "cyan")
    log("  1. Open Audacity, record or import audio")
    log("  2. Export as WAV -> examples/test_audio.wav (for GPT-4o)")
    log("  3. Export as OGG -> examples/test_audio.ogg (for STT)")
    log("")
    log("Then run examples:", "cyan")
    log("  poetry run python examples/audio_input_direct.py", "dim")
    log("  poetry run python examples/audio_input_stt.py", "dim")
    log("  poetry run python examples/image_input_azure_v1.py", "dim")


if __name__ == "__main__":
    main()
