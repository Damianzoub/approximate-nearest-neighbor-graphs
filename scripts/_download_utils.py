"""Shared download utility with User-Agent spoofing and progress reporting."""

import sys
import urllib.request
from pathlib import Path

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _progress(count, block_size, total_size):
    if total_size <= 0:
        print(f"\r  {count * block_size / 1e6:.1f} MB downloaded", end="", flush=True)
        return
    pct = min(count * block_size / total_size * 100, 100)
    done = int(pct / 2)
    bar = "█" * done + "░" * (50 - done)
    mb_done = min(count * block_size, total_size) / 1e6
    mb_total = total_size / 1e6
    print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.0f}/{mb_total:.0f} MB",
          end="", flush=True)


def download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, sending a browser User-Agent."""
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", _UA)]
    urllib.request.install_opener(opener)

    print(f"  URL : {url}")
    print(f"  Dest: {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()   # newline after progress bar
