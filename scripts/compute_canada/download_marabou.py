"""Fetch the pinned Linux CPython 3.11 wheel and check its published SHA-256."""

import hashlib
from pathlib import Path
import urllib.request


def main():
    import json
    with urllib.request.urlopen("https://pypi.org/pypi/maraboupy/2.0.0/json", timeout=60) as response:
        metadata = json.load(response)
    wheel = next(item for item in metadata["urls"] if "cp311-cp311-manylinux" in item["filename"])
    expected = "60ed32314f2251c8a12400c72fb8b0a8ff191fc644b96b7ea2cc95c437749922"
    with urllib.request.urlopen(wheel["url"], timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError("Marabou wheel SHA-256 mismatch")
    path = Path(wheel["filename"])
    path.write_bytes(content)
    print(f"Verified and saved {path}")


if __name__ == "__main__":
    main()
