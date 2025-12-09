#!/usr/bin/env python3
import sys
from pathlib import Path

TARGET_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp"}

def find_non_ascii(line):
    """Return list of (index, char) for all non-ASCII chars in a line."""
    result = []
    for i, ch in enumerate(line):
        if ord(ch) > 127:
            result.append((i, ch))
    return result


def check_file(path: Path) -> bool:
    ok = True
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            non_ascii = find_non_ascii(line)
            if non_ascii:
                ok = False
                print(f"\n❌ {path}:{lineno}: non-ASCII characters detected")

                # Print the full line
                print(f"   Line content:")
                print(f"   {line.rstrip()}")

                # Underline the exact non-ASCII characters
                underline = [" " for _ in line.rstrip("\n")]
                for idx, ch in non_ascii:
                    if idx < len(underline):
                        underline[idx] = "^"
                print(f"   {' '.join(underline)}")

                # Print what characters exactly
                chars = ", ".join(f"'{ch}' (U+{ord(ch):04X})" for _, ch in non_ascii)
                print(f"   Offending chars: {chars}")

    return ok


def main(files):
    ok = True
    for f in files:
        p = Path(f)
        if p.suffix.lower() in TARGET_SUFFIXES and p.exists():
            if not check_file(p):
                ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_ascii.py <files...>")
        sys.exit(1)
    main(sys.argv[1:])

