#!/usr/bin/env python3
import sys
import unicodedata
from pathlib import Path

TARGET_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".py"}


# --- Helpers --------------------------------------------------------


# Latin ranges we still consider "English-ish" and therefore allowed.
# (You can shrink this if you want to ban accented letters too.)
LATIN_RANGES = [
    (0x0000, 0x007F),  # Basic Latin (ASCII)
    (0x00C0, 0x024F),  # Latin-1 Supplement + Latin Extended-A/B
    (0x1E00, 0x1EFF),  # Latin Extended Additional
]


def in_ranges(ch: str, ranges) -> bool:
    cp = ord(ch)
    for start, end in ranges:
        if start <= cp <= end:
            return True
    return False


def is_forbidden_char(ch: str) -> bool:
    """
    Return True if ch should be *forbidden*.

    Policy:
      - ASCII (<= 0x7F): always OK
      - Non-ASCII letters (Unicode category starting with 'L')
        that are NOT in Latin ranges: forbidden
      - Everything else (emoji, arrows, symbols, etc.): allowed
    """
    cp = ord(ch)
    if cp <= 0x7F:
        return False  # pure ASCII

    cat = unicodedata.category(ch)

    # Forbid letters that are not Latin.
    if cat.startswith("L"):  # Letter
        if in_ranges(ch, LATIN_RANGES):
            return False  # Latin letters allowed
        return True       # Non-Latin letters forbidden

    # All non-letter stuff (emoji, arrows, symbols, punctuation) is allowed.
    return False


def find_forbidden_chars(line: str):
    """Return list of (index, char) for all forbidden chars in a line."""
    result = []
    for i, ch in enumerate(line):
        if is_forbidden_char(ch):
            result.append((i, ch))
    return result


# --- Core logic -----------------------------------------------------


def check_file(path: Path) -> bool:
    ok = True
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            forbidden = find_forbidden_chars(line)
            if forbidden:
                ok = False
                print(f"\n❌ {path}:{lineno}: non-English letters detected")

                # Print the full line
                print("   Line content:")
                print(f"   {line.rstrip()}")

                # Underline the forbidden characters
                underline = [" " for _ in line.rstrip("\n")]
                for idx, ch in forbidden:
                    if idx < len(underline):
                        underline[idx] = "^"
                print(f"   {''.join(underline)}")

                # Print what characters exactly
                chars = ", ".join(
                    f"'{ch}' (U+{ord(ch):04X}) [{unicodedata.name(ch, 'UNKNOWN')}]"
                    for _, ch in forbidden
                )
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
