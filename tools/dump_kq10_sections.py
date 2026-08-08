# FILE: tools/dump_kq10_sections.py
# Standalone check: extracts the narrative sections from a folder of 10-K/10-Q
# filings and dumps what was captured. No embeddings, no model loading.
#
# Usage:
#   python tools/dump_kq10_sections.py <folder_with_html_filings> [max_files]

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import (
    KQ10HtmlStructuredBlockExtractor,
)

PREVIEW_CHARS = 300
MIN_BLOCK_CHARS = 200  # same floor the tagger uses to discard 'None.' style items


def dump_folder(folder: str, max_files: int):
    extractor = KQ10HtmlStructuredBlockExtractor()

    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".html", ".htm")))
    files = files[:max_files]

    if not files:
        print(f"No html filings found in {folder}")
        return

    out_path = os.path.join(folder, "sections_dump.txt")

    with open(out_path, "w", encoding="utf-8") as out:
        for file_name in files:
            path = os.path.join(folder, file_name)
            report_type = extractor.resolve_report_type(file_name)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()

            try:
                blocks = extractor.extract_blocks(html, report_type)
            except Exception as e:
                line = f"[{file_name}] EXTRACTION FAILED: {e}"
                print(line)
                out.write(line + "\n\n")
                continue

            kept = {k: v for k, v in blocks.items() if len(v) >= MIN_BLOCK_CHARS}
            dropped = [k for k in blocks if k not in kept]

            header = (f"\n=== {file_name} | type={report_type} | "
                      f"sections={len(kept)} | too_short={len(dropped)}")
            print(header)
            out.write(header + "\n")

            for label, text in kept.items():
                words = len(text.split())
                summary = f"  {label} | chars={len(text)} | words={words}"
                print(summary)
                out.write(summary + "\n")
                out.write(f"    HEAD: {text[:PREVIEW_CHARS]}\n")
                out.write(f"    TAIL: {text[-PREVIEW_CHARS:]}\n")

            for label in dropped:
                note = f"  {label} | DROPPED (under {MIN_BLOCK_CHARS} chars)"
                print(note)
                out.write(note + "\n")

    print(f"\nFull dump written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/dump_kq10_sections.py <folder_with_html_filings> [max_files]")
        sys.exit(1)

    target_folder = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    dump_folder(target_folder, limit)
