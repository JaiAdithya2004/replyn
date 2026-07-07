"""
build_extension_zip.py
----------------------
Packages the extension/ folder into site/replyn-extension.zip so the landing
page's "Download Extension" button can serve it directly.

The zip contains the extension at its root (manifest.json at top level) so a user
can unzip and "Load unpacked" the resulting folder immediately.

Run:  python build_extension_zip.py
Re-run whenever the extension changes.
"""

import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = os.path.join(HERE, "extension")
OUT_ZIP = os.path.join(HERE, "replyn-extension.zip")

# Files we don't want in the shipped zip.
SKIP_NAMES = {".DS_Store", "make_icons.py"}
SKIP_DIRS = {"__pycache__"}


def main():
    os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)
    count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(EXT_DIR):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                if name in SKIP_NAMES:
                    continue
                full = os.path.join(root, name)
                # Store paths under a top-level "replyn-extension/" folder so the
                # unzipped result is a single, clearly-named folder.
                rel = os.path.relpath(full, EXT_DIR)
                arc = os.path.join("replyn-extension", rel)
                zf.write(full, arc)
                count += 1
    size_kb = os.path.getsize(OUT_ZIP) / 1024
    print(f"Wrote {OUT_ZIP} ({count} files, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
