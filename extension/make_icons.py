"""Generates the extension's PNG icons (16/48/128) with a purple->blue gradient
and a white sparkle. Run once: python extension/make_icons.py"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "icons")
os.makedirs(OUT, exist_ok=True)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c1, c2 = (123, 47, 247), (74, 108, 247)  # purple -> blue
    # rounded-square gradient background
    for y in range(size):
        d.line([(0, y), (size, y)], fill=lerp(c1, c2, y / max(1, size - 1)) + (255,))
    # mask to rounded corners
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    r = max(3, size // 5)
    md.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    img.putalpha(mask)

    # a simple 4-point sparkle in the center
    d = ImageDraw.Draw(img)
    cx = cy = size / 2
    s = size * 0.30
    thin = max(1, size // 22)
    white = (255, 255, 255, 235)
    d.polygon([(cx, cy - s), (cx + thin, cy - thin), (cx, cy), (cx - thin, cy - thin)], fill=white)
    d.polygon([(cx, cy + s), (cx + thin, cy + thin), (cx, cy), (cx - thin, cy + thin)], fill=white)
    d.polygon([(cx - s, cy), (cx - thin, cy + thin), (cx, cy), (cx - thin, cy - thin)], fill=white)
    d.polygon([(cx + s, cy), (cx + thin, cy + thin), (cx, cy), (cx + thin, cy - thin)], fill=white)
    return img


for sz in (16, 48, 128):
    make(sz).save(os.path.join(OUT, f"icon{sz}.png"))
    print("wrote", f"icon{sz}.png")
