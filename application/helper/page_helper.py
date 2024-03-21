from typing import List

from PIL import Image, ImageDraw, ImageFont
import os
from .pageEntry import PageEntry


def load_page_and_draw(page_entry: PageEntry, page_recipient_ids=[], precomputing=False):

    if precomputing:
        page_entry.image_file = page_entry.image_file.replace(".jpg", ".png")

    image_path = os.path.join(page_entry.base_path, page_entry.book, page_entry.image_file)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial_narrow_7.ttf", 72)
    for line in page_entry.page_line_entries:

        if precomputing:
            draw.text((line.hpos - 75, line.vpos), str(line.line_number), (0, 0, 0), font=font)

        if line.id in page_recipient_ids:
            draw_box(draw, line, (255, 0, 0))

    return image


def build_entries(data_path, book):
    base_path = os.path.join(data_path, book)
    if not os.path.isdir(base_path):
        raise Exception(f"Path '{base_path}' not found.")
    xml_path = os.path.join(base_path, "alto")
    xml_files = os.listdir(xml_path)
    xml_files.sort()
    entries: List[PageEntry] = []
    for xml_file in xml_files:
        image_file_name = f"{xml_file.split('.')[0]}.jpg"
        pageEntry = PageEntry(data_path, xml_file, image_file_name, book)
        if len(pageEntry.page_line_entries) > 0:
            entries.append(pageEntry)
    return entries


def draw_box(draw, line, color, width=5):
    w, h = line.width, line.height
    vpos, hpos = line.vpos, line.hpos
    shape = [(hpos, vpos), (hpos + w, vpos + h)]
    draw.rectangle(shape, fill=None, outline=color, width=width)
