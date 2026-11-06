"""Generates the official v3 presentation deck (HW2_G04_presentation.pptx)

Requirements:
- Exactly 11 slides
- Total rehearsal budget: exactly 480 seconds (sum across all slides)
- Speaker notes on every slide
- Native table on Slide 7 (benchmark results)
- High-res images embedded on image slides (eda, pipeline, pr_roc, tradeoffs, interpretability)
- 16:9 widescreen layout
"""

import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / "Presentation" / "content_v3.json"
OUT_PPTX_DIR = ROOT / "Presentation" / "HW2_G04_presentation.pptx"
OUT_PPTX_ROOT = ROOT / "HW2_G04_presentation.pptx"

# Color Palette
NAVY = RGBColor(15, 77, 146)       # #0F4D92
INK = RGBColor(24, 43, 64)          # #182B40
GRAY = RGBColor(91, 102, 116)       # #5B6674
LIGHT_BG = RGBColor(247, 249, 251)  # #F7F9FB
HERO_BG = RGBColor(234, 241, 248)   # #EAF1F8
WHITE = RGBColor(255, 255, 255)
ACCENT_BLUE = RGBColor(37, 99, 235) # #2563EB


def create_deck():
    with open(CONTENT_PATH, "r", encoding="utf-8") as f:
        slides_data = json.load(f)

    assert len(slides_data) == 11, f"Expected 11 slides, got {len(slides_data)}"
    total_sec = sum(s["seconds"] for s in slides_data)
    assert total_sec == 480, f"Expected 480 seconds, got {total_sec}"

    prs = Presentation()
    # 16:9 widescreen layout: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for idx, slide_info in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)

        # 1. Top accent line
        accent_bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12)
        )
        accent_bar.fill.solid()
        accent_bar.fill.fore_color.rgb = NAVY
        accent_bar.line.fill.background()

        # 2. Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.85))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = slide_info["title"]
        p_title.font.name = "Arial"
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

        # 3. Content layout depending on type
        if idx == 0:
            # Title slide layout
            content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.0))
            tf_content = content_box.text_frame
            tf_content.word_wrap = True
            for j, line in enumerate(slide_info["lines"]):
                p = tf_content.add_paragraph() if j > 0 else tf_content.paragraphs[0]
                p.text = line
                p.font.name = "Arial"
                if j == 0:
                    p.font.size = Pt(22)
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_BLUE
                    p.space_after = Pt(14)
                elif j == 1:
                    p.font.size = Pt(18)
                    p.font.bold = True
                    p.font.color.rgb = GRAY
                    p.space_after = Pt(24)
                else:
                    p.font.size = Pt(17)
                    p.font.color.rgb = INK
                    p.space_after = Pt(12)

        elif slide_info.get("table"):
            # Table layout (Slide 7)
            table_data = slide_info["table"]
            rows = len(table_data)
            cols = len(table_data[0])
            table_shape = slide.shapes.add_table(
                rows, cols, Inches(0.8), Inches(1.4), Inches(11.733), Inches(3.6)
            )
            table = table_shape.table
            col_widths = [Inches(2.1), Inches(1.6), Inches(1.7), Inches(1.7), Inches(1.6), Inches(1.6), Inches(1.433)]
            for c_idx, w in enumerate(col_widths):
                table.columns[c_idx].width = w

            for r_idx, row in enumerate(table_data):
                is_hero = "CM-BAE" in row[0]
                for c_idx, val in enumerate(row):
                    cell = table.cell(r_idx, c_idx)
                    cell.text = val
                    p = cell.text_frame.paragraphs[0]
                    p.font.name = "Arial"
                    if r_idx == 0:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = NAVY
                        p.font.bold = True
                        p.font.color.rgb = WHITE
                        p.font.size = Pt(12)
                    else:
                        cell.fill.solid()
                        if is_hero:
                            cell.fill.fore_color.rgb = HERO_BG
                            p.font.bold = True
                            p.font.color.rgb = NAVY
                        else:
                            cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
                            p.font.color.rgb = INK
                        p.font.size = Pt(11)

            # Bullet summary below table
            bullet_box = slide.shapes.add_textbox(Inches(0.8), Inches(5.2), Inches(11.733), Inches(1.6))
            tf_bullets = bullet_box.text_frame
            tf_bullets.word_wrap = True
            for j, line in enumerate(slide_info["lines"]):
                p = tf_bullets.add_paragraph() if j > 0 else tf_bullets.paragraphs[0]
                p.text = f"•  {line}"
                p.font.name = "Arial"
                p.font.size = Pt(14)
                p.font.color.rgb = NAVY if j == 0 else INK
                p.font.bold = (j == 0)
                p.space_after = Pt(4)

        elif slide_info.get("image"):
            # Image + Bullets split layout
            img_path = ROOT / slide_info["image"]
            assert img_path.exists(), f"Image not found: {img_path}"

            # Left side: Image (width 6.2 inches, height up to 5.0 inches)
            slide.shapes.add_picture(
                str(img_path), Inches(0.8), Inches(1.45), Inches(6.0), Inches(5.1)
            )

            # Right side: Bullets (width 5.5 inches)
            text_box = slide.shapes.add_textbox(Inches(7.1), Inches(1.45), Inches(5.4), Inches(5.1))
            tf_text = text_box.text_frame
            tf_text.word_wrap = True
            for j, line in enumerate(slide_info["lines"]):
                p = tf_text.add_paragraph() if j > 0 else tf_text.paragraphs[0]
                p.text = f"•  {line}"
                p.font.name = "Arial"
                p.font.size = Pt(15)
                p.font.color.rgb = INK
                p.space_after = Pt(10)

        else:
            # Standard bullets layout
            content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.0))
            tf_content = content_box.text_frame
            tf_content.word_wrap = True
            for j, line in enumerate(slide_info["lines"]):
                p = tf_content.add_paragraph() if j > 0 else tf_content.paragraphs[0]
                p.text = f"•  {line}"
                p.font.name = "Arial"
                p.font.size = Pt(17)
                p.font.color.rgb = INK
                p.space_after = Pt(12)

        # 4. Footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.9), Inches(11.733), Inches(0.4))
        tf_footer = footer_box.text_frame
        p_footer = tf_footer.paragraphs[0]
        p_footer.text = f"Team G04 · NYCU Data Mining (Fall 2026)  |  Slide {idx + 1} of 11"
        p_footer.font.name = "Arial"
        p_footer.font.size = Pt(11)
        p_footer.font.color.rgb = GRAY

        # 5. Speaker Notes
        notes_slide = slide.notes_slide
        tf_notes = notes_slide.notes_text_frame
        tf_notes.text = f"Rehearsal budget: {slide_info['seconds']} seconds.\n\n{slide_info['notes']}"

    # Save to Presentation/ and root
    prs.save(str(OUT_PPTX_DIR))
    prs.save(str(OUT_PPTX_ROOT))
    print(f"Saved {OUT_PPTX_DIR}")
    print(f"Saved {OUT_PPTX_ROOT}")


if __name__ == "__main__":
    create_deck()
