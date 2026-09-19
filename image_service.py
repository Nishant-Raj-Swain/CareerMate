import io
from PIL import Image, ImageDraw, ImageFont

def generate_roadmap_card(topic: str, sections: dict) -> bytes:
    width, height = 1080, 1350
    bg_color = (15, 15, 26)
    accent_color = (168, 85, 247)
    title_color = (250, 204, 21)
    text_color = (255, 255, 255)
    
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 44)
        header_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        title_font = header_font = body_font = ImageFont.load_default()

    draw.text((width // 2, 60), f"{topic.upper()} ROADMAP", font=title_font, fill=title_color, anchor="mm")
    draw.line([(60, 110), (width - 60, 110)], fill=accent_color, width=3)

    col_width = (width - 160) // 2
    x_positions = [60, 60 + col_width + 40]
    y_start = 140
    
    for idx, (category, items) in enumerate(sections.items()):
        if idx >= 6:
            break
        col = idx % 2
        row = idx // 2
        x = x_positions[col]
        y = y_start + (row * 380)

        draw.text((x, y), category, font=header_font, fill=accent_color)
        item_y = y + 50
        for item in items[:7]:
            draw.text((x + 10, item_y), f"• {item}", font=body_font, fill=text_color)
            item_y += 36

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()