from typing import List, Dict
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def export_summary_pdf(output_path: str, title: str, summary: Dict, image_paths: List[str] | None = None):
    p = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    p.setFont("Helvetica-Bold", 16)
    p.drawString(2*cm, height-2*cm, title)
    p.setFont("Helvetica", 11)
    y = height - 3*cm
    avg = summary.get('avg_total')
    if avg is not None:
        p.drawString(2*cm, y, f"Average Total Score: {avg}")
        y -= 0.8*cm
    items = summary.get('items') or summary
    if isinstance(items, list):
        p.drawString(2*cm, y, f"Items: {len(items)}")
        y -= 0.8*cm
        for i, it in enumerate(items[:20]):
            s = it.get('scores', {})
            p.drawString(2*cm, y, f"[{i}] total={s.get('total')} S={s.get('S')} Q={s.get('Q')} R={s.get('R')} A={s.get('A')}")
            y -= 0.6*cm
            if y < 4*cm:
                p.showPage()
                p.setFont("Helvetica", 11)
                y = height - 3*cm
    # Add first image if provided
    if image_paths:
        try:
            img = image_paths[0]
            if img and Path(img).exists():
                p.showPage()
                p.drawString(2*cm, height-2*cm, "Sample Maze")
                p.drawImage(img, 2*cm, 4*cm, width=16*cm, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
    p.save()
