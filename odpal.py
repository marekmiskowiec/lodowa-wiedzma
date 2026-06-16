"""
Odpal Detector — rozpoznawanie aktywnych wzmocnień (Echo Wygnańców).
Uruchom: python odpal.py

Osobny moduł niezwiązany z drop trackerem (app.py).
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


# ── Ścieżki ──────────────────────────────────────────────────────────────────

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


TEMPLATES_DIR = resource_path("odpal/templates")

# Nazwy ikon — klucz = nazwa pliku PNG bez rozszerzenia, wartość = wyświetlana nazwa
# Pliki z odpal/templates/ są ładowane automatycznie; tutaj możesz nadać czytelne nazwy.
# Jeśli ikona nie ma wpisu tutaj, wyświetlana jest nazwa pliku.
ICON_NAMES: dict[str, str] = {
    "atak_boga_smokow":          "Atak Boga Smoków",
    "aura_miecza":               "Aura Miecza",
    "berserk":                   "Berserk",
    "bezowa_rosa":               "Beżowa Rosa",
    "blogoslawienstwo":          "Błogosławieństwo",
    "brazowa_rosa":              "Brązowa Rosa",
    "czerwona_rosa":             "Czerwona Rosa",
    "dlon_krytyka":              "Dłoń Krytyka",
    "dlon_przebicia":            "Dłoń Przebicia",
    "fioletowa_mikstura":        "Fioletowa Mikstura",
    "karmazynowa_mikstura":      "Karmazynowa Mikstura",
    "kon":                       "Koń",
    "mikstura_szybkiej_analizy": "Mikstura Szybkiej Analizy",
    "obrona_zycie_boga_smokow":  "Obrona/Życie Boga Smoków",
    "pieczona_shiri":            "Pieczona Shiri",
    "platynowa_rosa":            "Platynowa Rosa",
    "pomoc_smoka":               "Pomoc Smoka",
    "punkty_milosci":            "Punkty Miłości",
    "rozowa_rosa":               "Różowa Rosa",
    "ryba":                      "Ryba",
    "sprint":                    "Sprint",
    "zielona_mikstura":          "Zielona Mikstura",
    "zielona_rosa":              "Zielona Rosa",
    "zwiekszenie_ataku":         "Zwiększenie Ataku",
    "zwinnosc":                  "Zwinność",
    "łezka":                     "Łezka",
}

SCALES          = [0.90, 0.95, 1.0, 1.05, 1.1]
MATCH_THRESHOLD = 0.80
NMS_OVERLAP     = 0.3

# Obszar przeszukiwania: górny lewy róg ekranu z ikonami odpalonych buffów
BUFF_BAR_Y = 120   # max y paska z ikonami
BUFF_BAR_X = 600   # max x paska z ikonami

PREVIEW_W, PREVIEW_H = 640, 400
THUMB_SIZE            = 40


# ── Detekcja ─────────────────────────────────────────────────────────────────

def load_templates() -> list[dict]:
    templates = []
    if not os.path.isdir(TEMPLATES_DIR):
        return templates
    for f in sorted(os.listdir(TEMPLATES_DIR)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        name = os.path.splitext(f)[0]
        img  = cv2.imread(os.path.join(TEMPLATES_DIR, f))
        if img is None:
            continue
        templates.append({"name": name, "bgr": img})
    return templates


def multi_scale_match(screenshot_bgr, template_bgr, scales, threshold):
    th, tw = template_bgr.shape[:2]
    sh, sw = screenshot_bgr.shape[:2]
    detections = []
    for scale in scales:
        nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
        if nw >= sw or nh >= sh:
            continue
        resized = cv2.resize(template_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        result  = cv2.matchTemplate(screenshot_bgr, resized, cv2.TM_CCOEFF_NORMED)
        for pt in zip(*np.where(result >= threshold)[::-1]):
            detections.append({
                "x": int(pt[0]), "y": int(pt[1]),
                "w": nw, "h": nh,
                "score": round(float(result[pt[1], pt[0]]), 4),
            })
    return detections


def non_max_suppression(detections, overlap_thresh):
    if not detections:
        return []
    boxes  = np.array([[d["x"], d["y"], d["x"]+d["w"], d["y"]+d["h"]]
                       for d in detections], dtype=float)
    scores = np.array([d["score"] for d in detections])
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2-x1+1) * (y2-y1+1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        overlap = (np.maximum(0, xx2-xx1+1) * np.maximum(0, yy2-yy1+1)) / areas[order[1:]]
        order   = order[np.where(overlap <= overlap_thresh)[0] + 1]
    return [detections[k] for k in keep]


def detect_odpal(screenshot_path: str, templates: list[dict]) -> list[dict]:
    """Zwraca listę wykrytych ikon z pozycjami i scores."""
    img = cv2.imread(screenshot_path)
    if img is None:
        return []

    # Ogranicz przeszukiwanie do paska z ikonami buffów (górny lewy róg)
    h, w = img.shape[:2]
    region = img[:min(BUFF_BAR_Y, h), :min(BUFF_BAR_X, w)]

    # Globalny NMS przez wszystkie ikony: jeśli dwie ikony nakładają się,
    # wygrywa ta z wyższym score.
    all_candidates: list[dict] = []
    for tpl in templates:
        raw   = multi_scale_match(region, tpl["bgr"], SCALES, MATCH_THRESHOLD)
        found = non_max_suppression(raw, NMS_OVERLAP)
        for det in found:
            all_candidates.append({**det, "name": tpl["name"]})

    if not all_candidates:
        return []

    kept = non_max_suppression(all_candidates, NMS_OVERLAP)
    kept.sort(key=lambda d: (d["y"], d["x"]))  # porządek wierszy od góry do dołu, lewo→prawo
    return kept


# ── GUI ──────────────────────────────────────────────────────────────────────

class OdpalApp:
    def __init__(self, root: tk.Tk):
        self.root  = root
        self.root.title("Odpal Detector — Echo Wygnańców")
        self.root.resizable(False, False)
        self._templates  = load_templates()
        self._photo      = None
        self._result_img = None
        self._build()

    def _build(self):
        root = self.root

        top = tk.Frame(root, padx=8, pady=6)
        top.pack(fill="x")

        tk.Button(top, text="Wybierz zdjęcie…", command=self._pick,
                  width=18).pack(side="left")
        self._lbl_file = tk.Label(top, text="—", anchor="w", fg="#555")
        self._lbl_file.pack(side="left", padx=8)

        # Podgląd
        self._canvas = tk.Canvas(root, width=PREVIEW_W, height=PREVIEW_H,
                                 bg="#1a1a1a", highlightthickness=0)
        self._canvas.pack(padx=8)

        # Wyniki
        res_frame = tk.LabelFrame(root, text="Wykryte odpalenia", padx=6, pady=4)
        res_frame.pack(fill="x", padx=8, pady=(4, 8))

        self._result_inner = tk.Frame(res_frame)
        self._result_inner.pack(fill="x")

        self._status = tk.Label(root, text="", fg="#555", pady=2)
        self._status.pack()

        if not self._templates:
            messagebox.showwarning(
                "Brak szablonów",
                f"Nie znaleziono plików w {TEMPLATES_DIR}\n"
                "Upewnij się że folder odpal/templates/ zawiera ikony."
            )

    def _pick(self):
        path = filedialog.askopenfilename(
            title="Wybierz screenshot",
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp"), ("Wszystkie", "*.*")]
        )
        if not path:
            return
        self._lbl_file.config(text=os.path.basename(path))
        self._status.config(text="Wykrywam…")
        self.root.update()
        self._run(path)

    def _run(self, path: str):
        detections = detect_odpal(path, self._templates)
        self._show_preview(path, detections)
        self._show_results(detections)
        n = len(detections)
        self._status.config(text=f"Znaleziono {n} odpaleni{'e' if n==1 else 'a' if 2<=n<=4 else 'ń'}.")

    def _show_preview(self, path: str, detections: list[dict]):
        img = Image.open(path).convert("RGB")
        # Dopasuj do PREVIEW_W×PREVIEW_H zachowując proporcje
        img.thumbnail((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        iw, ih  = img.size
        scale_x = iw / Image.open(path).width
        scale_y = ih / Image.open(path).height

        # Narysuj ramki wykrytych ikon
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(img)
        colors = ["#00ff88", "#ffcc00", "#ff6666", "#66aaff", "#ff88ff",
                  "#88ffff", "#ffaa44", "#aaffaa", "#ffaaaa"]
        for det in detections:
            x0 = int(det["x"] * scale_x)
            y0 = int(det["y"] * scale_y)
            x1 = int((det["x"] + det["w"]) * scale_x)
            y1 = int((det["y"] + det["h"]) * scale_y)
            col = colors[list(ICON_NAMES).index(det["name"]) % len(colors)] \
                  if det["name"] in ICON_NAMES else "#ffffff"
            draw.rectangle([x0, y0, x1, y1], outline=col, width=2)

        # Wyśrodkuj na canvas
        self._canvas.delete("all")
        self._photo = ImageTk.PhotoImage(img)
        cx = PREVIEW_W // 2
        cy = PREVIEW_H // 2
        self._canvas.create_image(cx, cy, image=self._photo, anchor="center")

    def _show_results(self, detections: list[dict]):
        for w in self._result_inner.winfo_children():
            w.destroy()

        if not detections:
            tk.Label(self._result_inner, text="Nie wykryto żadnych odpalonych ikon.",
                     fg="#888").pack(pady=4)
            return

        icons_dir = TEMPLATES_DIR
        col = 0
        row_frame = tk.Frame(self._result_inner)
        row_frame.pack(fill="x", pady=2)

        for det in detections:
            name        = det["name"]
            label_text  = ICON_NAMES.get(name, name.replace("_", " ").title())

            cell = tk.Frame(row_frame, relief="groove", bd=1, padx=4, pady=4)
            cell.grid(row=0, column=col, padx=4, pady=2, sticky="n")

            # Miniatura ikony
            tpl_path = os.path.join(icons_dir, f"{name}.png")
            if os.path.exists(tpl_path):
                thumb = Image.open(tpl_path).convert("RGB")
                thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                ph = ImageTk.PhotoImage(thumb)
                lbl_img = tk.Label(cell, image=ph)
                lbl_img.image = ph  # zapobiegaj GC
                lbl_img.pack()

            tk.Label(cell, text=label_text, font=("", 8),
                     wraplength=60).pack()
            tk.Label(cell, text=f"{det['score']:.0%}", fg="#555",
                     font=("", 7)).pack()

            col += 1
            if col >= 6:
                col = 0
                row_frame = tk.Frame(self._result_inner)
                row_frame.pack(fill="x", pady=2)


# ── Uruchomienie ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    OdpalApp(root)
    root.mainloop()
