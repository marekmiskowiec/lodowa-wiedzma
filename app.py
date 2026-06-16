"""
Witcher Drop Tracker — zintegrowana aplikacja.
Uruchom: python app.py
"""

import json
import os
import re
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract


# ── Ścieżki zasobów (działa też po spakowaniu PyInstaller) ───────────────────

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


TEMPLATES_DIR = resource_path("templates/items")
NUMBERS_DIR   = resource_path("templates/numbers")

# ── Konfiguracja detekcji ─────────────────────────────────────────────────────

SCALES           = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
NUMBER_SCALES    = [0.8, 0.9, 1.0, 1.1, 1.2]
MATCH_THRESHOLD  = 0.75
NUMBER_THRESHOLD = 0.73
NMS_OVERLAP      = 0.3
ITEM_THRESHOLDS: dict[str, float] = {
    "wzmocnienie":     0.90,
    "rada_pustelnika": 0.80,
}

# ── Konfiguracja GUI ──────────────────────────────────────────────────────────

APP_W, APP_H = 840, 580   # stały rozmiar wszystkich okien
PREVIEW_W, PREVIEW_H = 500, 420
ICON_SLIDE = 24
ICON_SUM   = 32
RIGHT_W    = 300
RIGHT_H    = 440           # stała wysokość prawego panelu

ITEM_NAMES: dict[str, str] = {
    "broszura_szermierki":   "Broszura Szermierki",
    "cert":                  "Cert",
    "krwisty_kamien":        "Krwisty Kamień",
    "rada_pustelnika":       "Rada Pustelnika",
    "skrzynia boga smokow":  "Skrzynia Boga Smoków",
    "skrzynia setou":        "Skrzynia Setou",
    "strategia":             "Strategia",
    "wzmocnienie":           "Wzmocnienie",
    "zmianka":               "Zmianka",
    "zwoj_blogoslawienstwa": "Zwój Błogosławieństwa",
    "zwoj_egzorcyzmu":       "Zwój Egzorcyzmu",
}


def fmt_item(name: str) -> str:
    return ITEM_NAMES.get(name, name.replace("_", " ").title())


def fmt_yang(val: int | None) -> str:
    return f"{val or 0:,}".replace(",", ".")


def parse_yang(text: str) -> int | None:
    raw = text.strip().replace(".", "").replace(",", "").replace(" ", "")
    return int(raw) if raw.isdigit() else None


# ── Detekcja ──────────────────────────────────────────────────────────────────

def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def multi_scale_match(screenshot_gray, template_gray, scales, threshold):
    th, tw = template_gray.shape[:2]
    sh, sw = screenshot_gray.shape[:2]
    detections = []
    for scale in scales:
        nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
        if nw >= sw or nh >= sh:
            continue
        resized = cv2.resize(template_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        result  = cv2.matchTemplate(screenshot_gray, resized, cv2.TM_CCOEFF_NORMED)
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


def _digit_nms(detections, overlap_thresh):
    """NMS z metryką IoU zamiast intersection/area. Sąsiednie cyfry mają niskie IoU
    (~0.08) i nie są wzajemnie tłumione; fałszywe duplikaty w tym samym miejscu
    mają wysokie IoU (~0.44+) i są poprawnie eliminowane."""
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
        inter   = np.maximum(0, xx2-xx1+1) * np.maximum(0, yy2-yy1+1)
        union   = areas[i] + areas[order[1:]] - inter
        overlap = inter / np.maximum(union, 1)
        order   = order[np.where(overlap <= overlap_thresh)[0] + 1]
    return [detections[k] for k in keep]


def detect_quantity(screenshot_gray, item_box, number_templates):
    x, y, w, h = item_box["x"], item_box["y"], item_box["w"], item_box["h"]
    sh, sw = screenshot_gray.shape[:2]
    rx, ry   = x + w // 2, y + h * 2 // 3
    rx2, ry2 = min(x + w + 2, sw), min(y + h + 8, sh)
    if rx2 <= rx or ry2 <= ry:
        return 1, None
    region = screenshot_gray[ry:ry2, rx:rx2]
    _, region_bin = cv2.threshold(region, 190, 255, cv2.THRESH_BINARY)
    all_dets: list[dict] = []
    for tpl in number_templates:
        for det in multi_scale_match(region_bin, tpl["bin"], NUMBER_SCALES, NUMBER_THRESHOLD):
            all_dets.append({**det, "digit": tpl["digit"]})
    if not all_dets:
        return 1, None
    kept = _digit_nms(all_dets, NMS_OVERLAP)
    # Odrzuć cyfry z innej pozycji pionowej niż najlepsze trafienie
    # (eliminuje fałszywe cyfry wychwycone z sąsiednich slotów)
    anchor = max(kept, key=lambda d: d["score"])
    ref_cy = anchor["y"] + anchor["h"] // 2
    kept   = [d for d in kept if abs((d["y"] + d["h"] // 2) - ref_cy) <= anchor["h"] // 2]
    kept.sort(key=lambda d: d["x"])
    number_str  = "".join(str(d["digit"]) for d in kept)
    best_score  = max(d["score"] for d in kept)
    return int(number_str), round(best_score, 4)


def read_yang(screenshot_path: str) -> int | None:
    img  = cv2.imread(screenshot_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Lewa 1/3 obrazu: yang jest zawsze po lewej stronie okna handlu;
    # przycięcie eliminuje ikonę yang i "0" drugiej strony, które zaburzały OCR.
    strip = gray[h // 3:h - 5, 5:w // 3]
    _, binary = cv2.threshold(strip, 170, 255, cv2.THRESH_BINARY)
    row_sums = binary.sum(axis=1) // 255

    # Find contiguous active-row blocks instead of sliding window per row.
    # Sliding window creates ~14 overlapping crops per text line, causing OCR
    # to misread partial glyphs and pick the wrong (larger) value.
    line_regions: list[tuple[int, int]] = []
    in_line = False
    line_start = 0
    for i, s in enumerate(row_sums):
        if not in_line and s >= 6:
            in_line, line_start = True, i
        elif in_line and s < 6:
            in_line = False
            line_regions.append((line_start, i))
    if in_line:
        line_regions.append((line_start, len(row_sums)))

    MIN_H = 14  # minimalna wysokość regionu dla OCR (po 5x skalowaniu = 70px)
    strip_h = strip.shape[0]
    best: int | None = None
    for y0_raw, y1_raw in line_regions:
        if y1_raw - y0_raw < MIN_H:
            pad = (MIN_H - (y1_raw - y0_raw)) // 2
            y0 = max(0, y0_raw - pad)
            y1 = min(strip_h, y0 + MIN_H)
        else:
            y0, y1 = y0_raw, y1_raw
        region = strip[y0:y1, :]
        big    = cv2.resize(region, (region.shape[1]*5, region.shape[0]*5),
                            interpolation=cv2.INTER_NEAREST)
        _, big_bin = cv2.threshold(big, 170, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(
            big_bin, config="--psm 7 -c tessedit_char_whitelist=0123456789.,").strip()
        # \d+ zamiast \d{1,3}: OCR czasem gubi separator między pierwszą cyfrą
        # a resztą (np. "2.400.000" → "2400,000"); zezwolenie na dowolny prefix
        # cyfr pozwala to obsłużyć. Końcowe artefakty (np. "7" z ikony yang)
        # nadal są odrzucane bo nie pasują do grupy [.,]\d{3}.
        for m in re.finditer(r"\d+(?:[.,]\d{3})+", text):
            raw = re.sub(r"[.,]", "", m.group())
            if 6 <= len(raw) <= 9:
                val = int(raw)
                if best is None or val > best:
                    best = val
    return best


def load_templates():
    templates = []
    for f in sorted(os.listdir(TEMPLATES_DIR)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        name = os.path.splitext(f)[0]
        gray = load_image_gray(os.path.join(TEMPLATES_DIR, f))
        templates.append({
            "name": name,
            "gray": gray,
            "threshold": ITEM_THRESHOLDS.get(name, MATCH_THRESHOLD),
        })

    number_templates = []
    if os.path.isdir(NUMBERS_DIR):
        for f in sorted(os.listdir(NUMBERS_DIR)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            digit_str = os.path.splitext(f)[0]
            if digit_str.isdigit():
                gray = load_image_gray(os.path.join(NUMBERS_DIR, f))
                _, bin_tpl = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
                number_templates.append({"digit": int(digit_str),
                                         "gray": gray, "bin": bin_tpl})
    return templates, number_templates


def cross_item_nms(best_by_name: dict, overlap_thresh: float = 0.3) -> set[str]:
    """Jeśli dwa różne przedmioty leżą w tym samym miejscu, zostaw tylko pewniejszy."""
    items = sorted(best_by_name.items(), key=lambda kv: kv[1]["score"], reverse=True)
    keep: set[str] = set()
    suppressed: set[str] = set()
    for name_i, det_i in items:
        if name_i in suppressed:
            continue
        keep.add(name_i)
        xi1, yi1 = det_i["x"], det_i["y"]
        xi2, yi2 = xi1 + det_i["w"], yi1 + det_i["h"]
        area_i   = (xi2 - xi1) * (yi2 - yi1)
        for name_j, det_j in items:
            if name_j in keep or name_j in suppressed:
                continue
            xj1, yj1 = det_j["x"], det_j["y"]
            xj2, yj2 = xj1 + det_j["w"], yj1 + det_j["h"]
            area_j   = (xj2 - xj1) * (yj2 - yj1)
            inter_w  = max(0, min(xi2, xj2) - max(xi1, xj1))
            inter_h  = max(0, min(yi2, yj2) - max(yi1, yj1))
            inter    = inter_w * inter_h
            union    = area_i + area_j - inter
            if union > 0 and inter / union > overlap_thresh:
                suppressed.add(name_j)
    return keep


def process_screenshot(path: str, templates, number_templates) -> dict:
    img_gray = load_image_gray(path)

    # 1. Per-item NMS: usuń nakładające się dopasowania tego samego szablonu
    candidates: list[dict] = []
    for tpl in templates:
        raw   = multi_scale_match(img_gray, tpl["gray"], SCALES, tpl["threshold"])
        found = non_max_suppression(raw, NMS_OVERLAP)
        for det in found:
            candidates.append({**det, "name": tpl["name"]})

    # 2. Globalny NMS przez wszystkie itemy jednocześnie: eliminuje fałszywe dopasowania
    #    nakładające się z pewniejszymi detekcjami innego przedmiotu (np. fałszywa zmianka
    #    na pozycji wzmocnienia). Każda pozycja idzie do przedmiotu z najwyższym score.
    kept_flat = non_max_suppression(candidates, NMS_OVERLAP)

    # 3. Zgrupuj z powrotem po nazwie
    all_by_name: dict[str, list[dict]] = {}
    for det in kept_flat:
        all_by_name.setdefault(det["name"], []).append(det)

    detections = {}
    for tpl in templates:
        name      = tpl["name"]
        instances = all_by_name.get(name, [])
        best      = max(instances, key=lambda d: d["score"]) if instances else None

        total_qty = 0
        for inst in instances:
            q, _ = (detect_quantity(img_gray, inst, number_templates)
                    if number_templates else (None, None))
            total_qty += q or 1

        detections[name] = {
            "found": best is not None,
            "quantity": total_qty if total_qty > 0 else None,
            "position": {
                "x":  best["x"], "y":  best["y"],
                "cx": best["x"] + best["w"] // 2,
                "cy": best["y"] + best["h"] // 2,
            } if best else None,
        }

    yang = read_yang(path)
    return {"screenshot": path, "yang": yang, "detections": detections}


def to_slide_data(results: list[dict]) -> list[dict]:
    slides = []
    for r in results:
        found = [(n, d) for n, d in r["detections"].items() if d["found"]]
        found.sort(key=lambda nd: (nd[1]["position"]["cy"] // 25,
                                   nd[1]["position"]["cx"]))
        slides.append({
            "screenshot": r["screenshot"],
            "yang": r["yang"],
            "items": {n: d["quantity"] or 1 for n, d in found},
        })
    return slides


# ── Ekran startowy ────────────────────────────────────────────────────────────

class StartScreen:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Witcher Drop Tracker")
        self.root.resizable(False, False)
        self._paths: list[str] = []
        self._build()
        root.update()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{APP_W}x{APP_H}+{(sw-APP_W)//2}+{(sh-APP_H)//2}")

    def _build(self):
        tk.Label(self.root, text="Witcher Drop Tracker",
                 font=("Arial", 22, "bold"), pady=24).pack()

        tk.Button(self.root, text="Wybierz screenshoty...", width=28,
                  font=("Arial", 13), command=self._pick).pack()

        self._listbox = tk.Listbox(self.root, height=8, width=60,
                                   font=("Arial", 11), activestyle="none")
        self._listbox.pack(padx=30, pady=14)

        self._btn_run = tk.Button(self.root, text="Analizuj",
                                  font=("Arial", 13, "bold"), width=22,
                                  state="disabled", command=self._run)
        self._btn_run.pack(pady=(0, 12))

        style = ttk.Style()
        style.configure("Big.Horizontal.TProgressbar",
                        thickness=22, troughcolor="#888888", background="#4a90d9")
        self._progress = ttk.Progressbar(self.root, length=500, mode="determinate",
                                         style="Big.Horizontal.TProgressbar")
        self._progress.pack(padx=30)

        self._status = tk.Label(self.root, text="", font=("Arial", 11), fg="white")
        self._status.pack(pady=8)

    def _pick(self):
        paths = filedialog.askopenfilenames(
            title="Wybierz screenshoty",
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp"), ("Wszystkie", "*.*")],
        )
        if not paths:
            return
        self._paths = list(paths)
        self._listbox.delete(0, "end")
        for p in self._paths:
            self._listbox.insert("end", os.path.basename(p))
        self._btn_run.config(state="normal",
                             text=f"Analizuj ({len(self._paths)} plików)")

    def _run(self):
        self._btn_run.config(state="disabled")
        self._progress["maximum"] = len(self._paths)
        self._progress["value"]   = 0
        self._status.config(text="Wczytywanie szablonów...")
        self.root.update()
        threading.Thread(target=self._detect, daemon=True).start()

    def _detect(self):
        try:
            templates, number_templates = load_templates()
        except Exception as e:
            self.root.after(0, messagebox.showerror,
                            "Błąd szablonów", str(e))
            self.root.after(0, self._btn_run.config, {"state": "normal"})
            return

        results = []
        for i, path in enumerate(self._paths):
            name = os.path.basename(path)
            self.root.after(0, self._status.config,
                            {"text": f"Analizuję {name}..."})
            try:
                results.append(process_screenshot(path, templates, number_templates))
            except Exception as e:
                self.root.after(0, messagebox.showwarning,
                                "Błąd", f"Pomijam {name}:\n{e}")
            self.root.after(0, self._progress.config, {"value": i + 1})

        data = to_slide_data(results)
        if not data or all(not d["items"] for d in data):
            self.root.after(0, messagebox.showinfo,
                            "Brak wyników",
                            "Nie wykryto żadnych przedmiotów.\n"
                            "Sprawdź czy screenshoty są z okna handlu.")
            self.root.after(0, self._btn_run.config, {"state": "normal"})
            return

        self.root.after(0, self._launch, data)

    def _launch(self, data: list[dict]):
        for w in self.root.winfo_children():
            w.destroy()
        self.root.resizable(False, False)
        VerificationApp(self.root, data)


# ── Weryfikacja ───────────────────────────────────────────────────────────────

class VerificationApp:
    def __init__(self, root: tk.Tk, data: list[dict]):
        self.root = root
        self.root.title("Weryfikacja dropu — The Witcher")

        self.data     = data
        self.verified: list[dict] = []
        self.idx      = 0
        self.item_vars: dict[str, tk.StringVar] = {}
        self.yang_var = tk.StringVar()
        self.yang_var.trace_add("write", self._validate_yang)
        self._yang_updating = False
        self._in_summary = False

        all_items = {name for e in data for name in e["items"]}
        self._icons_slide: dict[str, ImageTk.PhotoImage] = {}
        self._icons_sum:   dict[str, ImageTk.PhotoImage] = {}
        for name in all_items:
            path = os.path.join(TEMPLATES_DIR, f"{name}.png")
            if os.path.exists(path):
                img = Image.open(path)
                self._icons_slide[name] = ImageTk.PhotoImage(
                    img.resize((ICON_SLIDE, ICON_SLIDE), Image.LANCZOS))
                self._icons_sum[name] = ImageTk.PhotoImage(
                    img.resize((ICON_SUM, ICON_SUM), Image.LANCZOS))

        self._build()
        self._load_screenshot()
        self.root.update()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        _, pos = self.root.geometry().split("+", 1)
        x, y = (int(v) for v in pos.split("+"))
        # zachowaj pozycję ze StartScreen, tylko upewnij się że rozmiar jest stały
        self.root.geometry(f"{APP_W}x{APP_H}+{x}+{y}")

    # ── Szkielet ─────────────────────────────────────────────────────────────

    def _build(self):
        self.header = tk.Label(self.root, text="",
                               font=("Arial", 13, "bold"), pady=8)
        self.header.pack(fill="x")

        self._main = tk.Frame(self.root)
        self._main.pack(padx=12, pady=4)
        self._main.columnconfigure(0, minsize=PREVIEW_W + 16)

        self.img_label = tk.Label(self._main, bg="#222",
                                  width=PREVIEW_W, height=PREVIEW_H)
        self.img_label.grid(row=0, column=0, padx=(0, 16), sticky="n")

        self.right = tk.Frame(self._main, width=RIGHT_W, height=RIGHT_H)
        self.right.grid(row=0, column=1, sticky="n")
        self.right.grid_propagate(False)
        self.right.columnconfigure(0, minsize=ICON_SLIDE + 6)
        self.right.columnconfigure(1, weight=1)
        self.right.columnconfigure(2, minsize=64)

        total_w = PREVIEW_W + 16 + RIGHT_W
        self._sum_frame = tk.Frame(self._main, width=total_w, height=RIGHT_H)
        self._sum_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self._sum_frame.grid_propagate(False)
        self._sum_frame.grid_remove()

        nav = tk.Frame(self.root)
        nav.pack(fill="x", padx=12, pady=(6, 2))

        self.btn_prev = tk.Button(nav, text="<- Poprzedni", width=18,
                                  font=("Arial", 11), command=self._prev)
        self.btn_prev.pack(side="left")

        self.lbl_progress = tk.Label(nav, text="", font=("Arial", 11))
        self.lbl_progress.pack(side="left", expand=True)

        self.btn_next = tk.Button(nav, text="Nastepny ->", width=20,
                                  font=("Arial", 11, "bold"), command=self._next)
        self.btn_next.pack(side="right")

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=12, pady=(0, 10))

        self.btn_skip = tk.Button(bottom, text="Przejdz do podsumowania",
                                  font=("Arial", 11), fg="#555", command=self._jump_to_summary)
        self.btn_skip.pack(side="left")

        self.btn_save = tk.Button(bottom, text="Zapisz JSON", width=18,
                                  font=("Arial", 11, "bold"), command=self._save)
        self.btn_save.pack(side="right")
        self.btn_save.pack_forget()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate_yang(self, *_):
        if self._yang_updating:
            return
        digits = "".join(c for c in self.yang_var.get() if c.isdigit())
        formatted = f"{int(digits):,}".replace(",", ".") if digits else "0"
        if formatted != self.yang_var.get():
            self._yang_updating = True
            self.yang_var.set(formatted)
            self._yang_updating = False

    def _force_redraw(self):
        self.root.update_idletasks()
        geo  = self.root.geometry()
        size, pos = geo.split("+", 1)
        w, h = (int(v) for v in size.split("x"))
        self.root.geometry(f"{w+1}x{h}+{pos}")
        self.root.update_idletasks()
        self.root.geometry(geo)
        self.root.update()

    def _clear_right(self):
        for w in self.right.winfo_children():
            w.destroy()
        self.item_vars.clear()

    def _clear_sum(self):
        for w in self._sum_frame.winfo_children():
            w.destroy()

    # ── Slajd ────────────────────────────────────────────────────────────────

    def _load_screenshot(self):
        self._in_summary = False
        entry = self.data[self.idx]
        n     = len(self.data)
        name  = os.path.basename(entry["screenshot"])

        self.header.config(text=f"Screenshot {self.idx+1}/{n}: {name}")
        self.lbl_progress.config(text=f"{self.idx+1} / {n}")
        self.btn_prev.config(state="normal" if self.idx > 0 else "disabled",
                             text="<- Poprzedni")
        self.btn_next.config(
            text="Zatwierdz ->" if self.idx == n-1 else "Nastepny ->",
            command=self._next)
        self.btn_skip.pack(side="left")
        self.btn_save.pack_forget()

        self._sum_frame.grid_remove()
        self.img_label.grid(row=0, column=0, padx=(0, 16), sticky="n")
        self.right.grid(row=0, column=1, sticky="n")

        img = cv2.imread(entry["screenshot"])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            pil.thumbnail((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(pil)
            self.img_label.config(image=self._photo, text="")
        else:
            self.img_label.config(image="", text="brak obrazu", fg="#aaa")

        self._clear_right()

        tk.Label(self.right, text="", width=2).grid(row=0, column=0)
        tk.Label(self.right, text="Przedmiot",
                 font=("Arial", 9, "bold"), anchor="w").grid(row=0, column=1, sticky="w")
        tk.Label(self.right, text="Sztuk",
                 font=("Arial", 9, "bold"), anchor="center").grid(row=0, column=2)
        ttk.Separator(self.right, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=3)

        for i, (item, qty) in enumerate(entry["items"].items(), start=2):
            icon = self._icons_slide.get(item)
            tk.Label(self.right, image=icon if icon else None,
                     width=ICON_SLIDE+4).grid(row=i, column=0, padx=(0, 4))
            tk.Label(self.right, text=fmt_item(item), anchor="w").grid(
                row=i, column=1, sticky="w", pady=2)
            var = tk.StringVar(value=str(qty))
            self.item_vars[item] = var
            tk.Spinbox(self.right, from_=0, to=999, width=4,
                       textvariable=var).grid(row=i, column=2, padx=6)

        sep_row = len(entry["items"]) + 2
        ttk.Separator(self.right, orient="horizontal").grid(
            row=sep_row, column=0, columnspan=3, sticky="ew", pady=(10, 4))

        yang = entry.get("yang")
        tk.Label(self.right, text="Yang:", font=("Arial", 12, "bold"),
                 anchor="w").grid(row=sep_row+1, column=0, columnspan=2, sticky="w")
        self.yang_var.set(fmt_yang(yang))
        tk.Entry(self.right, textvariable=self.yang_var, width=10,
                 font=("Arial", 12, "bold")).grid(
            row=sep_row+1, column=2, padx=6, sticky="ew")

        self._force_redraw()

    # ── Podsumowanie ──────────────────────────────────────────────────────────

    def _aggregate(self) -> tuple[dict[str, int], int | None]:
        totals: dict[str, int] = {}
        total_yang: int | None = None
        for entry in self.verified:
            for item, qty in entry["items"].items():
                totals[item] = totals.get(item, 0) + qty
            yang = entry.get("yang")
            if yang:
                total_yang = (total_yang or 0) + yang
        return totals, total_yang

    def _show_summary(self):
        self._in_summary = True
        totals, total_yang = self._aggregate()

        self.header.config(text="Łączny drop")
        self.lbl_progress.config(text="")
        self.btn_prev.config(state="normal", text="<- Poprzedni")
        self.btn_next.config(text="Zamknij", command=self.root.destroy)
        self.btn_skip.pack_forget()
        self.btn_save.pack(side="right")

        self.img_label.grid_remove()
        self.right.grid_remove()
        self._clear_sum()
        self._sum_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        items = sorted(totals.items())
        bg    = self.root.cget("bg")

        outer = tk.Frame(self._sum_frame, bg=bg)
        outer.place(relx=0.5, rely=0.5, anchor="center")

        style = ttk.Style()
        style.configure("Drop.Treeview", rowheight=30, font=("Arial", 11))
        style.configure("Drop.Treeview.Heading", font=("Arial", 11, "bold"))

        tree_f = tk.Frame(outer, bg=bg)
        tree_f.pack()
        tree = ttk.Treeview(tree_f, columns=("qty",), show="tree headings",
                             height=len(items), style="Drop.Treeview",
                             selectmode="none")
        tree.heading("#0",  text="Przedmiot", anchor="w")
        tree.heading("qty", text="Łącznie",   anchor="center")
        tree.column("#0",  width=320, stretch=False, anchor="w")
        tree.column("qty", width=100, stretch=False, anchor="center")
        tree.pack()

        for item, qty in items:
            icon = self._icons_sum.get(item)
            tree.insert("", "end", image=icon if icon else "",
                        text=f"  {fmt_item(item)}", values=(qty,))

        tk.Frame(outer, height=14, bg=bg).pack()
        yang_f = tk.Frame(outer, bg=bg)
        yang_f.pack(fill="x")
        yang_str = fmt_yang(total_yang) + " yang"
        tk.Label(yang_f, text="Yang lacznie:", bg=bg,
                 font=("Arial", 12, "bold")).pack(side="left")
        tk.Label(yang_f, text=yang_str, bg=bg,
                 font=("Arial", 12, "bold"), fg="white").pack(side="left", padx=(12, 0))

        self.root.update()

    # ── Zapis ─────────────────────────────────────────────────────────────────

    def _save(self):
        totals, total_yang = self._aggregate()
        path = filedialog.asksaveasfilename(
            title="Zapisz wyniki",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="drop_totals.json",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"yang": total_yang, "items": totals}, f,
                      ensure_ascii=False, indent=2)
        self.btn_save.config(text="Zapisano!", state="disabled")
        self.root.after(2000, lambda: self.btn_save.config(
            text="Zapisz JSON", state="normal"))

    # ── Nawigacja ─────────────────────────────────────────────────────────────

    def _collect(self) -> dict:
        entry = self.data[self.idx]
        items = {n: int(v.get() or 0) for n, v in self.item_vars.items()
                 if int(v.get() or 0) > 0}
        parsed = parse_yang(self.yang_var.get())
        yang   = parsed if parsed is not None else entry.get("yang")
        return {"screenshot": entry["screenshot"], "yang": yang, "items": items}

    def _prev(self):
        if self._in_summary:
            self._in_summary = False
            self.verified.pop()
            self._load_screenshot()
        elif self.idx > 0:
            if self.verified:
                self.verified.pop()
            self.idx -= 1
            self._load_screenshot()

    def _next(self):
        self.verified.append(self._collect())
        if self.idx < len(self.data) - 1:
            self.idx += 1
            self._load_screenshot()
        else:
            self._show_summary()

    def _jump_to_summary(self):
        self.verified.append(self._collect())
        for i in range(self.idx + 1, len(self.data)):
            e = self.data[i]
            self.verified.append({
                "screenshot": e["screenshot"],
                "yang": e.get("yang"),
                "items": dict(e["items"]),
            })
        self.idx = len(self.data) - 1
        self._show_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    StartScreen(root)
    root.mainloop()


if __name__ == "__main__":
    main()
