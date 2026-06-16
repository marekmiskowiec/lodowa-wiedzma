"""
Odpal Detector — rozpoznawanie aktywnych wzmocnień (Echo Wygnańców).
Uruchom: python odpal.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw


# ── Ścieżki ──────────────────────────────────────────────────────────────────

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


TEMPLATES_DIR = resource_path("odpal/templates")

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
BUFF_BAR_Y      = 120
BUFF_BAR_X      = 600

PREVIEW_W, PREVIEW_H = 560, 380


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
    img = cv2.imread(screenshot_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    region = img[:min(BUFF_BAR_Y, h), :min(BUFF_BAR_X, w)]
    all_candidates: list[dict] = []
    for tpl in templates:
        raw   = multi_scale_match(region, tpl["bgr"], SCALES, MATCH_THRESHOLD)
        found = non_max_suppression(raw, NMS_OVERLAP)
        for det in found:
            all_candidates.append({**det, "name": tpl["name"]})
    if not all_candidates:
        return []
    kept = non_max_suppression(all_candidates, NMS_OVERLAP)
    kept.sort(key=lambda d: (d["y"], d["x"]))
    return kept


def icon_display_name(name: str) -> str:
    return ICON_NAMES.get(name, name.replace("_", " ").title())


# ── StartScreen ───────────────────────────────────────────────────────────────

class StartScreen:
    def __init__(self, root: tk.Tk):
        self.root      = root
        self.root.title("Odpal Detector — Echo Wygnańców")
        self.root.resizable(False, False)
        self._files    = []
        self._templates = load_templates()
        self._build()

    def _build(self):
        root = self.root
        root.geometry("420x300")

        tk.Label(root, text="Echo Wygnańców — Odpal Detector",
                 font=("", 13, "bold")).pack(pady=(18, 4))
        tk.Label(root, text="Wybierz screenshoty z odpaleniami graczy",
                 fg="#555").pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Wybierz screenshoty…", width=22,
                  command=self._pick).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Wyczyść", width=8,
                  command=self._clear).pack(side="left", padx=4)

        self._listbox = tk.Listbox(root, height=7, width=52, selectmode="extended")
        self._listbox.pack(padx=12)

        self._progress = ttk.Progressbar(root, length=380, mode="determinate")
        self._progress.pack(pady=(8, 0))

        self._btn_start = tk.Button(root, text="Wykryj →", width=16,
                                    state="disabled", command=self._start)
        self._btn_start.pack(pady=8)

        if not self._templates:
            messagebox.showwarning("Brak szablonów",
                f"Nie znaleziono ikon w {TEMPLATES_DIR}")

    def _pick(self):
        paths = filedialog.askopenfilenames(
            title="Wybierz screenshoty",
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp"), ("Wszystkie", "*.*")]
        )
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                self._listbox.insert("end", os.path.basename(p))
        self._btn_start.config(state="normal" if self._files else "disabled")

    def _clear(self):
        self._files.clear()
        self._listbox.delete(0, "end")
        self._btn_start.config(state="disabled")

    def _start(self):
        self._btn_start.config(state="disabled")
        self._progress["maximum"] = len(self._files)
        self._progress["value"]   = 0
        results = []

        def worker():
            for path in self._files:
                dets = detect_odpal(path, self._templates)
                results.append({"path": path, "detections": dets})
                self.root.after(0, lambda: self._progress.step(1))
            self.root.after(0, lambda: self._launch(results))

        threading.Thread(target=worker, daemon=True).start()

    def _launch(self, results):
        for w in self.root.winfo_children():
            w.destroy()
        self.root.geometry("")
        VerificationApp(self.root, results, self._templates)


# ── VerificationApp ───────────────────────────────────────────────────────────

_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#8bc34a",
    "#ff5722", "#607d8b", "#795548", "#ffc107", "#03a9f4",
]

class VerificationApp:
    def __init__(self, root: tk.Tk, results: list[dict], templates: list[dict]):
        self.root      = root
        self.root.title("Odpal Detector — Weryfikacja")
        self._results   = results          # [{path, detections}]
        self._templates = templates
        self._all_names = sorted([t["name"] for t in templates])
        self._verified: list[list[str]] = []   # lista zatwierdzonych nazw per slajd
        self._idx       = 0
        self._photo     = None
        self._checks: list[tuple[tk.BooleanVar, str]] = []  # (var, name)
        self._build()
        self._show(0)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self):
        root = self.root

        # Górny pasek nawigacji
        nav = tk.Frame(root, pady=4)
        nav.pack(fill="x", padx=8)

        self._btn_prev = tk.Button(nav, text="← Wstecz", width=10,
                                   command=self._prev, state="disabled")
        self._btn_prev.pack(side="left")

        self._lbl_nav = tk.Label(nav, text="", font=("", 10))
        self._lbl_nav.pack(side="left", expand=True)

        self._btn_next = tk.Button(nav, text="Dalej →", width=10,
                                   command=self._next)
        self._btn_next.pack(side="right")

        # Główny obszar: podgląd + panel boczny
        main = tk.Frame(root)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Lewo: canvas z podglądem
        left = tk.Frame(main)
        left.pack(side="left", fill="both")

        self._canvas = tk.Canvas(left, width=PREVIEW_W, height=PREVIEW_H,
                                 bg="#111", highlightthickness=0)
        self._canvas.pack()

        # Prawo: panel weryfikacji
        right = tk.Frame(main, width=250, padx=8)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Osoba (nazwa pliku):", anchor="w").pack(fill="x")
        self._var_person = tk.StringVar()
        tk.Entry(right, textvariable=self._var_person).pack(fill="x", pady=(0, 8))

        tk.Label(right, text="Wykryte odpalenia:", anchor="w",
                 font=("", 9, "bold")).pack(fill="x")

        # Scrollowalista checkboxów
        chk_outer = tk.Frame(right)
        chk_outer.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(chk_outer)
        scrollbar.pack(side="right", fill="y")

        self._chk_canvas = tk.Canvas(chk_outer, yscrollcommand=scrollbar.set,
                                     highlightthickness=0)
        self._chk_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._chk_canvas.yview)

        self._chk_frame = tk.Frame(self._chk_canvas)
        self._chk_canvas.create_window((0, 0), window=self._chk_frame, anchor="nw")
        self._chk_frame.bind("<Configure>",
            lambda e: self._chk_canvas.configure(
                scrollregion=self._chk_canvas.bbox("all")))

        # Dodaj ręcznie
        add_frame = tk.Frame(right)
        add_frame.pack(fill="x", pady=(6, 0))
        tk.Label(add_frame, text="Dodaj ręcznie:", anchor="w").pack(fill="x")
        self._var_add = tk.StringVar()
        cb = ttk.Combobox(add_frame, textvariable=self._var_add,
                          values=[icon_display_name(n) for n in self._all_names],
                          state="readonly", width=22)
        cb.pack(side="left", pady=2)
        tk.Button(add_frame, text="+", width=3,
                  command=self._add_manual).pack(side="left", padx=2)

    # ── Wyświetlanie slajdu ───────────────────────────────────────────────────

    def _show(self, idx: int):
        self._idx = idx
        data      = self._results[idx]
        total     = len(self._results)

        self._lbl_nav.config(text=f"{idx+1} / {total}")
        self._btn_prev.config(state="normal" if idx > 0 else "disabled")
        self._btn_next.config(
            text="Podsumowanie →" if idx == total - 1 else "Dalej →")

        # Nazwa osoby z nazwy pliku
        basename = os.path.splitext(os.path.basename(data["path"]))[0]
        self._var_person.set(basename)

        # Buduj checkboxy z wykrytych ikon
        self._build_checks(data["detections"])
        self._draw_preview(data["path"], data["detections"])

    def _build_checks(self, detections: list[dict]):
        for w in self._chk_frame.winfo_children():
            w.destroy()
        self._checks = []

        for det in detections:
            var  = tk.BooleanVar(value=True)
            name = det["name"]
            label = f"{icon_display_name(name)}  ({det['score']:.0%})"
            cb = tk.Checkbutton(self._chk_frame, text=label, variable=var,
                                anchor="w", wraplength=200,
                                command=self._on_check_change)
            cb.pack(fill="x", pady=1)
            self._checks.append((var, name))

    def _draw_preview(self, path: str, detections: list[dict]):
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        img.thumbnail((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        disp_w, disp_h = img.size
        sx = disp_w / orig_w
        sy = disp_h / orig_h

        draw = ImageDraw.Draw(img)
        for i, det in enumerate(detections):
            col = _COLORS[i % len(_COLORS)]
            x0, y0 = int(det["x"] * sx), int(det["y"] * sy)
            x1, y1 = int((det["x"] + det["w"]) * sx), int((det["y"] + det["h"]) * sy)
            draw.rectangle([x0, y0, x1, y1], outline=col, width=2)
            draw.text((x0 + 1, y0 + 1), str(i + 1), fill=col)

        self._canvas.delete("all")
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2,
                                  image=self._photo, anchor="center")

    def _on_check_change(self):
        pass  # można dodać live-odświeżanie ramek w przyszłości

    def _add_manual(self):
        display = self._var_add.get()
        if not display:
            return
        # Znajdź nazwę klucza odpowiadającą wyświetlanej nazwie
        name = next((n for n in self._all_names
                     if icon_display_name(n) == display), display)
        # Sprawdź czy już jest
        existing = [n for _, n in self._checks]
        if name in existing:
            return
        var = tk.BooleanVar(value=True)
        cb  = tk.Checkbutton(self._chk_frame,
                             text=icon_display_name(name),
                             variable=var, anchor="w", wraplength=200,
                             command=self._on_check_change)
        cb.pack(fill="x", pady=1)
        self._checks.append((var, name))
        self._var_add.set("")

    # ── Nawigacja ─────────────────────────────────────────────────────────────

    def _save_current(self):
        person  = self._var_person.get().strip() or os.path.splitext(
                    os.path.basename(self._results[self._idx]["path"]))[0]
        aktywne = [name for var, name in self._checks if var.get()]
        self._results[self._idx]["person"]  = person
        self._results[self._idx]["aktywne"] = aktywne

    def _prev(self):
        self._save_current()
        self._show(self._idx - 1)

    def _next(self):
        self._save_current()
        if self._idx == len(self._results) - 1:
            self._summary()
        else:
            self._show(self._idx + 1)

    # ── Podsumowanie ──────────────────────────────────────────────────────────

    def _summary(self):
        # Uzupełnij brakujące slajdy jeśli ktoś pominął
        for r in self._results:
            if "aktywne" not in r:
                r["person"]  = os.path.splitext(os.path.basename(r["path"]))[0]
                r["aktywne"] = [det["name"] for det in r["detections"]]

        for w in self.root.winfo_children():
            w.destroy()
        self.root.title("Odpal Detector — Podsumowanie")

        tk.Label(self.root, text="Podsumowanie odpalań",
                 font=("", 12, "bold")).pack(pady=(12, 4))

        # Tabela
        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=10, pady=4)

        cols = ("Osoba", "Odpalenia", "Liczba")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=20)
        tree.heading("Osoba",    text="Osoba")
        tree.heading("Odpalenia",text="Odpalenia")
        tree.heading("Liczba",   text="Liczba")
        tree.column("Osoba",    width=160, anchor="w")
        tree.column("Odpalenia",width=420, anchor="w")
        tree.column("Liczba",   width=60,  anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        for r in self._results:
            names = [icon_display_name(n) for n in r.get("aktywne", [])]
            tree.insert("", "end", values=(
                r["person"],
                ", ".join(names),
                len(names),
            ))

        # Przyciski eksportu
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Kopiuj do schowka",
                  command=lambda: self._copy(tree)).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Eksportuj JSON…",
                  command=self._export_json).pack(side="left", padx=6)

    def _copy(self, tree):
        lines = []
        for iid in tree.get_children():
            vals = tree.item(iid)["values"]
            lines.append(f"{vals[0]}: {vals[1]}")
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))

    def _export_json(self):
        import json
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Wszystkie", "*.*")],
            title="Zapisz wyniki")
        if not path:
            return
        data = [
            {"osoba": r["person"], "odpalenia": r.get("aktywne", [])}
            for r in self._results
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ── Uruchomienie ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    StartScreen(root)
    root.mainloop()
