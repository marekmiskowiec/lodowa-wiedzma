"""
Echo Drop Tracker — odczyt dropu z logów chatu The Witcher Online.
Parsuje linie z "otrzymał" / "otrzymałeś" i zbiera zestawienie po graczu.
Uruchom: python echo_drop.py
"""

import json
import os
import re
import sys
import threading
import tkinter as tk
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pytesseract

# ── Stałe ──────────────────────────────────────────────────────────────────────

APP_W, APP_H = 920, 660

# Wzór dropów Echo Wygnańców: miejsce → oczekiwana liczba przedmiotów
_ECHO_PATTERN: dict[int, int] = {
    **{i: 4 for i in range(1,  4)},   # miejsca 1–3:  4 przedmioty
    **{i: 3 for i in range(4,  7)},   # miejsca 4–6:  3 przedmioty
    **{i: 2 for i in range(7, 16)},   # miejsca 7–15: 2 przedmioty
}

def expected_drops(position: int) -> int:
    return _ECHO_PATTERN.get(position, 2)

_DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echo-drop")

def _load_list(filename: str) -> list[str]:
    path = os.path.join(_DICT_DIR, filename)
    if not os.path.exists(path):
        return []
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

KNOWN_PLAYERS: list[str] = _load_list("nabijacy.txt")
KNOWN_ITEMS:   list[str] = _load_list("drop.txt")

_PLAYER_CUTOFF   = 0.60
_ITEM_CUTOFF     = 0.60
_CORRECTIONS_PATH = os.path.join(_DICT_DIR, "corrections.json")


def _load_corrections() -> dict:
    if not os.path.exists(_CORRECTIONS_PATH):
        return {"players": {}, "items": {}, "unrecognized_players": [], "unrecognized_items": []}
    with open(_CORRECTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("players", {})
    data.setdefault("items", {})
    data.setdefault("unrecognized_players", [])
    data.setdefault("unrecognized_items", [])
    return data


def _save_corrections(corrections: dict) -> None:
    with open(_CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)


# Stan sesji — załadowany raz przy starcie, aktualizowany w trakcie
_corrections: dict = _load_corrections()
_session_unrecognized: dict = {"players": set(), "items": set()}


def normalize_player(name: str) -> str:
    if name == "Ty" or not KNOWN_PLAYERS:
        return name
    # 1. Exact match z zapamiętanych korekt
    if name in _corrections["players"]:
        return _corrections["players"][name]
    # 2. Fuzzy match
    matches = get_close_matches(name, KNOWN_PLAYERS, n=1, cutoff=_PLAYER_CUTOFF)
    if matches:
        _corrections["players"][name] = matches[0]   # zapamiętaj na przyszłość
        return matches[0]
    # 3. Nierozpoznany
    _session_unrecognized["players"].add(name)
    return name


def normalize_item(item: str) -> str | None:
    """Normalizuje nazwę itemu przez słownik. Zwraca None jeśli brak dopasowania."""
    if not KNOWN_ITEMS:
        return item  # brak słownika — akceptuj wszystko
    if item in _corrections["items"]:
        return _corrections["items"][item]
    matches = get_close_matches(item, KNOWN_ITEMS, n=1, cutoff=_ITEM_CUTOFF)
    if matches:
        _corrections["items"][item] = matches[0]
        return matches[0]
    _session_unrecognized["items"].add(item)
    return None  # brak dopasowania — odfiltruj śmieć OCR


def flush_corrections() -> None:
    """Zapisz corrections.json z bieżącymi danymi sesji."""
    _corrections["unrecognized_players"] = sorted(
        set(_corrections.get("unrecognized_players", [])) | _session_unrecognized["players"]
    )
    _corrections["unrecognized_items"] = sorted(
        set(_corrections.get("unrecognized_items", [])) | _session_unrecognized["items"]
    )
    _save_corrections(_corrections)

# OCR garbi "otrzymał" na wiele sposobów (otreymat, atraymat, otfzymat …),
# ale prawie każdy wariant kończy się na "mat". Ilość "1x" bywa czytana
# jako lx / ix / tx / 1s lub samo "x".
# Ilość: 1x i wszystkie warianty OCR (lx, ix, tx, iz, 1%, {x, Ix, 12, samo 1)
_QTY = r"(?:\d[x×*s%2]|[{li1I][x×Xz]|lx|ix|tx|iz|[x×X]|\d+(?=\s))\s*:?"

# Nazwa gracza: litery/cyfry + ASCII apostrophe (\\x27) i Unicode apostrofy (U+2018, U+2019)
_PLAYER = "[A-Za-z0-9\\x27\\u2018\\u2019_.]{2,20}"

# "[Gracz][,.?]? [garbled-otrzymał] 1x [Przedmiot]"
# (?:m|rn?|ro) obsługuje warianty OCR: rnat→rn, raat→r, roat→ro
# [a-z]{1,2} obsługuje "mal", "mak", "mat", "mia", "mab" itp.
PATTERN_OTHER = re.compile(
    r"^(" + _PLAYER + r")[,.:]?\s+\w{3,}(?:m|rn?|ro)[ai][a-z]{1,2}\s+" + _QTY + r"\s*(.+)$",
    re.IGNORECASE | re.UNICODE,
)

# "Otrzymałeś [Przedmiot]" — OCR czyta jako "Obrzyrmakes …" / "Otrzymales …"
# Brak ilości "1x", zaczyna z wielkiej litery, zawiera "rma"/"tma"/"trzy"
# [a-z]* (zero lub więcej) obsługuje "Otrzymales" gdzie "trzy" jest tuż po "O"
PATTERN_SELF = re.compile(
    r"^[A-Z][a-z]*(?:rma|zyrm|tma|trzy)\S{0,10}\s+(.{3,})$",
    re.IGNORECASE | re.UNICODE,
)

# Fallback: OCR zlał nazwę gracza z czasownikiem w jeden token bez spacji
# np. "Giebrostotreymat 1x Szkatulka" → player="Giebrost", item="Szkatulka"
_VERB_MARKER = r"(?:otrzy|otrey|otzr|trzym|trzyma|treyma)"
PATTERN_CONCAT = re.compile(
    r"^([A-Za-z0-9_.]{2,15}" + _VERB_MARKER + r"\S{0,8})\s+" + _QTY + r"\s*(.+)$",
    re.IGNORECASE | re.UNICODE,
)
_VERB_SPLIT = re.compile(_VERB_MARKER, re.IGNORECASE)


def _split_player_from_concat(token: str) -> str:
    """Wyciąga prefiks (nazwę gracza) z tokenu player+verb zlanego przez OCR."""
    m = _VERB_SPLIT.search(token)
    return token[:m.start()] if m and m.start() >= 2 else token


# ── OCR i parsowanie ───────────────────────────────────────────────────────────

def _preprocess(img: np.ndarray) -> list[np.ndarray]:
    """Zwraca kilka wariantów preprocessingu do próby OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    big = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    inv = cv2.bitwise_not(big)

    # Wariant 1: inwersja + stały próg 130
    _, inv_thresh = cv2.threshold(inv, 130, 255, cv2.THRESH_BINARY)

    # Wariant 2: inwersja + OTSU (automatyczny próg — działa gdy tekst jest żółty/przyciemniony)
    _, otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Wariant 3: próg adaptacyjny na oryginale (bez inwersji)
    adapt = cv2.adaptiveThreshold(big, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 10)

    return [inv_thresh, otsu, adapt]


def _ocr_image(path: str) -> str:
    """Uruchamia OCR na screenshocie chatu; zwraca surowy tekst."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    lang = "pol" if "pol" in pytesseract.get_languages() else "eng"
    variants = _preprocess(img)

    best_text = ""
    best_count = -1
    for variant in variants:
        text = pytesseract.image_to_string(
            variant, lang=lang,
            config="--psm 6 --oem 3",
        )
        # Wybierz wariant z największą liczbą trafień słów kluczowych
        hits = len(re.findall(r"otrzyma", text, re.IGNORECASE))
        if hits > best_count:
            best_count = hits
            best_text = text

    return best_text


def _clean_item(text: str) -> str:
    """Usuwa końcowe artefakty OCR i interpunkcję z nazwy przedmiotu."""
    # Strip znaków gildii i szumu: ® © ° i wszystko po nich
    text = re.sub(r"[®©°].*$", "", text)
    # Strip nawiasów gildyjnych i wszystkiego po nich: [Fenix] itp.
    text = re.sub(r"\[.*$", "", text)
    # Strip szumu za kropką + wielka litera (np. ". Flay Fenix 4")
    # Chroni "Złota Sztab.(2Mil.Yang)" — tam po kropce jest "(" nie spacja
    text = re.sub(r"\.\s+[A-Z][a-z].*$", "", text)
    # Usuń typowe ogony: ". * x", trailing przecinki/gwiazdki/spacje
    text = re.sub(r"\s*[.*]\s*[x×*]\s*$", "", text)
    return re.sub(r"[,.\s*:]+$", "", text).strip()


def parse_drops(text: str) -> list[dict]:
    """Wyciąga zdarzenia dropu z tekstu OCR i normalizuje nazwy przez słowniki."""
    drops = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip('*"“”‘’ \t')
        line = line.replace('’', "'").replace('‘', "'")
        if len(line) < 6:
            continue
        m = PATTERN_SELF.match(line)
        if m:
            item = normalize_item(_clean_item(m.group(1)))
            if item:
                drops.append({"player": "Ty", "item": item})
            continue
        m = PATTERN_OTHER.match(line)
        if m:
            player = normalize_player(m.group(1).strip().rstrip("."))
            item   = normalize_item(_clean_item(m.group(2)))
            if player and item:
                drops.append({"player": player, "item": item})
            continue
        m = PATTERN_CONCAT.match(line)
        if m:
            raw_player = _split_player_from_concat(m.group(1))
            player = normalize_player(raw_player)
            item   = normalize_item(_clean_item(m.group(2)))
            if player and item:
                drops.append({"player": player, "item": item})
    return drops


def process_screenshots(paths: list[str],
                        progress_cb=None) -> list[dict]:
    """Przetwarza listę ścieżek; zwraca listę {path, raw_text, drops}."""
    results = []
    for i, path in enumerate(paths):
        text  = _ocr_image(path)
        drops = parse_drops(text)
        results.append({"path": path, "raw_text": text, "drops": drops})
        if progress_cb:
            progress_cb(i + 1)
    return results


def aggregate_drops(results: list[dict]) -> dict[str, list[str]]:
    """Agreguje dropy po graczu, zachowując kolejność pierwszego wystąpienia (góra→dół)."""
    by_player: dict[str, list[str]] = {}
    for r in results:
        for d in r["drops"]:
            by_player.setdefault(d["player"], []).append(d["item"])
    return by_player


# ── Ekran startowy ─────────────────────────────────────────────────────────────

class EchoDropStartScreen:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self._paths: list[str] = []
        self.root.title("Echo Drop Tracker")
        self.root.resizable(False, False)
        self._build()
        root.update()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{APP_W}x{APP_H}+{(sw - APP_W) // 2}+{(sh - APP_H) // 2}")

    def _build(self):
        tk.Label(self.root, text="Echo Drop Tracker",
                 font=("Arial", 22, "bold"), pady=22).pack()
        tk.Label(self.root,
                 text='Wczytaj screenshoty chatu — wykrywa linie z "otrzymal" / "otrzymales"',
                 font=("Arial", 11), fg="#aaa").pack()

        tk.Button(self.root, text="Wybierz screenshoty chatu...", width=34,
                  font=("Arial", 13), command=self._pick).pack(pady=(18, 4))

        self._listbox = tk.Listbox(self.root, height=8, width=66,
                                   font=("Arial", 11), activestyle="none")
        self._listbox.pack(padx=30, pady=10)

        self._btn_run = tk.Button(
            self.root, text="Analizuj", font=("Arial", 13, "bold"),
            width=22, state="disabled", command=self._run,
        )
        self._btn_run.pack(pady=(0, 12))

        style = ttk.Style()
        style.configure("ED.Horizontal.TProgressbar",
                        thickness=22, troughcolor="#888888", background="#d97a4a")
        self._progress = ttk.Progressbar(
            self.root, length=520, mode="determinate",
            style="ED.Horizontal.TProgressbar",
        )
        self._progress.pack(padx=30)

        self._status = tk.Label(self.root, text="", font=("Arial", 11), fg="white")
        self._status.pack(pady=8)

    def _pick(self):
        paths = filedialog.askopenfilenames(
            title="Wybierz screenshoty chatu",
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
        self._status.config(text="Analizuję chat (OCR)…")
        self.root.update()
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        results: list[dict] = []
        for i, path in enumerate(self._paths):
            name = os.path.basename(path)
            self.root.after(0, self._status.config, {"text": f"OCR: {name}…"})
            try:
                text  = _ocr_image(path)
                drops = parse_drops(text)
                results.append({"path": path, "raw_text": text, "drops": drops})
            except Exception as e:
                self.root.after(0, messagebox.showwarning,
                                "Błąd", f"Pomijam {name}:\n{e}")
            self.root.after(0, self._progress.config, {"value": i + 1})

        total = sum(len(r["drops"]) for r in results)
        if total == 0:
            self.root.after(0, messagebox.showinfo,
                            "Brak wyników",
                            "Nie wykryto zadnych linii dropu.\n"
                            "Sprawdz czy screenshoty zawieraja tekst z 'otrzymal' / 'otrzymales'.\n\n"
                            "Wskazowka: uzyj przycisku 'Surowy OCR' zeby zobaczyc co Tesseract odczytal.")
            # Pokaż wyniki mimo to (z surowym OCR do diagnostyki)
            if results:
                self.root.after(0, self._launch, results)
            else:
                self.root.after(0, self._btn_run.config, {"state": "normal"})
            return

        flush_corrections()
        self.root.after(0, self._launch, results)

    def _launch(self, results: list[dict]):
        for w in self.root.winfo_children():
            w.destroy()
        self.root.geometry("")
        EchoDropResultsApp(self.root, results)


# ── Ekran wyników ──────────────────────────────────────────────────────────────

class EchoDropResultsApp:
    def __init__(self, root: tk.Tk, results: list[dict]):
        self.root      = root
        self.results   = results
        self.by_player = aggregate_drops(results)
        self.root.title("Echo Drop — Wyniki")
        self._build()
        root.update()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{APP_W}x{APP_H}+{(sw - APP_W) // 2}+{(sh - APP_H) // 2}")

    def _build(self):
        bg = self.root.cget("bg")

        tk.Label(self.root, text="Drop z chatu Echo Wygnańców",
                 font=("Arial", 16, "bold"), pady=12).pack()

        total   = sum(len(v) for v in self.by_player.values())
        players = len(self.by_player)
        tk.Label(self.root,
                 text=f"{total} dropów  •  {players} graczy  •  {len(self.results)} screenshot(y)",
                 font=("Arial", 10), fg="#888").pack()

        # Scrollowalna lista kart
        outer = tk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=16, pady=10)

        vsb = ttk.Scrollbar(outer, orient="vertical")
        vsb.pack(side="right", fill="y")

        self._canvas = tk.Canvas(outer, yscrollcommand=vsb.set,
                                  highlightthickness=0, bg=bg)
        self._canvas.pack(side="left", fill="both", expand=True)
        vsb.config(command=self._canvas.yview)

        self._inner = tk.Frame(self._canvas, bg=bg)
        self._win   = self._canvas.create_window((0, 0), window=self._inner,
                                                  anchor="nw")

        self._inner.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(
            self._win, width=e.width))
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self._canvas.bind("<Button-4>",   self._on_scroll)
        self._canvas.bind("<Button-5>",   self._on_scroll)

        self._fill_cards(bg)

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=16, pady=(0, 12))
        tk.Button(bottom, text="Surowy OCR", font=("Arial", 11),
                  command=self._show_raw).pack(side="left")
        tk.Button(bottom, text="Nierozpoznane", font=("Arial", 11),
                  command=self._show_unrecognized).pack(side="left", padx=(8, 0))
        tk.Button(bottom, text="Zapisz JSON", font=("Arial", 11, "bold"),
                  width=16, command=self._save).pack(side="right")

    def _on_scroll(self, event):
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            self._canvas.yview_scroll(-1, "units")
        else:
            self._canvas.yview_scroll(1, "units")

    def _fill_cards(self, bg: str):
        HDR_BG  = "#2c5282"
        HDR_FG  = "#ffffff"
        ROW_ODD = "#2a2a2a"
        OK_FG   = "#68d391"   # zielony — komplet
        MISS_FG = "#fc8181"   # czerwony — brakuje
        PART_FG = "#f6ad55"   # pomarańczowy — częściowy

        for pos, (player, items) in enumerate(self.by_player.items(), start=1):
            expected = expected_drops(pos)
            found    = len(items)
            counts: dict[str, int] = {}
            for item in items:
                counts[item] = counts.get(item, 0) + 1

            if found >= expected:
                score_fg, mark = OK_FG,   "✓"
            elif found == 0:
                score_fg, mark = MISS_FG, "✗"
            else:
                score_fg, mark = PART_FG, "~"

            card = tk.Frame(self._inner, bg=bg)
            card.pack(fill="x", padx=10, pady=(8, 0))

            # Nagłówek gracza
            hdr = tk.Frame(card, bg=HDR_BG)
            hdr.pack(fill="x")
            hdr.columnconfigure(1, weight=1)
            tk.Label(hdr, text=f"  #{pos}", bg=HDR_BG, fg="#90cdf4",
                     font=("Arial", 10), width=4, anchor="w").grid(row=0, column=0)
            tk.Label(hdr, text=player, bg=HDR_BG, fg=HDR_FG,
                     font=("Arial", 11, "bold"), pady=5,
                     anchor="w").grid(row=0, column=1, sticky="ew")
            tk.Label(hdr, text=f"{mark} {found}/{expected}  ", bg=HDR_BG, fg=score_fg,
                     font=("Arial", 10, "bold")).grid(row=0, column=2)

            # Wiersze itemów
            tbl = tk.Frame(card, bg=bg)
            tbl.pack(fill="x")
            tbl.columnconfigure(0, weight=1)

            for i, (item, cnt) in enumerate(sorted(counts.items())):
                row_bg = ROW_ODD if i % 2 else bg
                row = tk.Frame(tbl, bg=row_bg)
                row.grid(row=i, column=0, sticky="ew")
                row.columnconfigure(0, weight=1)
                tk.Label(row, text=f"  {item}", bg=row_bg, font=("Arial", 10),
                         anchor="w", pady=3).grid(row=0, column=0, sticky="ew")
                tk.Label(row, text=f"x{cnt}  " if cnt > 1 else "   ",
                         bg=row_bg, font=("Arial", 10), fg="#aaa",
                         width=5, anchor="e").grid(row=0, column=1)

    def _save(self):
        data = {
            player: sorted(items)
            for player, items in self.by_player.items()
        }
        path = filedialog.asksaveasfilename(
            title="Zapisz wyniki dropu",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="echo_drop_results.json",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Zapisano", f"Wyniki zapisane:\n{path}")

    def _show_unrecognized(self):
        players = sorted(_corrections.get("unrecognized_players", []))
        items   = sorted(_corrections.get("unrecognized_items", []))

        top = tk.Toplevel(self.root)
        top.title("Nierozpoznane stringi OCR")
        top.geometry("560x420")

        tk.Label(top, text="Stringi OCR poniżej progu dopasowania",
                 font=("Arial", 12, "bold"), pady=10).pack()
        tk.Label(top,
                 text="Jeśli tu widzisz prawdziwego gracza lub item — dopisz go do nabijacy.txt / drop.txt.\n"
                      "Przy następnym uruchomieniu zostanie rozpoznany.",
                 font=("Arial", 9), fg="#888", justify="left").pack(padx=16)

        txt = tk.Text(top, font=("Courier", 10), wrap="word")
        sb  = ttk.Scrollbar(top, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True, padx=8, pady=8)

        if players:
            txt.insert("end", "=== GRACZE ===\n")
            for p in players:
                txt.insert("end", f"  {p}\n")
            txt.insert("end", "\n")
        if items:
            txt.insert("end", "=== ITEMY ===\n")
            for i in items:
                txt.insert("end", f"  {i}\n")
        if not players and not items:
            txt.insert("end", "Brak nierozpoznanych stringów — wszystko dopasowane!\n")

        txt.config(state="disabled")

    def _show_raw(self):
        top = tk.Toplevel(self.root)
        top.title("Surowy tekst OCR")
        top.geometry("720x520")
        txt = tk.Text(top, font=("Courier", 10), wrap="word")
        sb  = ttk.Scrollbar(top, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        for r in self.results:
            txt.insert("end", f"=== {os.path.basename(r['path'])} ===\n")
            txt.insert("end", r["raw_text"])
            txt.insert("end", "\n\n")
        txt.config(state="disabled")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    EchoDropStartScreen(root)
    root.mainloop()


if __name__ == "__main__":
    main()
