"""
Launcher — Witcher Tools
Uruchom: python launcher.py
"""

import subprocess
import sys
import os
import tkinter as tk


def run_script(script: str):
    subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), script)])


def main():
    root = tk.Tk()
    root.title("Witcher Tools")
    root.resizable(False, False)
    root.geometry("280x160")

    tk.Label(root, text="Witcher Tools", font=("", 14, "bold")).pack(pady=(20, 4))

    tk.Button(root, text="🔮  Wróżka  (Drop Tracker)",
              width=26, height=2,
              command=lambda: run_script("app.py")).pack(pady=6)

    tk.Button(root, text="⚔️  Echo Wygnańców  (Odpal)",
              width=26, height=2,
              command=lambda: run_script("odpal.py")).pack(pady=6)

    root.mainloop()


if __name__ == "__main__":
    main()
