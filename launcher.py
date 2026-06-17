"""
Launcher — Witcher Tools
Uruchom: python launcher.py
"""

import os
import subprocess
import sys
import tkinter as tk


def run_script(script: str):
    subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), script)])


def main():
    root = tk.Tk()
    root.title("Witcher Tools")
    root.resizable(False, False)

    tk.Label(root, text="Witcher Tools", font=("", 16, "bold")).pack(pady=(24, 16))

    for text, script in [
        ("🔮  Wróżka  (Drop Tracker)",       "app.py"),
        ("⚔️  Echo Wygnańców  (Odpal)",       "odpal.py"),
        ("💬  Echo Drop  (Logi chatu)",        "echo_drop.py"),
    ]:
        tk.Button(root, text=text, width=30, height=2,
                  command=lambda s=script: run_script(s)).pack(pady=6, padx=24)

    tk.Frame(root, height=12).pack()
    root.mainloop()


if __name__ == "__main__":
    main()
