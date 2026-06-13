# Witcher Drop Tracker

Narzędzie do automatycznego wykrywania przedmiotów i ilości ze screenshotów okna handlu w grze The Witcher Online. Uruchamia GUI z wbudowanym plikiem-pickerem, detekcją w tle i weryfikacją wyników.

## Wymagania

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (do wykrywania yang)

```bash
pip install -r requirements.txt
```

Na macOS Tesseract można zainstalować przez Homebrew:

```bash
brew install tesseract
```

## Użycie

```bash
python app.py
```

1. Kliknij **Wybierz screenshoty...** i wskaż pliki PNG/JPG z okna handlu.
2. Kliknij **Analizuj** — detekcja uruchamia się w tle, pasek postępu pokazuje kolejne pliki.
3. Przejrzyj slajdy, popraw liczby i yang w każdym screenshocie.
4. Po ostatnim slajdzie wyświetli się zsumowane podsumowanie dropu.
5. Kliknij **Zapisz JSON**, żeby wyeksportować wyniki do pliku.

Możesz przetestować aplikację na przykładowym screenshocie z folderu [`examples/`](examples/).

## Struktura

```
app.py         # główny plik — uruchamiaj to
examples/      # przykładowy screenshot do testów
templates/
  items/       # obrazy przedmiotów używane jako wzorce (PNG)
  numbers/     # cyfry do detekcji ilości (PNG, ~7×8px)
screenshots/   # screenshoty okna handlu (wykluczone z repo)
results/       # wyniki detekcji (wykluczone z repo)
```

## Dodawanie nowych przedmiotów

1. Dodaj wycięty obraz przedmiotu (PNG) do `templates/items/`. Nazwa pliku (bez rozszerzenia) stanie się kluczem w wynikowym JSON.
2. Opcjonalnie dodaj czytelną nazwę do słownika `ITEM_NAMES` w `app.py`.
3. Jeśli przedmiot generuje fałszywe trafienia, dodaj go do `ITEM_THRESHOLDS` w `app.py` z wyższym progiem (np. `0.85`).

## Pakowanie jako .exe

```bash
pip install pyinstaller
pyinstaller --onefile --windowed \
    --add-data "templates:templates" \
    app.py
```

Plik wykonywalny pojawi się w `dist/app.exe`.
