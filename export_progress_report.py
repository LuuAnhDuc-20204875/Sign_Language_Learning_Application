import json
import sys
from pathlib import Path

def summarize_progress(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    letters = obj.get("letters", {})
    words = obj.get("words", {})

    def agg(d):
        total = sum(v.get("total", 0) for v in d.values())
        correct = sum(v.get("correct", 0) for v in d.values())
        acc = (correct / total * 100.0) if total else 0.0
        return total, correct, acc

    lt, lc, lacc = agg(letters)
    wt, wc, wacc = agg(words)

    print(f"FILE: {path.resolve()}")
    print(f"LETTERS: total={lt} correct={lc} acc={lacc:.2f}%")
    print(f"WORDS:   total={wt} correct={wc} acc={wacc:.2f}%")

    hard = []
    for k, v in letters.items():
        t = int(v.get("total", 0))
        c = int(v.get("correct", 0))
        if t >= 5:
            hard.append((c / t * 100.0, t, k, c))
    hard.sort(key=lambda x: x[0])

    print("\nHardest letters (min 5 attempts):")
    for acc, t, k, c in hard[:10]:
        print(f"  {k:>2} : acc={acc:>6.1f}%  total={t:<4} correct={c}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_progress_report.py <path_to_progress_json>")
        sys.exit(1)
    summarize_progress(Path(sys.argv[1]))
