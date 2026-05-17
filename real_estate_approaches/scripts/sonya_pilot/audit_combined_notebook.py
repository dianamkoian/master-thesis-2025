"""
Audit of 00_combined_final.ipynb — целостность и содержимое.

Checks:
  1. Все code-ячейки executed (have execution_count).
  2. Нет error outputs / tracebacks.
  3. Outputs не пустые (для нерасчётных ячеек ОК).
  4. Структура: markdown заголовки и section headers.
  5. Headline numbers — есть ли актуальные (0.7556 PR, 0.7375 PR на команд).
  6. Дубли ячеек по hash содержимого.
  7. Длинные блоки кода без markdown между ними (намёк на «слепленность»).
"""
import json
import hashlib
from pathlib import Path
from collections import Counter

NB = Path(__file__).resolve().parents[2] / "notebooks" / "00_combined_final.ipynb"

nb = json.load(open(NB))
cells = nb["cells"]

print(f"Total cells: {len(cells)}\n")

# 1-3. Code cells: executed, no errors, has outputs
errors = []
unexecuted = []
exec_ords = []
output_lens = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if c.get("execution_count") is None:
        unexecuted.append((i, src[:80].replace("\n", " ")))
    else:
        exec_ords.append(c["execution_count"])
    for out in c.get("outputs", []):
        ot = out.get("output_type", "")
        if ot == "error":
            errors.append((i, out.get("ename"), out.get("evalue", ""),
                            src[:80].replace("\n", " ")))
    output_lens.append((i, sum(len(json.dumps(o)) for o in c.get("outputs", []))))

print(f"=== 1-3. Execution health ===")
print(f"  Unexecuted code cells: {len(unexecuted)}")
for i, s in unexecuted:
    print(f"    cell {i}: {s}")
print(f"  Error outputs: {len(errors)}")
for i, ename, ev, s in errors:
    print(f"    cell {i}: {ename}: {ev[:80]}  source: {s}")
print(f"  Exec_count range: {min(exec_ords)}..{max(exec_ords)} (n={len(exec_ords)})")
print(f"  Exec_count gaps: {len(set(range(min(exec_ords), max(exec_ords)+1)) - set(exec_ords))}")

# 4. Structure: markdown headers
print(f"\n=== 4. Section structure (markdown headers) ===")
sections = []
for i, c in enumerate(cells):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    for line in src.split("\n"):
        if line.startswith("#") and not line.startswith("####"):
            sections.append((i, line.strip()))
            break
print(f"  Sections found: {len(sections)}")
for i, h in sections[:25]:
    print(f"    cell {i:3d}: {h[:120]}")
if len(sections) > 25:
    print(f"    ... +{len(sections)-25} more")

# 5. Headline numbers
print(f"\n=== 5. Headline numbers in notebook ===")
text_all = "\n".join("".join(c.get("source", [])) +
                     "\n".join(json.dumps(o) for o in c.get("outputs", []) if c["cell_type"] == "code")
                     for c in cells)
checks = {
    "0.7556 (M2-FE+ + M2 индив, PR best)": ["0.7556", "0,7556"],
    "0.7375 (M2-FE+ команд PR)":            ["0.7375", "0,7375"],
    "0.7482 (M2-FE+ + M2 команд)":          ["0.7482", "0,7482"],
    "0.7228 (M2 mainline индив)":           ["0.7228", "0,7228"],
    "0.7111 (M2 clean команд)":             ["0.7111", "0,7111"],
    "0.1154 (M2-FE+ команд R@P)":           ["0.1154", "0,1154"],
    "0.2747 (R@P индив headline)":          ["0.2747", "0,2747"],
    "0.2213 (M2-FE+ одиноч индив R@P)":     ["0.2213", "0,2213"],
    "typosquat (должно отсутствовать)":     ["typosquat"],
    "brand_fuzzy (должно отсутствовать)":   ["brand_fuzzy"],
    "clip_cat_sim (должно отсутствовать)":  ["clip_cat_sim"],
    "FADAML (атрибуция)":                   ["FADAML"],
    "M2-FE+":                               ["M2-FE+"],
}
for label, patterns in checks.items():
    found = any(p in text_all for p in patterns)
    sym = "✓" if found else "✗"
    print(f"  {sym} {label}: {'found' if found else 'NOT FOUND'}")

# 6. Duplicate cells by source-hash
print(f"\n=== 6. Duplicate cells ===")
hashes = Counter()
for c in cells:
    src = "".join(c.get("source", [])).strip()
    if not src:
        continue
    h = hashlib.md5(src.encode()).hexdigest()[:10]
    hashes[h] += 1
dups = {h: cnt for h, cnt in hashes.items() if cnt > 1}
print(f"  Duplicate hashes: {len(dups)}")
for h, cnt in list(dups.items())[:5]:
    # Find one example
    for i, c in enumerate(cells):
        src = "".join(c.get("source", [])).strip()
        if src and hashlib.md5(src.encode()).hexdigest()[:10] == h:
            print(f"    {h} ×{cnt}: cell {i}: {src[:80].replace(chr(10), ' ')}")
            break

# 7. Long code-only stretches (no markdown for N+ code cells in a row)
print(f"\n=== 7. Long code-only stretches (>=5 code cells without markdown) ===")
streak = 0
streak_start = -1
streaks = []
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        if streak == 0:
            streak_start = i
        streak += 1
    else:
        if streak >= 5:
            streaks.append((streak_start, i - 1, streak))
        streak = 0
if streak >= 5:
    streaks.append((streak_start, len(cells) - 1, streak))
print(f"  Long stretches: {len(streaks)}")
for start, end, length in streaks:
    print(f"    cells {start}..{end} ({length} code)")

# 8. Total output size + execution timing if any
print(f"\n=== 8. Output statistics ===")
total_out = sum(L for _, L in output_lens)
print(f"  Total output bytes: {total_out:,} ({total_out/1024:.0f} KB)")
biggest = sorted(output_lens, key=lambda x: -x[1])[:5]
for i, L in biggest:
    src = "".join(cells[i].get("source", []))[:80].replace("\n", " ")
    print(f"    cell {i}: {L:,} bytes — {src}")
