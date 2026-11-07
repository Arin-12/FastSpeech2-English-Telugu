from pathlib import Path

p = Path("English/prepare_features.py")
s = p.read_text(encoding="utf-8")

old = '''    candidates = [p, ROOT / p, OUT / p, ENG / p]'''
new = '''    candidates = [
        p,
        ROOT / p,
        OUT / p,
        ENG / p,
        OUT / "mel" / p.name,
        ENG / "data" / p.name
    ]'''

if old not in s:
    print("Could not find original resolver. Inspect function manually.")
else:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("Path resolver corrected.")
