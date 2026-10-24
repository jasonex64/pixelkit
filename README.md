# pixelkit

Deterministic pixel-art spec extraction and lint. Used by Claude (the `pixel-art` skill) to build art from Jason's refs without grid breaks, blur, stray colors, broken lines, or muddy shading.

```
pip install -r requirements.txt
python3 pixelkit.py selftest            # must pass (49 tests)
python3 pixelkit.py --help
```

- `GUIDELINES.md`: the workflow and every check, plus how trustworthy each one is (leave-one-out numbers)
- `specs/style.json`: current calibrated rules (measurements and hex codes only)
- `specs/prompt_*.txt`: the rule blocks pasted into generation

Reference images are other artists' work and are **not** committed. Keep them in `refs/` (gitignored) and re-run `extract` / `validate` locally.
