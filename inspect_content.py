#!/usr/bin/env python
# Inspect app.py content variable
with open('app.py', 'rb') as f:
    raw = f.read()

import re
positions = [m.start() for m in re.finditer(b'content', raw)]
print(f'Total occurrences of "content": {len(positions)}')
for i, pos in enumerate(positions):
    start = max(0, pos - 40)
    end = min(len(raw), pos + 40)
    try:
        ctx = raw[start:end].decode('latin-1', errors='replace')
    except:
        ctx = 'Could not decode'
    print(f'Position {pos}: {repr(ctx)}')
    print()
PYEOF