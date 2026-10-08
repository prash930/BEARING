#!/usr/bin/env python
# Check content variable in app.py
with open('D:\\4.+Bearings\\4. Bearings\\IMS\\app.py', 'rb') as f:
    raw = f.read()
text = raw.decode('latin-1', errors='replace')

# Find 'content' occurrences
import re
positions = [m.start() for m in re.finditer(b'content', raw)]
print(f'Total bytes occurrences of "content": {len(positions)}')
for i, pos in enumerate(positions):
    start = max(0, pos - 30)
    end = min(len(raw), pos + 30)
    ctx = raw[start:end].decode('latin-1', errors='replace')
    print(f'Position {pos}: {repr(ctx)}')