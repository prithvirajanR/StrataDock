#!/usr/bin/env python3
txt = open('generate_manual.py', encoding='utf-8').read()
replacements = {
    '\u2022': '-',
    '\u26a0\ufe0f': '[!]',
    '\u26a0': '[!]',
    '\u2192': '->',
    '\u00b7': '.',
    '\u00a9': '(c)',
    '\u00c5': 'A',
    '\u2014': '--',
    '\u2013': '-',
    '\u0394': 'Delta',
    '\u2265': '>=',
    '\u2264': '<=',
    '\u00d7': 'x',
    '\u2019': "'",
    '\u2018': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u00fc': 'u',
    '\u00e4': 'a',
    '\u00e9': 'e',
    '\u00e0': 'a',
    '\u00c4': 'A',
    '\u2018': "'",
    '\u2080': '0',
    '\u2081': '1',
    '\u2082': '2',
    '\u2083': '3',
    '\u00b2': '2',
    '\u00b3': '3',
    '\u2248': '~',
    '\u2713': 'OK',
    '\u2717': 'X',
    '\u00e9': 'e',
    '\u00fc': 'u',
    '\u00df': 'ss',
    '\u00e0': 'a',
    '\u00e8': 'e',
    '\u00ef': 'i',
    '\u00f4': 'o',
    '\u00fb': 'u',
    '\uff01': '!',
    '\u25b6': '>',
    '\u2764': '<3',
    '\u2729': '*',
    '\u2736': '*',
    '\u26a1': '!',
    '\ufe0f': '',  # variation selector
    '\u200b': '',  # zero-width space
}
for k, v in replacements.items():
    txt = txt.replace(k, v)
# Remove any remaining non-latin-1 chars
cleaned = txt.encode('latin-1', errors='replace').decode('latin-1')
open('generate_manual.py', 'w', encoding='latin-1').write(cleaned)
print('Cleaned OK')
