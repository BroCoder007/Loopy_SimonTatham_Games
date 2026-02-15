#!/usr/bin/env python3
"""
Automated conflict marker removal script.
Removes all git merge conflict markers and keeps the feature branch version.
"""

import re

file_path = "logic/solvers/dynamic_programming_solver.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to match conflict blocks:
# <<<<<<< feature/loop-solver-improvements
# ...code...
# =======
# ...code...
# >>>>>>> main

# Strategy: Remove conflict markers and the "main" version, keep feature version
lines = content.split('\n')
result_lines = []
skip_mode = False
feature_mode = False

for line in lines:
    if line.startswith('<' * 7):  # Start of conflict
        feature_mode = True
        continue
    elif line.startswith('=' * 7):  # Middle separator
        feature_mode = False
        skip_mode = True
        continue
    elif line.startswith('>' * 7):  # End of conflict
        skip_mode = False
        continue
    
    # Keep lines that are not in "main" section
    if not skip_mode:
        result_lines.append(line)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(result_lines))

print(f"Fixed {file_path}")
print("Removed all conflict markers, kept feature/loop-solver-improvements version")
