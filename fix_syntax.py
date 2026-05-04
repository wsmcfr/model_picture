import re, os

files = [
    r"D:\model_picture\capture.py",
    r"D:\model_picture\dataset.py",
    r"D:\model_picture\train.py",
    r"D:\model_picture\export_onnx.py",
    r"D:\model_picture\convert_labelme.py",
]

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        # Check if next line is a standalone colon
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            next_stripped = next_line.strip()

            if next_stripped == ":":
                # Merge: append colon to current line, skip next line
                # Preserve indentation context
                merged = stripped + ":"
                fixed.append(merged + "\n")
                i += 2
                continue

        fixed.append(line)
        i += 1

    with open(fpath, "w", encoding="utf-8") as f:
        f.writelines(fixed)

    print(f"Fixed: {fpath}")
