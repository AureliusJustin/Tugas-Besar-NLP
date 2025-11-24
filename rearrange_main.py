
with open('src/main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1-based line numbers to 0-based indices
# Part 1: Lines 1 to 107
# Index 0 to 107 (exclusive)
part1 = lines[0:107]

# Part Helpers: Lines 108 to 277
# Index 107 to 277 (exclusive)
part_helpers = lines[107:277]

# Part Methods: Lines 278 to 473
# Index 277 to 473 (exclusive)
part_methods = lines[277:473]

# Part Rest: Lines 474 to End
# Index 473 to End
part_rest = lines[473:]

# Reconstruct
new_content = part1 + part_methods + part_helpers + part_rest

with open('src/main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_content)

print("Successfully rearranged src/main.py")

