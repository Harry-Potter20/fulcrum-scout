"""demo: a synthetic defensive shape with a gap in the back line -> fulcrum should find the gap.

Attacking left->right (goal at x=105). A back four with a MISSING/PULLED defender leaves a central-left hole
between the lines; the topological hole-finder should rank that pocket as the top danger hole.
ASCII-renders the pitch so you see it immediately.
"""
from __future__ import annotations
import numpy as np
from core import find_holes, PITCH_L, PITCH_W

# --- a defending team (11) pressing a left->right attack; note the GAP in the back line around y=37 ---
defenders = [
    (104, 34),                                   # GK
    (86, 12), (86, 22),           (86, 50), (86, 60),   # back four with a HOLE where a 3rd CB (y~37) should be
    (72, 16), (72, 30), (72, 44), (72, 56),      # midfield line
    (58, 26), (58, 42),                          # two pressing forwards
]
ball = (66, 40)                                  # attacker on the ball, central-left, ahead of the mid line

field, gx, gy, holes = find_holes(defenders, res=1.5, ball=ball, min_persistence=1.5, top=3)

print("TOP EXPLOITABLE HOLES (topological, danger-ranked):")
for k, h in enumerate(holes, 1):
    print(f"  {k}. ({h['x']:5.1f}, {h['y']:4.1f})  openness={h['openness']:4.1f}m  "
          f"persistence={h['persistence']:4.1f}  danger={h['danger']:.2f}  SCORE={h['score']}")

# --- ASCII pitch render ---
COLS, ROWS = 62, 22
grid = [[" " for _ in range(COLS)] for _ in range(ROWS)]
def cell(x, y):
    return int(np.clip(y / PITCH_W * (ROWS - 1), 0, ROWS - 1)), int(np.clip(x / PITCH_L * (COLS - 1), 0, COLS - 1))
# shade the openness field (light dots = open space)
op = (field - field.min()) / (np.ptp(field) + 1e-9)
for cx in range(COLS):
    for cy in range(ROWS):
        x = cx / (COLS - 1) * PITCH_L; y = cy / (ROWS - 1) * PITCH_W
        i = int(np.clip(x / 1.5, 0, field.shape[0] - 1)); j = int(np.clip(y / 1.5, 0, field.shape[1] - 1))
        if op[i, j] > 0.75: grid[cy][cx] = "."
for (px, py) in defenders:
    r, c = cell(px, py); grid[r][c] = "D"
r, c = cell(*ball); grid[r][c] = "O"
for k, h in enumerate(holes, 1):
    r, c = cell(h["x"], h["y"]); grid[r][c] = str(k)          # 1 = top hole
print("\nPitch (attack ->, goal on right |; D=defender O=ball  1/2/3=holes  .=open space):")
print("+" + "-" * COLS + "+")
for row in grid:
    print("|" + "".join(row) + "|")
print("+" + "-" * COLS + "+")
print("=> the '1' should sit in the gap of the back line (central-left, ahead of the ball).")
