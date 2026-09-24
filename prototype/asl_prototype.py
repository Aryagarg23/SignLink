"""
Explanatory diagram, not a results chart. There is no dataset or accuracy number
behind SignLink's matcher (see the <script type="module"> block in app.html) — it's
a hand-coded if/else tree over five joint-angle sums from MediaPipe hand landmarks.
This script draws the recognition pipeline end to end: camera frame -> MediaPipe's
21 hand landmarks -> joint angles via vector algebra -> match against the hand-coded
letter table -> letter -> caption in the call.

The 21-landmark topology and connections are MediaPipe's real hand model. The pose
itself (a static "L" shape) and its joint coordinates are computed with the same
forward-kinematics + vector-angle math app.html uses (see `angle()` in app.html),
applied to a stated toy pose — real math, not a measurement.

Run: python3 -m pip install matplotlib numpy
     python3 prototype/asl_prototype.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#282215"
BORDER = "#c6b99f"
BLUE = "#3b42db"
ORANGE = "#e85b30"
RUST = "#c2491d"
PURPLE = "#6f2f96"

# --- MediaPipe's real 21-point hand topology -------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm
]

# Per-finger forward-kinematics spec: an anchor bone from the wrist, then three
# more bones per finger (matches MediaPipe's CMC/MCP/PIP/DIP/TIP joint chain).
# `curl` in [0, 1] bends each successive bone away from its resting direction —
# 0 keeps the finger straight, 1 folds it toward the palm. This is a stated toy
# pose (a static "L" handshape) built with real forward kinematics, not a capture.
FINGER_SPEC = {
    "thumb":  dict(landmarks=[1, 2, 3, 4],     anchor_dir=150, anchor_len=0.55,
                   bone_lens=[0.42, 0.32, 0.26], base_dir=140, max_bend=55, curl=0.08),
    "index":  dict(landmarks=[5, 6, 7, 8],     anchor_dir=100, anchor_len=0.95,
                   bone_lens=[0.45, 0.28, 0.22], base_dir=97,  max_bend=90, curl=0.12),
    "middle": dict(landmarks=[9, 10, 11, 12],  anchor_dir=90,  anchor_len=1.00,
                   bone_lens=[0.50, 0.30, 0.24], base_dir=90,  max_bend=95, curl=0.85),
    "ring":   dict(landmarks=[13, 14, 15, 16], anchor_dir=78,  anchor_len=0.95,
                   bone_lens=[0.46, 0.29, 0.22], base_dir=82,  max_bend=95, curl=0.85),
    "pinky":  dict(landmarks=[17, 18, 19, 20], anchor_dir=65,  anchor_len=0.82,
                   bone_lens=[0.38, 0.24, 0.20], base_dir=70,  max_bend=90, curl=0.85),
}


def build_hand():
    """Forward-kinematics chain from the wrist out to each fingertip."""
    pts = {0: np.array([0.0, 0.0])}
    for spec in FINGER_SPEC.values():
        anchor_rad = np.radians(spec["anchor_dir"])
        anchor = pts[0] + spec["anchor_len"] * np.array(
            [np.cos(anchor_rad), np.sin(anchor_rad)]
        )
        anchor_idx = spec["landmarks"][0]
        pts[anchor_idx] = anchor

        current = anchor
        current_dir = spec["base_dir"]
        bend_per_seg = spec["curl"] * spec["max_bend"] / len(spec["bone_lens"])
        for i, bone_len in enumerate(spec["bone_lens"]):
            current_dir -= bend_per_seg
            rad = np.radians(current_dir)
            current = current + bone_len * np.array([np.cos(rad), np.sin(rad)])
            pts[spec["landmarks"][i + 1]] = current
    return pts


def joint_angle(p1, p2, p3):
    """Same formula as app.html's angle(): arccos of the bone-to-bone turn."""
    v1 = p2 - p1
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


HAND = build_hand()

# --- figure ------------------------------------------------------------
fig = plt.figure(figsize=(18, 5.6))
fig.suptitle(
    "From camera frame to on-screen letter: SignLink's recognition pipeline",
    x=0.02, y=0.98, ha="left", fontsize=13, fontweight="medium", color=INK,
)
fig.text(
    0.98, 0.98, "mechanism as built", ha="right", va="top",
    fontsize=9, style="italic", color=RUST,
)

n_stages = 6
bw, gw = 0.132, 0.032
x0 = 0.018
lefts = [x0 + i * (bw + gw) for i in range(n_stages)]
bottom, height = 0.14, 0.68


def new_panel(i, title, tag):
    ax = fig.add_axes([lefts[i], bottom, bw, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor=INK, linewidth=1.1)
    )
    ax.text(0.06, 0.94, title, fontsize=9.5, fontweight="medium", color=INK, va="top")
    ax.text(0.06, 0.035, tag, fontsize=7.5, style="italic", color=RUST, va="bottom")
    return ax


def connect_panels(i):
    """Arrow in the figure-fraction gap between panel i and panel i+1."""
    x_start = lefts[i] + bw
    x_end = lefts[i + 1]
    y = bottom + height / 2
    arrow = FancyArrowPatch(
        (x_start, y), (x_end, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=INK,
        clip_on=False,
    )
    fig.add_artist(arrow)


# Panel 1: camera frame -------------------------------------------------
ax1 = new_panel(0, "camera frame", "concept sketch")
ax1.add_patch(Rectangle((0.16, 0.2), 0.68, 0.58, fill=False, edgecolor=INK, linewidth=1.4))
# viewfinder corner brackets
for cx, cy, dx, dy in [(0.16, 0.78, 1, -1), (0.84, 0.78, -1, -1), (0.16, 0.2, 1, 1), (0.84, 0.2, -1, 1)]:
    ax1.plot([cx, cx + 0.05 * dx], [cy, cy], color=BLUE, linewidth=1.6)
    ax1.plot([cx, cx], [cy, cy + 0.05 * dy], color=BLUE, linewidth=1.6)
ax1.add_patch(plt.Circle((0.22, 0.72), 0.02, color=ORANGE))
ax1.text(0.27, 0.72, "live", fontsize=7.5, color=ORANGE, va="center")
ax1.text(0.5, 0.14, "peer's video, read frame by frame", fontsize=7,
          ha="center", va="top", color=INK)

# Panel 2: MediaPipe 21 landmarks ---------------------------------------
ax2 = new_panel(1, "MediaPipe: 21 hand landmarks", "toy pose, real topology")
hand_ax = fig.add_axes([lefts[1] + 0.012, bottom + 0.16, bw - 0.024, height - 0.34])
hand_ax.set_xlim(-1.9, 1.9)
hand_ax.set_ylim(-0.4, 2.2)
hand_ax.set_aspect("equal")
hand_ax.axis("off")
for a, b in HAND_CONNECTIONS:
    p, q = HAND[a], HAND[b]
    hand_ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, linewidth=1.3, zorder=2)
xs = [p[0] for p in HAND.values()]
ys = [p[1] for p in HAND.values()]
hand_ax.scatter(xs, ys, s=16, color=BLUE, edgecolor=INK, linewidth=0.4, zorder=3)
hand_ax.scatter(*HAND[0], s=26, color=ORANGE, edgecolor=INK, linewidth=0.5, zorder=4)
hand_ax.text(HAND[0][0], HAND[0][1] - 0.22, "0: wrist", fontsize=6.5, ha="center", color=INK)
ax2.text(0.5, 0.09, "static \"L\" pose, real MediaPipe hand model", fontsize=7,
          ha="center", color=INK)

# Panel 3: joint angles via vector algebra -------------------------------
ax3 = new_panel(2, "joint angles: vector algebra", "toy example, real math")
vec_ax = fig.add_axes([lefts[2] + 0.012, bottom + 0.30, bw - 0.024, height - 0.46])
vec_ax.set_xlim(-0.3, 1.6)
vec_ax.set_ylim(-0.2, 1.4)
vec_ax.set_aspect("equal")
vec_ax.axis("off")
p1, p2, p3 = HAND[5] - HAND[5], HAND[6] - HAND[5], HAND[7] - HAND[5]
scale = 1.1 / max(p3[1], p2[1], 1e-6)
p1, p2, p3 = p1 * scale, p2 * scale, p3 * scale
vec_ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
vec_ax.annotate("", xy=p3, xytext=p2, arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.6))
vec_ax.scatter([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], s=18, color=INK, zorder=4)
vec_ax.text(p1[0] - 0.06, p1[1] - 0.06, "L5", fontsize=7, ha="right", color=INK)
vec_ax.text(p2[0] + 0.05, p2[1], "L6", fontsize=7, color=INK)
vec_ax.text(p3[0] + 0.05, p3[1], "L7", fontsize=7, color=INK)
vec_ax.text(0.35, -0.02, "v1", fontsize=7.5, color=BLUE)
vec_ax.text(1.0, 0.7, "v2", fontsize=7.5, color=PURPLE)
index_sum = joint_angle(HAND[5], HAND[6], HAND[7]) + joint_angle(HAND[6], HAND[7], HAND[8])
middle_sum = joint_angle(HAND[9], HAND[10], HAND[11]) + joint_angle(HAND[10], HAND[11], HAND[12])
ax3.text(
    0.5, 0.20,
    r"$\theta = \arccos\left(\dfrac{v_1\!\cdot\!v_2}{|v_1||v_2|}\right)$,"
    r" summed over 2 joints/finger",
    fontsize=7.5, ha="center", color=INK,
)
ax3.text(
    0.06, 0.075,
    f"index angle-sum ≈ {index_sum:.0f}° (< 50° → extended)\n"
    f"middle angle-sum ≈ {middle_sum:.0f}° (> 50° → curled)",
    fontsize=6.6, va="bottom", color=INK,
)

# Panel 4: match against hand-coded letter table -------------------------
ax4 = new_panel(3, "match: hand-coded letter table", "mechanism as built")
rows = [
    ("A", "all curled, thumb low", False),
    ("L", "index straight, rest curled", True),
    ("V", "index+middle straight", False),
    ("Y", "thumb+pinky straight", False),
]
y = 0.80
for letter, cond, matched in rows:
    if matched:
        ax4.add_patch(Rectangle((0.03, y - 0.055), 0.94, 0.11, color="#ded4b8", zorder=1))
    ax4.text(0.08, y, letter, fontsize=9, fontweight="medium",
              color=ORANGE if matched else INK, zorder=2, va="center")
    ax4.text(0.24, y, cond, fontsize=6.5, color=INK, va="center", zorder=2)
    y -= 0.155
ax4.text(0.5, 0.12, "if/else thresholds on the 5\nangle sums (app.html)", fontsize=6.5,
          ha="center", color=INK, va="top")

# Panel 5: letter ---------------------------------------------------------
ax5 = new_panel(4, "matched letter", "mechanism as built")
ax5.text(0.5, 0.52, "L", fontsize=46, ha="center", va="center", color=BLUE, fontweight="bold")

# Panel 6: caption in the call ---------------------------------------------
ax6 = new_panel(5, "caption in the WebRTC call", "concept sketch")
ax6.add_patch(Rectangle((0.08, 0.28), 0.84, 0.52, fill=False, edgecolor=INK, linewidth=1.3))
ax6.add_patch(Rectangle((0.08, 0.28), 0.84, 0.14, color="#ded4b8", zorder=1))
ax6.text(0.5, 0.35, "L", fontsize=13, ha="center", va="center", color=ORANGE,
          fontweight="bold", zorder=2)
ax6.text(0.5, 0.62, "remote peer's\nvideo tile", fontsize=7, ha="center", va="center", color=INK)
ax6.text(0.5, 0.16, "letter drawn on a canvas\nover the video (PeerJS)", fontsize=6.8,
          ha="center", va="top", color=INK)

for i in range(n_stages - 1):
    connect_panels(i)

fig.text(
    0.02, 0.02,
    "Static poses only: this reads one frame at a time, on angle sums summarized "
    "per finger. Motion letters (J, Z) need a trajectory across frames and are out of scope.",
    fontsize=8, color=INK, ha="left",
)

fig.savefig(os.path.join(FIGDIR, "recognition_flow.png"), dpi=200)
plt.close(fig)

print("Saved figure to", FIGDIR)
