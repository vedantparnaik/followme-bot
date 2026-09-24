"""Top-down animation of a recorded ``Sim`` run (matplotlib, GIF)."""
from __future__ import annotations

import math

STATE_COLORS = {
    "IDLE": "#888888", "ACQUIRE": "#bbbbbb", "FOLLOW": "#3ddc84", "AVOID": "#ffc107",
    "BLOCKED": "#ff7043", "SEARCH": "#29b6f6", "PURSUE": "#26c6da", "WAIT": "#7e57c2", "ESTOP": "#ff1744",
}


def _bounds(sim):
    xs, ys = [], []
    for f in sim.frames:
        xs.append(f.pose[0])
        ys.append(f.pose[1])
        for _, x, y, _, _ in f.people:
            xs.append(x)
            ys.append(y)
    for ob in sim.sc.world.obstacles:
        for x, y in ob.shape.outline(8):
            xs.append(x)
            ys.append(y)
    pad = 1.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def write_gif(sim, path: str, every: int = 3, fps: int = 10, dpi: int = 80) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, patches, transforms

    frames = sim.frames[::every]
    x0, x1, y0, y1 = _bounds(sim)
    body = sim.cfg.body
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * (y1 - y0) / (x1 - x0) + 0.6), dpi=dpi)
    fig.patch.set_facecolor("#16181c")

    def draw(i):
        f = frames[i]
        ax.clear()
        ax.set_facecolor("#1d2026")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.tick_params(colors="#666", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#333")
        ax.grid(color="#2a2e36", lw=0.6)
        for ob in sim.sc.world.obstacles:
            sh = ob.shape
            if hasattr(sh, "r"):
                ax.add_patch(patches.Circle((sh.x, sh.y), sh.r, color=ob.color))
            else:
                ax.add_patch(patches.Rectangle((sh.x - sh.sx / 2, sh.y - sh.sy / 2), sh.sx, sh.sy,
                                               color=ob.color))
            label = ob.name + ("" if ob.camera_label else "  (camera-blind)")
            ax.text(sh.x, sh.y, label, color="#eee", fontsize=6, ha="center", va="center")
        trail = [g.pose for g in frames[max(0, i - 60):i + 1]]
        ax.plot([p[0] for p in trail], [p[1] for p in trail], color="#6ea8ff", lw=1, alpha=0.6)
        for name, x, y, is_t, col in f.people:
            ax.add_patch(patches.Circle((x, y), 0.24, color=col, zorder=4))
            tag = name + (" <- locked" if f.locked == name else "")
            ax.text(x + 0.3, y + 0.3, tag, color=col, fontsize=7, zorder=5)
        if f.obstacles:
            ax.scatter([p[0] for p in f.obstacles], [p[1] for p in f.obstacles], s=4,
                       color="#ff5252", zorder=3)
        x, y, yaw = f.pose
        rect = patches.Rectangle((-body.length / 2, -body.width / 2), body.length, body.width,
                                 color="#e0e0e0", zorder=6)
        rect.set_transform(transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData)
        ax.add_patch(rect)
        ax.plot([x, x + 0.45 * math.cos(yaw)], [y, y + 0.45 * math.sin(yaw)], color="#222",
                lw=2, zorder=7)
        if f.heading is not None:
            ax.annotate("", xy=(x + 1.2 * math.cos(f.heading), y + 1.2 * math.sin(f.heading)),
                        xytext=(x, y), arrowprops=dict(arrowstyle="->", color="#ffc107", lw=1.5),
                        zorder=8)
        col = STATE_COLORS.get(f.state, "#fff")
        ax.set_title(f"{sim.sc.name}   t={f.t:5.1f}s   {f.state}", color=col, fontsize=10,
                     loc="left")

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
