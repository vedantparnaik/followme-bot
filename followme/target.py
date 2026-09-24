"""Who are we following?

Lock once, then follow that identity, not "whoever is biggest".

1. Lock: the biggest, most central person with a tracker ID, and only after
   the same candidate has persisted for ``lock_stable_s``.
2. Track by ID every frame. Anyone seen in the same frame as the target is
   remembered as "definitely not them".
3. If the ID disappears: coast on the last fix for ``lost_grace_s``, and try
   re-identification by appearance on every new track.
4. If the ID is still present but stops looking like the target for several
   frames, the tracker swapped identities (someone crossed in front). Treat
   the target as lost and let re-ID find them again.

A lost target is never replaced by a stranger automatically. Only an explicit
``reset()`` allows locking a new person.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from followme.config import Config
from followme.perception import Detection, clip_flags, max_overlap
from followme.reid import Gallery


@dataclass
class TrackResult:
    det: Optional[Detection]   # current (or coasted) target detection
    how: str                   # "none", "id", "reid", "coast", "lost"
    target_id: Optional[int]
    lost_for: float = 0.0
    event: str = ""
    reid_score: float = 0.0


def pick_lock(persons: Sequence[Detection], width: int) -> Optional[Detection]:
    best, best_score = None, -1.0
    for d in persons:
        if d.track_id is None:
            continue
        x, y, w, h = d.box
        cx = x + w / 2.0
        centrality = 1.0 - min(abs(cx - width / 2.0) / (width / 2.0), 1.0)
        score = (w * h) * (0.5 + 0.5 * centrality)
        if score > best_score:
            best, best_score = d, score
    return best


class TargetTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gallery = Gallery(cfg.reid.gallery)
        self.reset()

    def reset(self) -> None:
        self.locked = False
        self.target_id: Optional[int] = None
        self.last_det: Optional[Detection] = None
        self.last_seen = -1e9
        self.known_others: Set[int] = set()
        self.gallery.clear()
        self._pending_id: Optional[int] = None
        self._pending_since = 0.0
        self._reid_id: Optional[int] = None
        self._reid_count = 0
        self._mismatch = 0
        self._score_ema = 1.0

    # ------------------------------------------------------------------
    def update(self, dets: Sequence[Detection], now: float) -> TrackResult:
        persons = [d for d in dets if d.label == "person" and d.track_id is not None]
        if not self.locked:
            return self._try_lock(persons, now)

        event = ""
        match = None
        if self.target_id is not None:
            match = next((p for p in persons if p.track_id == self.target_id), None)
        if match is not None and (self._teleported(match, now) or self._looks_swapped(match)):
            self.known_others.add(match.track_id)
            self.target_id = None
            self.last_det = None          # that box was the other person: don't coast on it
            match = None
            event = "id swap - target lost"

        if match is not None:
            self.last_det, self.last_seen = match, now
            others = [p for p in persons if p is not match]
            self.known_others.update(p.track_id for p in others)
            self._maybe_learn(match, [d.box for d in dets if d is not match])
            self._reid_id, self._reid_count = None, 0
            return TrackResult(match, "id", self.target_id, 0.0, event)

        lost_for = now - self.last_seen
        found = self._try_reid(persons)
        if found is not None:
            det, score = found
            old = self.target_id
            self.target_id = det.track_id
            self.last_det, self.last_seen = det, now
            self._mismatch, self._score_ema = 0, 1.0
            self.gallery.add(det.feature)
            was = "" if old is None else f"was #{old}, "
            ev = f"re-identified #{det.track_id} ({was}sim {score:.2f})"
            return TrackResult(det, "reid", self.target_id, 0.0, ev, score)

        if lost_for <= self.cfg.follow.lost_grace_s and self.last_det is not None:
            return TrackResult(self.last_det, "coast", self.target_id, lost_for, event)
        return TrackResult(None, "lost", self.target_id, lost_for, event)

    # ------------------------------------------------------------------
    def _try_lock(self, persons: List[Detection], now: float) -> TrackResult:
        cand = pick_lock(persons, self.cfg.camera.width)
        if cand is None:
            self._pending_id = None
            return TrackResult(None, "none", None)
        event = ""
        if cand.track_id != self._pending_id:
            self._pending_id, self._pending_since = cand.track_id, now
            event = f"candidate #{cand.track_id}"
        if now - self._pending_since < self.cfg.reid.lock_stable_s:
            return TrackResult(None, "none", None, event=event)
        self.locked = True
        self.target_id = cand.track_id
        self.last_det, self.last_seen = cand, now
        self.known_others = {p.track_id for p in persons if p is not cand}
        self.gallery.add(cand.feature)
        return TrackResult(cand, "id", self.target_id, 0.0, f"locked #{cand.track_id}")

    def _teleported(self, match: Detection, now: float) -> bool:
        """People cannot jump. If the box for the same ID changes size or
        position far more than one frame of walking allows, the tracker has
        handed the ID to someone else. Works even when clothes look alike."""
        prev = self.last_det
        if prev is None or now - self.last_seen > 0.15:
            return False
        cam, f = self.cfg.camera, self.cfg.follow
        c0 = clip_flags(prev.box, cam, f.edge_margin)
        c1 = clip_flags(match.box, cam, f.edge_margin)
        if c0[2] or c0[3] or c1[2] or c1[3]:
            return False          # cut off at the side: size and centre both lie
        px, _, pw, ph = prev.box
        x, _, w, h = match.box
        # Compare like with like: widths if either box is cut at top/bottom.
        if c0[0] or c0[1] or c1[0] or c1[1]:
            r0, r1 = f.person_w * cam.fx / max(pw, 1.0), f.person_w * cam.fx / max(w, 1.0)
        else:
            r0, r1 = f.person_h * cam.fy / max(ph, 1.0), f.person_h * cam.fy / max(h, 1.0)
        shift = abs((x + w / 2) - (px + pw / 2)) / max(pw, 1.0)
        return abs(r1 - r0) > self.cfg.reid.jump_m or shift > self.cfg.reid.jump_shift

    def _looks_swapped(self, match: Detection) -> bool:
        """The tracked ID has stopped looking like the target (smoothed)."""
        r = self.cfg.reid
        if not r.enabled or len(self.gallery) < 5 or match.feature is None:
            return False
        self._score_ema = 0.7 * self._score_ema + 0.3 * self.gallery.score(match.feature)
        self._mismatch = self._mismatch + 1 if self._score_ema < r.reject else 0
        if self._mismatch >= r.reject_frames:
            self._mismatch, self._score_ema = 0, 1.0
            return True
        return False

    def _maybe_learn(self, det: Detection, other_boxes) -> None:
        """Only learn from clean views: unclipped, big enough, not overlapping,
        and still looking like the target (never learn a swapped identity)."""
        if det.feature is None:
            return
        cam, r = self.cfg.camera, self.cfg.reid
        if len(self.gallery) >= 5 and self.gallery.score(det.feature) < r.reject:
            return
        if any(clip_flags(det.box, cam, self.cfg.follow.edge_margin)):
            return
        if det.box[3] < r.min_box_h:
            return
        if max_overlap(det.box, other_boxes) > 0.02:
            return
        self.gallery.add(det.feature)

    def _try_reid(self, persons: List[Detection]):
        r = self.cfg.reid
        if not r.enabled or not len(self.gallery):
            return None
        scored = sorted(
            ((self.gallery.score(p.feature), p) for p in persons if p.feature is not None),
            key=lambda sp: sp[0], reverse=True)
        cands = [(s, p) for s, p in scored if p.track_id not in self.known_others]
        if not cands:
            self._reid_id, self._reid_count = None, 0
            return None
        best_s, best = cands[0]
        # People already known to be someone else can't be confused with the
        # target, however alike they look; only compare against unknowns.
        runner_up = cands[1][0] if len(cands) > 1 else 0.0
        if best_s < r.accept or best_s - runner_up < r.margin:
            self._reid_id, self._reid_count = None, 0
            return None
        if best.track_id == self._reid_id:
            self._reid_count += 1
        else:
            self._reid_id, self._reid_count = best.track_id, 1
        if self._reid_count < r.confirm_frames:
            return None
        self._reid_id, self._reid_count = None, 0
        return best, best_s
