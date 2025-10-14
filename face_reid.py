import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist


def face_logger():
    import logging
    return logging.getLogger("integrated_face")

class FaceReID:
    def __init__(self, max_age_s=300, match_threshold=0.42, min_size=80):
        # Person memory (temporary IDs)
        self.people = {}  # pid -> {"emb": np.array, "last_seen": t, "served": False, "seen_count": 0}
        self.next_pid = 1
        self.max_age_s = max_age_s
        self.match_threshold = match_threshold
        self.min_size = min_size  # still useful to ignore tiny detections

        # ---- INSIGHTFACE ONLY (Option B) ----
        # Explicit CPU provider; use a decent detector input size
        self.fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.fa.prepare(ctx_id=-1, det_size=(640, 640))  # you can bump to (800,800) if faces are small

    def _now(self):
        return time.time()

    def _purge_stale(self):
        now = self._now()
        stale = [pid for pid, p in self.people.items() if now - p["last_seen"] > self.max_age_s]
        for pid in stale:
            del self.people[pid]

    def _match(self, emb):
        """Return (best_pid, similarity) if above threshold; else (None, best_sim)."""
        if not self.people:
            return None, None
        gallery = np.stack([p["emb"] for p in self.people.values()], axis=0)  # (N, D)
        # cosine distance = 1 - cosine_similarity
        dists = cdist([emb], gallery, metric="cosine")[0]
        best_idx = int(np.argmin(dists))
        best_pid = list(self.people.keys())[best_idx]
        best_dist = dists[best_idx]
        sim = 1.0 - best_dist  # convert to cosine similarity
        if sim >= self.match_threshold:
            return best_pid, sim
        return None, sim

    def process_frame(self, rgb_frame):
        """
        Run detection + embedding on the FULL frame (Option B).
        Returns: list of dicts:
           [{"pid": int, "bbox": (x1,y1,x2,y2), "served": bool, "similarity": float|None}]
        """
        self._purge_stale()

        H, W = rgb_frame.shape[:2]
        # InsightFace expects BGR images
        bgr_full = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # One pass: detector + embedding
        faces = self.fa.get(bgr_full, max_num=10)  # adjust max_num as you like
        detections = []
        if not faces:
            return detections

        now = self._now()

        for f in faces:
            # f.bbox is [x1, y1, x2, y2] in image coords (float); cast to ints + clamp
            x1, y1, x2, y2 = map(int, f.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            # Optional: ignore very small faces
            if (x2 - x1) < self.min_size or (y2 - y1) < self.min_size:
                continue

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                # Shouldn't happen if recognition model is loaded, but be safe
                continue

            # Match against known people
            match_pid, sim = self._match(emb)
            if match_pid is not None:
                pid = match_pid
                person = self.people[pid]
                # EMA update to stabilize prototype
                person["emb"] = 0.7 * person["emb"] + 0.3 * emb
                person["emb"] /= (np.linalg.norm(person["emb"]) + 1e-9)
                person["last_seen"] = now
                person["seen_count"] += 1
            else:
                # New person
                pid = self.next_pid
                self.next_pid += 1
                self.people[pid] = {
                    "emb": emb.copy(),
                    "last_seen": now,
                    "served": False,
                    "seen_count": 1
                }

            detections.append({
                "pid": pid,
                "bbox": (x1, y1, x2, y2),
                "served": self.people[pid]["served"],
                "similarity": float(sim) if sim is not None else None
            })
        face_logger().debug("FaceReID detections: %s", detections)

        return detections

    def mark_served(self, pid: int):
        if pid in self.people:
            self.people[pid]["served"] = True

