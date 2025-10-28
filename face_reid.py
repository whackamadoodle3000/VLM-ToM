import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist


def face_logger():
    import logging
    return logging.getLogger("integrated_face")

class FaceReID:
    def __init__(
        self,
        max_age_s=300,
        match_threshold=0.42,
        min_match_threshold=0.3,
        recent_return_window=10.0,
        proximity_radius=220.0,
        proximity_bonus=0.15,
        adaptive_rate=0.18,
        min_size=80,
    ):
        # Person memory (temporary IDs)
        self.people = {}
        self.next_pid = 1
        self.max_age_s = max_age_s
        self.match_threshold = match_threshold
        self.min_match_threshold = min_match_threshold
        self.recent_return_window = recent_return_window
        self.proximity_radius = proximity_radius
        self.proximity_bonus = proximity_bonus
        self.adaptive_rate = adaptive_rate
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
            face_logger().debug("Purging stale track for PID %s", pid)
            del self.people[pid]

    def _best_match(self, emb):
        """Return (best_pid, similarity) using cosine distance (no thresholding)."""
        if not self.people:
            return None, None
        gallery = np.stack([p["emb"] for p in self.people.values()], axis=0)  # (N, D)
        # cosine distance = 1 - cosine_similarity
        dists = cdist([emb], gallery, metric="cosine")[0]
        best_idx = int(np.argmin(dists))
        best_pid = list(self.people.keys())[best_idx]
        best_dist = dists[best_idx]
        sim = 1.0 - best_dist  # convert to cosine similarity
        return best_pid, sim

    def _compute_threshold(self, pid, now, detection_center):
        threshold = self.match_threshold
        person = self.people[pid]

        # Recently seen person gets a relaxed threshold
        time_gap = now - person["last_seen"]
        if time_gap < self.recent_return_window:
            factor = (self.recent_return_window - time_gap) / max(self.recent_return_window, 1.0)
            threshold -= self.adaptive_rate * factor

        # If the detection is spatially close to the last known bbox, relax further
        prev_bbox = person.get("last_bbox")
        if prev_bbox is not None:
            px1, py1, px2, py2 = prev_bbox
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            cx, cy = detection_center
            dist = np.hypot(cx - pcx, cy - pcy)
            if dist < self.proximity_radius:
                proximity_factor = 1.0 - (dist / max(self.proximity_radius, 1.0))
                threshold -= self.proximity_bonus * proximity_factor

        return max(self.min_match_threshold, threshold)

    def _update_track_embeddings(self, pid, emb):
        person = self.people[pid]
        # Slightly faster adaptation than before to capture pose changes
        alpha = 0.45
        person["emb"] = (1 - alpha) * person["emb"] + alpha * emb
        person["emb"] /= (np.linalg.norm(person["emb"]) + 1e-9)
        person["seen_count"] += 1

    def _register_new_person(self, emb, now, bbox):
        pid = self.next_pid
        self.next_pid += 1
        norm_emb = emb / (np.linalg.norm(emb) + 1e-9)
        self.people[pid] = {
            "emb": norm_emb,
            "last_seen": now,
            "served": False,
            "seen_count": 1,
            "last_bbox": bbox,
        }
        face_logger().debug("Created new PID %s", pid)
        return pid

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
            match_pid, sim = self._best_match(emb)
            detection_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if match_pid is not None and sim is not None:
                adaptive_threshold = self._compute_threshold(match_pid, now, detection_center)
                if sim >= adaptive_threshold:
                    pid = match_pid
                    person = self.people[pid]
                    self._update_track_embeddings(pid, emb)
                    person["last_seen"] = now
                    person["last_bbox"] = (x1, y1, x2, y2)
                    face_logger().debug(
                        "Matched PID %s with sim %.3f (threshold %.3f)",
                        pid,
                        sim,
                        adaptive_threshold,
                    )
                else:
                    face_logger().debug(
                        "Similarity %.3f below adaptive threshold %.3f; creating new PID",
                        sim,
                        adaptive_threshold,
                    )
                    pid = self._register_new_person(emb, now, (x1, y1, x2, y2))
            else:
                pid = self._register_new_person(emb, now, (x1, y1, x2, y2))

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

