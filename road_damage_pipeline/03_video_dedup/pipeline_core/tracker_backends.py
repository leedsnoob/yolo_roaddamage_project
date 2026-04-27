import math
from collections import defaultdict

import cv2
import numpy as np

from pipeline_core.dedup_utils import box_iou_xyxy


def greedy_linear_assignment(cost_matrix):
    matches = []
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    used_rows = set()
    used_cols = set()
    flat_indices = np.argsort(cost_matrix, axis=None)
    rows, cols = cost_matrix.shape
    for flat_index in flat_indices:
        row = int(flat_index // cols)
        col = int(flat_index % cols)
        if row in used_rows or col in used_cols:
            continue
        matches.append([row, col])
        used_rows.add(row)
        used_cols.add(col)
    return np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)


def linear_assignment(cost_matrix):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return greedy_linear_assignment(cost_matrix)
    rows, cols = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(rows, cols)), dtype=int)


def iou_batch(detections, trackers):
    if len(detections) == 0 or len(trackers) == 0:
        return np.zeros((len(detections), len(trackers)), dtype=float)
    bboxes2 = np.expand_dims(trackers, 0)
    bboxes1 = np.expand_dims(detections, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    union = (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - inter
    )
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def speed_direction(previous_box, current_box):
    cx1, cy1 = (previous_box[0] + previous_box[2]) / 2.0, (previous_box[1] + previous_box[3]) / 2.0
    cx2, cy2 = (current_box[0] + current_box[2]) / 2.0, (current_box[1] + current_box[3]) / 2.0
    dx = cx2 - cx1
    dy = cy2 - cy1
    norm = math.hypot(dx, dy) + 1e-6
    return np.array([dy / norm, dx / norm], dtype=float)


def speed_direction_batch(detections, previous_obs):
    if len(detections) == 0 or len(previous_obs) == 0:
        return np.zeros((len(previous_obs), len(detections))), np.zeros((len(previous_obs), len(detections)))
    tracks = previous_obs[..., np.newaxis]
    cx1 = (detections[:, 0] + detections[:, 2]) / 2.0
    cy1 = (detections[:, 1] + detections[:, 3]) / 2.0
    cx2 = (tracks[:, 0] + tracks[:, 2]) / 2.0
    cy2 = (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = cx1 - cx2
    dy = cy1 - cy2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm


def k_previous_obs(observations, current_age, delta_t):
    if not observations:
        return np.array([-1, -1, -1, -1, -1], dtype=float)
    for step in range(delta_t, 0, -1):
        key = current_age - step
        if key in observations:
            return observations[key]
    latest = max(observations.keys())
    return observations[latest]


def associate_ocsort(detections, trackers, iou_threshold, velocities, previous_obs, inertia):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections[:, :4], trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)

    direction_y, direction_x = speed_direction_batch(detections[:, :4], previous_obs)
    inertia_y = np.repeat(velocities[:, 0][:, np.newaxis], direction_y.shape[1], axis=1)
    inertia_x = np.repeat(velocities[:, 1][:, np.newaxis], direction_x.shape[1], axis=1)
    diff_angle_cos = np.clip(inertia_x * direction_x + inertia_y * direction_y, -1.0, 1.0)
    diff_angle = np.arccos(diff_angle_cos)
    angle_score = ((np.pi / 2.0) - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0], dtype=float)
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0.0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], direction_x.shape[1], axis=1)
    angle_cost = (valid_mask * angle_score).T * inertia
    angle_cost = angle_cost * scores

    if min(iou_matrix.shape) > 0:
        adjacency = (iou_matrix > iou_threshold).astype(np.int32)
        if adjacency.sum(1).max() == 1 and adjacency.sum(0).max() == 1:
            matched_indices = np.stack(np.where(adjacency), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_cost))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [idx for idx in range(len(detections)) if idx not in matched_indices[:, 0]]
    unmatched_trks = [idx for idx in range(len(trackers)) if idx not in matched_indices[:, 1]]
    matches = []
    for det_idx, trk_idx in matched_indices:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_dets.append(int(det_idx))
            unmatched_trks.append(int(trk_idx))
            continue
        matches.append([int(det_idx), int(trk_idx)])

    return (
        np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int),
        np.array(unmatched_dets, dtype=int),
        np.array(unmatched_trks, dtype=int),
    )


class SparseFlowCMC:
    def __init__(self, minimum_features=10):
        self.minimum_features = minimum_features
        self.prev_gray = None
        self.prev_points = None
        self.flow_params = {
            "maxCorners": 3000,
            "qualityLevel": 0.01,
            "minDistance": 1,
            "blockSize": 3,
            "useHarrisDetector": False,
            "k": 0.04,
        }

    def compute_affine(self, frame_bgr, boxes_xyxy):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.ones_like(gray, dtype=np.uint8)
        for box in boxes_xyxy:
            x1, y1, x2, y2 = [max(0, int(round(v))) for v in box]
            x2 = min(x2, mask.shape[1])
            y2 = min(y2, mask.shape[0])
            mask[y1:y2, x1:x2] = 0

        points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.flow_params)
        affine = np.eye(2, 3, dtype=float)
        if self.prev_gray is None or self.prev_points is None or points is None:
            self.prev_gray = gray
            self.prev_points = points
            return affine

        matched_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None)
        if matched_points is None or status is None:
            self.prev_gray = gray
            self.prev_points = points
            return affine

        matched_points = matched_points.reshape(-1, 2)
        previous_points = self.prev_points.reshape(-1, 2)
        status = status.reshape(-1).astype(bool)
        previous_points = previous_points[status]
        matched_points = matched_points[status]

        if len(previous_points) > self.minimum_features:
            estimated, _ = cv2.estimateAffinePartial2D(previous_points, matched_points, method=cv2.RANSAC)
            if estimated is not None:
                affine = estimated

        self.prev_gray = gray
        self.prev_points = points
        return affine


class DeepOCSortTrack:
    def __init__(self, bbox, track_id, delta_t=3):
        try:
            from filterpy.kalman import KalmanFilter
        except Exception as exc:
            raise RuntimeError("DeepOC-SORT backend requires filterpy. Install it into the YOLO env.") from exc

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.track_id = int(track_id)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_observation = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 1.0], dtype=float)
        self.observations = {0: self.last_observation.copy()}
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.delta_t = delta_t

    @staticmethod
    def convert_bbox_to_z(bbox):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center_x = bbox[0] + width / 2.0
        center_y = bbox[1] + height / 2.0
        scale = width * height
        ratio = width / float(height + 1e-6)
        return np.array([center_x, center_y, scale, ratio]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(state):
        width = np.sqrt(max(state[2], 1e-6) * max(state[3], 1e-6))
        height = max(state[2], 1e-6) / max(width, 1e-6)
        return np.array(
            [state[0] - width / 2.0, state[1] - height / 2.0, state[0] + width / 2.0, state[1] + height / 2.0],
            dtype=float,
        )

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x.reshape(-1))

    def update(self, bbox, confidence=1.0):
        if bbox is None:
            self.kf.update(None)
            return

        previous_box = None
        for step in range(self.delta_t, 0, -1):
            key = self.age - step
            if key in self.observations:
                previous_box = self.observations[key][:4]
                break
        if previous_box is None:
            previous_box = self.last_observation[:4]
        self.velocity = speed_direction(previous_box, bbox)

        self.last_observation = np.array([bbox[0], bbox[1], bbox[2], bbox[3], confidence], dtype=float)
        self.observations[self.age] = self.last_observation.copy()
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def apply_affine_correction(self, affine):
        corners = np.array(
            [
                [self.last_observation[0], self.last_observation[1], 1.0],
                [self.last_observation[2], self.last_observation[3], 1.0],
            ],
            dtype=float,
        )
        transformed = (affine @ corners.T).T
        bbox = [transformed[0, 0], transformed[0, 1], transformed[1, 0], transformed[1, 1]]
        self.last_observation[:4] = bbox
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)


class DeepOCSortAdapter:
    """
    Minimal DeepOC-SORT-compatible adapter.

    This borrows the observation-centric association and sparse optical-flow CMC
    ideas from the public DeepOC-SORT repository, but keeps embeddings disabled
    for a lightweight single-video v1 integration.
    """

    def __init__(self, det_thresh=0.25, iou_threshold=0.3, max_age=30, min_hits=1, delta_t=3, inertia=0.2):
        self.det_thresh = float(det_thresh)
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.next_track_id = 1
        self.trackers = defaultdict(list)
        self.cmc = SparseFlowCMC()

    def _new_track(self, class_id, bbox):
        track = DeepOCSortTrack(bbox=bbox, track_id=self.next_track_id, delta_t=self.delta_t)
        self.next_track_id += 1
        self.trackers[int(class_id)].append(track)
        return track.track_id

    def _class_trackers(self, class_id):
        return self.trackers.setdefault(int(class_id), [])

    def update(self, detections, frame_bgr, frame_idx):
        assignments = [None] * len(detections)
        if not detections:
            for class_id, tracks in list(self.trackers.items()):
                live_tracks = []
                for track in tracks:
                    track.predict()
                    track.update(None)
                    if track.time_since_update <= self.max_age:
                        live_tracks.append(track)
                self.trackers[class_id] = live_tracks
            return assignments

        affine = self.cmc.compute_affine(frame_bgr, [det["xyxy"] for det in detections])
        for tracks in self.trackers.values():
            for track in tracks:
                track.apply_affine_correction(affine)

        detections_by_class = defaultdict(list)
        for det_idx, det in enumerate(detections):
            detections_by_class[int(det["class_id"])].append((det_idx, det))

        for class_id, indexed_dets in detections_by_class.items():
            class_tracks = self._class_trackers(class_id)
            det_array = np.array([det["xyxy"] + [float(det["confidence"])] for _, det in indexed_dets], dtype=float)
            predicted_boxes = []
            valid_tracks = []
            for track in class_tracks:
                pred = track.predict()
                if np.any(np.isnan(pred)):
                    continue
                predicted_boxes.append(pred)
                valid_tracks.append(track)
            class_tracks[:] = valid_tracks
            trks = np.array(predicted_boxes, dtype=float) if predicted_boxes else np.empty((0, 4), dtype=float)
            velocities = np.array([track.velocity for track in class_tracks], dtype=float) if class_tracks else np.empty((0, 2))
            previous_obs = (
                np.array([k_previous_obs(track.observations, track.age, self.delta_t) for track in class_tracks], dtype=float)
                if class_tracks
                else np.empty((0, 5), dtype=float)
            )

            matches, unmatched_dets, unmatched_trks = associate_ocsort(
                det_array, trks, self.iou_threshold, velocities, previous_obs, self.inertia
            )

            for det_local_idx, trk_local_idx in matches:
                det_idx, det = indexed_dets[int(det_local_idx)]
                track = class_tracks[int(trk_local_idx)]
                track.update(det["xyxy"], confidence=float(det["confidence"]))
                assignments[det_idx] = track.track_id

            if len(unmatched_dets) and len(unmatched_trks):
                left_dets = det_array[unmatched_dets]
                left_trks = np.array([class_tracks[idx].last_observation[:4] for idx in unmatched_trks], dtype=float)
                left_iou = iou_batch(left_dets[:, :4], left_trks)
                rematched = linear_assignment(-left_iou) if left_iou.size else np.empty((0, 2), dtype=int)
                consumed_dets = set()
                consumed_trks = set()
                for det_local_idx, trk_local_idx in rematched:
                    if left_iou[det_local_idx, trk_local_idx] < self.iou_threshold:
                        continue
                    det_global_idx = int(unmatched_dets[int(det_local_idx)])
                    trk_global_idx = int(unmatched_trks[int(trk_local_idx)])
                    det_idx, det = indexed_dets[det_global_idx]
                    track = class_tracks[trk_global_idx]
                    track.update(det["xyxy"], confidence=float(det["confidence"]))
                    assignments[det_idx] = track.track_id
                    consumed_dets.add(det_global_idx)
                    consumed_trks.add(trk_global_idx)
                unmatched_dets = np.array([idx for idx in unmatched_dets if idx not in consumed_dets], dtype=int)
                unmatched_trks = np.array([idx for idx in unmatched_trks if idx not in consumed_trks], dtype=int)

            for trk_local_idx in unmatched_trks:
                class_tracks[int(trk_local_idx)].update(None)

            for det_local_idx in unmatched_dets:
                det_idx, det = indexed_dets[int(det_local_idx)]
                assignments[det_idx] = self._new_track(class_id, det["xyxy"])

            class_tracks[:] = [track for track in class_tracks if track.time_since_update <= self.max_age]

        return assignments


def tracker_backend_to_yaml(backend):
    if backend == "bytetrack":
        return "bytetrack.yaml"
    if backend == "botsort":
        return "botsort.yaml"
    return None
