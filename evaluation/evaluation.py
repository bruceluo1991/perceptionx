import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from common.common import BBox2D, BBox3D, DetectionObject, LabelObject
from evaluation.geometry_utils import (
    polygon_area,
    polygon_intersection_with_convex_polygon,
)


def area_intersection_with_gt(detect: BBox2D, gt: BBox2D) -> float:
    detect_polygon = detect.corners()  # shape: (4, 2)
    gt_polygon = gt.corners()  # shape: (4, 2)
    intersection_polygon = polygon_intersection_with_convex_polygon(
        detect_polygon, gt_polygon
    )
    intersection_area = polygon_area(intersection_polygon)
    intersection_area = np.clip(intersection_area, 0.0)
    return intersection_area


def iou2d(detect: BBox2D, gt: BBox2D) -> float:
    intersection_area = area_intersection_with_gt(detect, gt)
    detect_area = detect.area()
    gt_area = gt.area()
    assert gt_area > 1e-8, "Ground truth box has zero area."
    union_area = gt_area + np.clip(detect_area - intersection_area, 0.0)
    return np.clip(intersection_area / union_area, 0.0, 1.0)


def iou3d(detect: BBox3D, gt: BBox3D) -> float:
    detect_bbox2d = detect.bbox2d()
    gt_bbox2d = gt.bbox2d()
    intersection_area_2d = area_intersection_with_gt(detect_bbox2d, gt_bbox2d)
    height_overlap = np.clip(
        min(detect.z + detect.h / 2, gt.z + gt.h / 2)
        - max(detect.z - detect.h / 2, gt.z - gt.h / 2),
        0.0,
    )
    intersection_volume = intersection_area_2d * height_overlap
    detect_volume = detect.volume()
    gt_volume = gt.volume()
    assert gt_volume > 1e-8, "Ground truth box has zero volume."
    union_volume = gt_volume + np.clip(detect_volume - intersection_volume, 0.0)
    return np.clip(intersection_volume / union_volume, 0.0, 1.0)


PostMatchLabelFilter = Callable[[LabelObject], bool]
UnmatchedDetectionFilter = Callable[[DetectionObject], bool]
SimilarityFunction = Callable[[DetectionObject, LabelObject], float]
MatchThresholdFunction = Callable[[float], bool]

# 如果希望在匹配前过滤掉检测/标签，可以在similarity function和post match filter中体现。


class EvaluationConfig(NamedTuple):
    post_match_label_filter: PostMatchLabelFilter
    unmatched_detection_filter: UnmatchedDetectionFilter
    similarity_function: SimilarityFunction
    match_threshold_function: MatchThresholdFunction


class MatchedPair(NamedTuple):
    detection_idx: int
    label_idx: int
    score: float


class UnmatchedDetection(NamedTuple):
    idx: int
    score: float


@dataclasses.dataclass
class SingleFrameMatchResult:
    matched: list[MatchedPair]
    unmatched_detection: list[UnmatchedDetection]
    unmatched_label: list[int]

    @classmethod
    def create_empty(cls) -> "SingleFrameMatchResult":
        return cls(matched=[], unmatched_detection=[], unmatched_label=[])


def match_detection_and_label(
    detection: list[DetectionObject], label: list[LabelObject], config: EvaluationConfig
) -> SingleFrameMatchResult:
    detection_idx_and_score = [(idx, det.score) for idx, det in enumerate(detection)]
    detection_idx_and_score.sort(key=lambda x: x[1], reverse=True)

    label_matched = [False] * len(label)
    matched_pair = []
    unmatched_detection = []
    for idx, score in detection_idx_and_score:
        highest_similarity = -1.0
        highest_similarity_label_idx = -1
        for label_idx, lbl in enumerate(label):
            if label_matched[label_idx]:
                continue
            similarity = config.similarity_function(detection[idx], lbl)
            if similarity > highest_similarity:
                highest_similarity = similarity
                highest_similarity_label_idx = label_idx

        if config.match_threshold_function(highest_similarity):
            label_matched[highest_similarity_label_idx] = True
            if not config.post_match_label_filter(label[highest_similarity_label_idx]):
                matched_pair.append(
                    MatchedPair(
                        detection_idx=idx,
                        label_idx=highest_similarity_label_idx,
                        score=score,
                    )
                )
        elif not config.unmatched_detection_filter(detection[idx]):
            unmatched_detection.append(UnmatchedDetection(idx=idx, score=score))

    unmatched_label = [
        idx
        for idx, matched in enumerate(label_matched)
        if not matched and not config.post_match_label_filter(label[idx])
    ]
    return SingleFrameMatchResult(
        matched=matched_pair,
        unmatched_detection=unmatched_detection,
        unmatched_label=unmatched_label,
    )


def calculate_ap(
    correctness: NDArray[np.bool_], score: NDArray[np.float64], num_gt: int
) -> float:
    score_sort_idx = np.argsort(score)[::-1]
    score_sorted_correctness = correctness[score_sort_idx]
    sorted_score = score[score_sort_idx]
    tp = np.cumsum(score_sorted_correctness.astype(np.int32))
    precision = tp / (np.arange(len(tp)) + 1)
    recall = tp / num_gt
    unique_score_mask = np.concatenate([sorted_score[:-1] != sorted_score[1:], [True]])
    precision = precision[unique_score_mask]
    recall = recall[unique_score_mask]
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    RECALL_LEVEL = 1024
    ap = 0.0
    for i in range(RECALL_LEVEL):
        this_recall_level = (i + 1) / RECALL_LEVEL
        this_level_idx = np.searchsorted(recall, this_recall_level, side="left")
        if this_level_idx < len(precision):
            ap += precision[this_level_idx].item()
    
    ap /= RECALL_LEVEL
    return ap


def get_ap_from_multi_frame_match(match: list[SingleFrameMatchResult]) -> float:
    num_gt = 0
    num_detection = 0
    for m in match:
        num_gt += (len(m.matched) + len(m.unmatched_label))
        num_detection += (len(m.matched) + len(m.unmatched_detection))

    scores = np.zeros((num_detection,), dtype=np.float64)
    correctness = np.zeros((num_detection,), dtype=bool)
    idx = 0
    for m in match:
        for pair in m.matched:
            scores[idx] = pair.score
            correctness[idx] = True
            idx += 1
        for und in m.unmatched_detection:
            scores[idx] = und.score
            correctness[idx] = False
            idx += 1
    
    assert idx == num_detection
    ap = calculate_ap(correctness, scores, num_gt)
    return ap
