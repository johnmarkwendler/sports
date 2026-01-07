import cv2
import numpy as np
import supervision as sv
from typing import Tuple, Optional, List

from sports.waterpolo.config import PoolConfiguration


def _to_pixel(
    point: Tuple[float, float],
    scale: float,
    padding: int,
) -> Tuple[int, int]:
    return (
        int(round(point[0] * scale + padding)),
        int(round(point[1] * scale + padding)),
    )


def draw_pool(
    config: PoolConfiguration,
    scale: float = 5,
    padding: int = 50,
    line_thickness: int = 4,
    line_color: sv.Color = sv.Color.WHITE,
    water_color: sv.Color = sv.Color(30, 144, 255),  # dodger blue
    goal_color: sv.Color = sv.Color.WHITE,
    two_meter_color: sv.Color = sv.Color.from_hex("#C62828"),  # red
    five_meter_color: sv.Color = sv.Color.from_hex("#F9A825"),  # yellow
    center_line_color: sv.Color = sv.Color.WHITE,
) -> np.ndarray:
    """Render a water polo pool with key lines and goals."""
    pool_height_px = int(round(config.field_width * scale))
    pool_length_px = int(round(config.field_length * scale))

    image = np.zeros(
        (pool_height_px + 2 * padding, pool_length_px + 2 * padding, 3),
        dtype=np.uint8,
    )
    image[:, :] = water_color.as_bgr()

    # Outer border
    for start_idx, end_idx in config.edges:
        start_px = _to_pixel(config.vertices[start_idx], scale, padding)
        end_px = _to_pixel(config.vertices[end_idx], scale, padding)
        cv2.line(image, start_px, end_px, line_color.as_bgr(), line_thickness)

    # Goals (top-down: short vertical segments on the left/right borders)
    half_goal = config.goal_width / 2.0
    for x in (0.0, config.field_length):
        cy = config.field_width / 2.0
        start = _to_pixel((x, cy - half_goal), scale, padding)
        end = _to_pixel((x, cy + half_goal), scale, padding)
        cv2.line(image, start, end, goal_color.as_bgr(), max(2, 2 * line_thickness))

    # 2m and 5m lines (draw across the pool width)
    y0 = _to_pixel((0.0, 0.0), scale, padding)[1]
    y1 = _to_pixel((0.0, config.field_width), scale, padding)[1]

    def draw_vertical_line(x_val: float, color: sv.Color) -> None:
        x_px = _to_pixel((x_val, 0.0), scale, padding)[0]
        cv2.line(image, (x_px, y0), (x_px, y1), color.as_bgr(), line_thickness)

    # Left side markers
    draw_vertical_line(config.two_meter_line, two_meter_color)
    draw_vertical_line(config.five_meter_line, five_meter_color)

    # Right side markers
    draw_vertical_line(config.field_length - config.two_meter_line, two_meter_color)
    draw_vertical_line(config.field_length - config.five_meter_line, five_meter_color)

    # Center line
    draw_vertical_line(config.center_line, center_line_color)

    return image


def draw_points_on_pool(
    config: PoolConfiguration,
    xy: Optional[np.ndarray] = None,
    labels: Optional[list[str]] = None,
    fill_color: Optional[sv.Color] = sv.Color.BLACK,
    text_color: sv.Color = sv.Color.WHITE,
    edge_color: Optional[sv.Color] = sv.Color.WHITE,
    size: int = 20,
    edge_thickness: Optional[int] = None,
    scale: float = 5,
    padding: int = 50,
    line_thickness: int = 4,
    pool: Optional[np.ndarray] = None,
) -> np.ndarray:
    if pool is None:
        pool = draw_pool(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=line_thickness,
        )

    if xy is None or np.size(xy) == 0:
        return pool

    pts = np.atleast_2d(xy)
    n = pts.shape[0]

    labels = labels if labels is not None else [None] * n
    if len(labels) < n:
        labels = list(labels) + [None] * (n - len(labels))

    stroke = edge_thickness if edge_thickness is not None else max(2, line_thickness // 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, size / 28.0)
    font_thickness = max(1, size // 8)

    for i in range(n):
        cx, cy = _to_pixel(tuple(pts[i]), scale=scale, padding=padding)

        # Face (fill)
        if fill_color is not None:
            cv2.circle(
                img=pool,
                center=(cx, cy),
                radius=size,
                color=fill_color.as_bgr(),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        # Edge (outline)
        if edge_color is not None and stroke > 0:
            cv2.circle(
                img=pool,
                center=(cx, cy),
                radius=size,
                color=edge_color.as_bgr(),
                thickness=stroke,
                lineType=cv2.LINE_AA,
            )

        # Label
        label = labels[i]
        if label is not None and str(label) != "":
            text = str(label)
            (tw, th), base = cv2.getTextSize(text, font, font_scale, font_thickness)
            tx = int(cx - tw / 2)
            ty = int(cy + th / 2)
            cv2.putText(
                img=pool,
                text=text,
                org=(tx, ty),
                fontFace=font,
                fontScale=font_scale,
                color=text_color.as_bgr(),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

    return pool


def draw_paths_on_pool(
    config: PoolConfiguration,
    paths: List[np.ndarray],
    color: Optional[sv.Color] = sv.Color.BLACK,
    thickness: Optional[int] = None,
    scale: float = 5,
    padding: int = 50,
    line_thickness: int = 4,
    pool: Optional[np.ndarray] = None,
) -> np.ndarray:
    if pool is None:
        pool = draw_pool(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=line_thickness,
        )

    if not paths or color is None:
        return pool

    stroke = thickness if thickness is not None else line_thickness
    bgr = color.as_bgr()

    def to_segments(pts: np.ndarray) -> list[np.ndarray]:
        pts = np.atleast_2d(pts).astype(float)
        segments = []
        cur = []
        for p in pts:
            if np.isnan(p).any():
                if len(cur) > 0:
                    segments.append(np.asarray(cur, dtype=float))
                    cur = []
            else:
                cur.append(p)
        if len(cur) > 0:
            segments.append(np.asarray(cur, dtype=float))
        return segments

    for path in paths:
        if path is None or np.size(path) == 0:
            continue

        for seg in to_segments(path):
            if seg.shape[0] >= 2:
                poly = np.array(
                    [[_to_pixel((float(x), float(y)), scale, padding) for x, y in seg]],
                    dtype=np.int32,
                )
                cv2.polylines(
                    img=pool,
                    pts=poly,
                    isClosed=False,
                    color=bgr,
                    thickness=stroke,
                    lineType=cv2.LINE_AA,
                )
            elif seg.shape[0] == 1:
                cx, cy = _to_pixel((float(seg[0, 0]), float(seg[0, 1])), scale, padding)
                cv2.circle(
                    img=pool,
                    center=(cx, cy),
                    radius=max(1, stroke // 2),
                    color=bgr,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    return pool



