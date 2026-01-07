import numpy as np
import cv2
import supervision as sv

from sports.waterpolo import PoolConfiguration, League, draw_pool, draw_points_on_pool, draw_paths_on_pool
from sports.common.core import MeasurementUnit


def main() -> None:
    config = PoolConfiguration(
        league=League.MEN,
        measurement_unit=MeasurementUnit.CENTIMETERS,
    )

    img = draw_pool(
        config=config,
        scale=5,
        padding=50,
        line_thickness=4,
        water_color=sv.Color(30, 144, 255),
    )

    # Demo: two players and a ball
    pts = np.array([
        [300.0, 500.0],   # player 1
        [2700.0, 1500.0], # player 2
        [1500.0, 1000.0], # ball
    ], dtype=float)
    labels = ["P1", "P2", "Ball"]
    img = draw_points_on_pool(
        config=config,
        xy=pts,
        labels=labels,
        fill_color=sv.Color.from_hex("#0D47A1"),
        edge_color=sv.Color.WHITE,
        text_color=sv.Color.WHITE,
        size=18,
        scale=5,
        padding=50,
        line_thickness=4,
        pool=img,
    )

    # Demo: a simple path with a gap
    path = np.array([
        [300.0, 500.0],
        [600.0, 700.0],
        [900.0, 900.0],
        [np.nan, np.nan],
        [1200.0, 1100.0],
        [1500.0, 1200.0],
    ], dtype=float)
    img = draw_paths_on_pool(
        config=config,
        paths=[path],
        color=sv.Color.BLACK,
        thickness=4,
        scale=5,
        padding=50,
        line_thickness=4,
        pool=img,
    )

    cv2.imwrite("waterpolo_pool_demo.png", img)


if __name__ == "__main__":
    main()



