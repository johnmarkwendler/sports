from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
from enum import Enum
from typing import Dict, List, Tuple

from sports.common.core import MeasurementUnit


# Dimensions reference (FINA guidelines, rounded):
# - Field of play: Men 30m x 20m, Women 25m x 20m
# - Goal: 3.0m width x 0.9m height (rendering uses width only in top-down)
# - Key lines from goal line: 2m, 5m; center line at 1/2 length
CENTIMETERS_PER_FOOT = Decimal("30.48")


class League(Enum):
    MEN = "men"
    WOMEN = "women"


# presets stored in centimeters
PRESETS_CENTIMETERS: Dict[League, Dict[str, int]] = {
    League.MEN: dict(
        field_width=2000,     # 20.0 m
        field_length=3000,    # 30.0 m
        goal_width=300,       # 3.0 m
        goal_height=90,       # 0.9 m (not used in top-down, kept for completeness)
        two_meter=200,        # 2.0 m
        five_meter=500,       # 5.0 m
    ),
    League.WOMEN: dict(
        field_width=2000,     # 20.0 m
        field_length=2500,    # 25.0 m
        goal_width=300,       # 3.0 m
        goal_height=90,       # 0.9 m
        two_meter=200,        # 2.0 m
        five_meter=500,       # 5.0 m
    ),
}


@dataclass
class PoolConfiguration:
    """Configure water polo pool geometry for MEN or WOMEN variants.
    Exposes field/pool dimensions in centimeters or feet with consistent rounding.
    """
    league: League
    measurement_unit: MeasurementUnit = MeasurementUnit.CENTIMETERS

    # internal values in centimeters
    _field_width_cm: int = field(init=False)
    _field_length_cm: int = field(init=False)
    _goal_width_cm: int = field(init=False)
    _goal_height_cm: int = field(init=False)
    _two_meter_cm: int = field(init=False)
    _five_meter_cm: int = field(init=False)

    def __post_init__(self) -> None:
        preset = PRESETS_CENTIMETERS[self.league]
        self._field_width_cm = preset["field_width"]
        self._field_length_cm = preset["field_length"]
        self._goal_width_cm = preset["goal_width"]
        self._goal_height_cm = preset["goal_height"]
        self._two_meter_cm = preset["two_meter"]
        self._five_meter_cm = preset["five_meter"]

    def _to_output_unit_rounded_up(self, value_in_centimeters: float) -> float:
        value = Decimal(str(value_in_centimeters))
        if self.measurement_unit == MeasurementUnit.FEET:
            value = value / CENTIMETERS_PER_FOOT
        return float(value.quantize(Decimal("0.01"), rounding=ROUND_CEILING))

    # public properties in the selected unit
    @property
    def field_width(self) -> float:
        return self._to_output_unit_rounded_up(self._field_width_cm)

    @property
    def field_length(self) -> float:
        return self._to_output_unit_rounded_up(self._field_length_cm)

    @property
    def goal_width(self) -> float:
        return self._to_output_unit_rounded_up(self._goal_width_cm)

    @property
    def goal_height(self) -> float:
        return self._to_output_unit_rounded_up(self._goal_height_cm)

    @property
    def two_meter_line(self) -> float:
        return self._to_output_unit_rounded_up(self._two_meter_cm)

    @property
    def five_meter_line(self) -> float:
        return self._to_output_unit_rounded_up(self._five_meter_cm)

    @property
    def center_line(self) -> float:
        return self._to_output_unit_rounded_up(self._field_length_cm / 2.0)

    # vertices and edges (in configured unit) for outer border convenience
    def _vertices_centimeters(self) -> List[Tuple[int, int]]:
        w = self._field_width_cm
        l = self._field_length_cm
        return [
            (0, 0),          # bottom-left
            (0, w),          # top-left
            (l, w),          # top-right
            (l, 0),          # bottom-right
        ]

    def _vertices_in_unit(self) -> List[Tuple[float, float]]:
        return [
            (
                self._to_output_unit_rounded_up(x),
                self._to_output_unit_rounded_up(y),
            )
            for x, y in self._vertices_centimeters()
        ]

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return self._vertices_in_unit()

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 1), (1, 2), (2, 3), (3, 0)
    ])

    # helpful accessors for goal areas in configured unit (top-down)
    @property
    def left_goal_center(self) -> Tuple[float, float]:
        return (0.0, self.field_width / 2.0)

    @property
    def right_goal_center(self) -> Tuple[float, float]:
        return (self.field_length, self.field_width / 2.0)



