import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"


from sports.common.core import MeasurementUnit
from sports.common.view import ViewTransformer
from sports.common.temporal import ConsecutiveValueTracker
from sports.common.path import clean_paths

# Optional: TeamClassifier depends on torch; avoid failing import for users
# who don't need it (e.g., water polo rendering).
try:
    from sports.common.team import TeamClassifier, TeamClassifierByColor  # type: ignore
except Exception:  # torch or other optional deps may be missing
    TeamClassifier = None  # type: ignore
    TeamClassifierByColor = None  # type: ignore

__all__ = [
    "clean_paths",
    "ConsecutiveValueTracker",
    "MeasurementUnit",
    "ViewTransformer"
]

if TeamClassifier is not None:
    __all__.append("TeamClassifier")
if TeamClassifierByColor is not None:
    __all__.append("TeamClassifierByColor")