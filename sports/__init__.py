import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"


from sports.common.core import MeasurementUnit
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier

__all__ = [
    "MeasurementUnit",
    "TeamClassifier",
    "ViewTransformer"
]