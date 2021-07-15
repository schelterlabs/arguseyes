from __future__ import annotations
import pandas as pd

import dataclasses
from enum import Enum


class SourceType(Enum):
    TRAIN_FACTS = "TrainFacts"
    TRAIN_DIMENSIONS = "TrainDimensions"


@dataclasses.dataclass
class Source:
    operator_id: int
    source_type: SourceType
    data: pd.DataFrame
