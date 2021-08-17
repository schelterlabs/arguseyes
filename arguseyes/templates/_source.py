from __future__ import annotations
import pandas as pd

import dataclasses
from enum import Enum


class SourceType(Enum):
    ENTITIES = "Entities"
    SIDE_DATA = "SideData"


@dataclasses.dataclass
class Source:
    operator_id: int
    source_type: SourceType
    data: pd.DataFrame
