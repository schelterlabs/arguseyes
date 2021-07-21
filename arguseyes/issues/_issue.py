from __future__ import annotations
import dataclasses
from abc import ABC, abstractmethod


@dataclasses.dataclass
class Issue:
    id: str
    is_present: bool
    details: dict


class IssueDetector(ABC):
    @abstractmethod
    def _detect(self, pipeline) -> Issue:
        raise NotImplementedError

    def error_msg(self, issue) -> str:
        return f'Found {issue.id}'

