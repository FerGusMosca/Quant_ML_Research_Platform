from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class CollectionInfo:
    name: str
    points_count: int
    indexed_vectors_count: int
    status: str


@dataclass
class ScrollResult:
    points: list  # list[ChunkPoint | MetadataPoint]
    next_page_offset: Optional[str]
    total_returned: int


@dataclass
class SearchResult:
    points: list  # list[ChunkPoint | MetadataPoint]
    scores: list[float]