"""
Processing Result Models
Typed result objects for every engine operation.
Ensures consistent output format consumed by the agent tool wrappers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ProcessingResult:
    """Base result for all image operations."""
    output_path: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def with_meta(self, **kwargs) -> "ProcessingResult":
        self.metadata.update(kwargs)
        return self