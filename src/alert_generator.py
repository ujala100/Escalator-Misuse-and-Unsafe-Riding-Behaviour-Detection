"""
Alert Generator Module
Generates timestamped alerts and structured logs for detected safety violations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging


class AlertGenerator:
    """
    Constructs, filters, formats, and persists alerts for unsafe escalator events.
    """

    def __init__(
        self,
        output_file: Optional[Union[str, Path]] = None,
        cooldown_seconds: float = 5.0,
    ) -> None:
        """
        Initialize the AlertGenerator.

        Args:
            output_file: Path to write structured alert records (JSON/JSONL).
            cooldown_seconds: Minimum delay before repeating identical alerts for a track.
        """
        self.output_file = Path(output_file) if output_file else None
        self.cooldown_seconds = cooldown_seconds
        self.alert_history: List[Dict[str, Any]] = []
        self._last_alert_times: Dict[str, float] = {}
        self.logger = logging.getLogger("EscalatorAlerts")

    def create_alert(
        self,
        event_type: str,
        person_id: int,
        severity: str,
        timestamp: Optional[str] = None,
        frame_idx: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured alert dictionary.

        Args:
            event_type: Category of violation (e.g. 'reverse_walking', 'handrail_climbing').
            person_id: Track ID of the involved individual.
            severity: Alert severity level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
            timestamp: ISO formatted timestamp (defaults to current UTC time).
            frame_idx: Video frame index where violation was recorded.
            details: Additional context metrics (speed, position, bounding box).

        Returns:
            Dict containing the structured alert record.
        """
        ts = timestamp or datetime.utcnow().isoformat() + "Z"
        alert_record = {
            "alert_id": f"ALT-{int(datetime.utcnow().timestamp() * 1000)}-{person_id}",
            "timestamp": ts,
            "frame_idx": frame_idx,
            "person_id": person_id,
            "event_type": event_type,
            "severity": severity,
            "details": details or {},
        }
        return alert_record

    def dispatch(self, alert_record: Dict[str, Any]) -> None:
        """
        Persists and emits the alert record.

        Args:
            alert_record: Structured alert dictionary to persist.
        """
        self.alert_history.append(alert_record)
        self.logger.warning(
            "ALERT [%s] - Person %d - %s at %s",
            alert_record["severity"],
            alert_record["person_id"],
            alert_record["event_type"],
            alert_record["timestamp"],
        )

        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_record) + "\n")
