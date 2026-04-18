"""
CSV/JSON Loader — Multi-format file ingestion for review data.

Supports:
  - CSV files with flexible column mapping
  - JSON files (array of objects or single object)
  - Auto-detection of text/review/comment/feedback columns
  - Category, timestamp, and rating column extraction
"""

from __future__ import annotations
import csv
import json
import io
from datetime import datetime
from typing import Optional

from models.schemas import ReviewInput


# Common column name mappings
_TEXT_COLUMNS = {"text", "review", "review_text", "comment", "feedback", "content", "body", "message"}
_CATEGORY_COLUMNS = {"category", "cat", "product_category", "type", "product_type"}
_TIMESTAMP_COLUMNS = {"timestamp", "date", "created_at", "review_date", "time", "posted_at"}
_RATING_COLUMNS = {"rating", "stars", "score", "star_rating"}


class CSVJSONLoader:
    """
    Flexible multi-format review data loader.

    Auto-detects column names and maps them to ReviewInput fields.
    Handles messy real-world data with graceful fallbacks.
    """

    def __init__(self):
        self._loaded_count = 0

    def load_csv(self, content: str | bytes, source: str = "csv") -> list[ReviewInput]:
        """
        Parse CSV content into ReviewInput objects.

        Args:
            content: CSV string or bytes content
            source: Source identifier for metadata

        Returns:
            List of ReviewInput objects
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames:
            return []

        # Map columns
        col_map = self._detect_columns(reader.fieldnames)
        reviews = []

        for row in reader:
            text = self._get_field(row, col_map.get("text"))
            if not text or not text.strip():
                continue

            category = self._get_field(row, col_map.get("category")) or "general"
            timestamp = self._parse_timestamp(self._get_field(row, col_map.get("timestamp")))
            rating = self._parse_rating(self._get_field(row, col_map.get("rating")))

            reviews.append(ReviewInput(
                text=text.strip(),
                category=category.strip().lower(),
                source=source,
                timestamp=timestamp,
                rating=rating,
            ))

        self._loaded_count += len(reviews)
        return reviews

    def load_json(self, content: str | bytes, source: str = "json") -> list[ReviewInput]:
        """
        Parse JSON content into ReviewInput objects.

        Handles:
          - Array of review objects: [{"text": "...", ...}, ...]
          - Object with reviews key: {"reviews": [...]}
          - Single review object: {"text": "..."}
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []

        # Normalize to list
        if isinstance(data, dict):
            if "reviews" in data:
                items = data["reviews"]
            elif "data" in data:
                items = data["data"]
            else:
                items = [data]
        elif isinstance(data, list):
            items = data
        else:
            return []

        reviews = []
        for item in items:
            if not isinstance(item, dict):
                continue

            # Try common text field names
            text = None
            for key in _TEXT_COLUMNS:
                if key in item:
                    text = str(item[key])
                    break

            if not text or not text.strip():
                continue

            category = "general"
            for key in _CATEGORY_COLUMNS:
                if key in item:
                    category = str(item[key])
                    break

            timestamp = None
            for key in _TIMESTAMP_COLUMNS:
                if key in item:
                    timestamp = self._parse_timestamp(str(item[key]))
                    break

            rating = None
            for key in _RATING_COLUMNS:
                if key in item:
                    rating = self._parse_rating(str(item[key]))
                    break

            reviews.append(ReviewInput(
                text=text.strip(),
                category=category.strip().lower(),
                source=source,
                timestamp=timestamp,
                rating=rating,
                reviewer_id=item.get("reviewer_id") or item.get("user_id"),
            ))

        self._loaded_count += len(reviews)
        return reviews

    def _detect_columns(self, fieldnames: list[str]) -> dict[str, str]:
        """Auto-detect column mappings from CSV headers."""
        col_map = {}
        lower_fields = {f.lower().strip(): f for f in fieldnames}

        for target, candidates in [
            ("text", _TEXT_COLUMNS),
            ("category", _CATEGORY_COLUMNS),
            ("timestamp", _TIMESTAMP_COLUMNS),
            ("rating", _RATING_COLUMNS),
        ]:
            for candidate in candidates:
                if candidate in lower_fields:
                    col_map[target] = lower_fields[candidate]
                    break

        # Fallback: use first column as text if no text column found
        if "text" not in col_map and fieldnames:
            col_map["text"] = fieldnames[0]

        return col_map

    @staticmethod
    def _get_field(row: dict, field: Optional[str]) -> Optional[str]:
        if field and field in row:
            val = row[field]
            return str(val) if val else None
        return None

    @staticmethod
    def _parse_timestamp(val: Optional[str]) -> Optional[datetime]:
        if not val:
            return None
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(val.strip(), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_rating(val: Optional[str]) -> Optional[float]:
        if not val:
            return None
        try:
            r = float(val.strip())
            return r if 1 <= r <= 5 else None
        except (ValueError, TypeError):
            return None

    @property
    def loaded_count(self) -> int:
        return self._loaded_count


# Singleton
csv_json_loader = CSVJSONLoader()
