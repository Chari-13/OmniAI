"""
Live API Simulator — Simulated real-time review stream.

Provides:
  - Async generator for streaming reviews at configurable rates
  - Draws from a pool of sample reviews with randomized timing
  - WebSocket-compatible for real-time dashboard updates
"""

from __future__ import annotations
import asyncio
import random
from datetime import datetime
from typing import AsyncGenerator, Optional

from models.schemas import ReviewInput


# Sample review pool for simulation
_SAMPLE_POOL = [
    ("The battery life is absolutely fantastic, lasts me 2 full days!", "electronics"),
    ("Worst product ever. Battery dies in 2 hours, total waste.", "electronics"),
    ("Great taste and reasonable price. Will buy again!", "food"),
    ("Oh wow, the packaging was SO premium... if you consider crushed cardboard premium", "food"),
    ("Sure, the customer support is 'amazing' -- only took 3 weeks to reply", "electronics"),
    ("Packaging is beautiful and product arrived in perfect condition.", "food"),
    ("The service was prompt and the staff was very helpful.", "services"),
    ("Terrible experience. Staff was rude and unhelpful.", "services"),
    ("Price is way too high for what you get. Not worth it.", "electronics"),
    ("Absolutely love this product! Best purchase I've made.", "electronics"),
    ("The taste was bland and packaging was damaged on arrival.", "food"),
    ("Customer support resolved my issue within minutes. Impressed!", "electronics"),
    ("Battery drains super fast. Very disappointed with the quality.", "electronics"),
    ("Fresh ingredients and great flavor. Highly recommend!", "food"),
    ("The price point is perfect. Great value for money.", "food"),
]


class LiveReviewSimulator:
    """
    Simulates a live review API stream for real-time processing demos.

    Usage:
        simulator = LiveReviewSimulator(rate=2.0)
        async for review in simulator.stream(count=50):
            result = await pipeline.process_single(review)
    """

    def __init__(self, rate: float = 1.0, pool: Optional[list[tuple[str, str]]] = None):
        """
        Args:
            rate: Reviews per second (approximate)
            pool: Custom review pool, or uses built-in samples
        """
        self._rate = rate
        self._pool = pool or _SAMPLE_POOL
        self._streamed_count = 0
        self._is_streaming = False

    async def stream(
        self,
        count: int = 100,
        randomize_delay: bool = True,
    ) -> AsyncGenerator[ReviewInput, None]:
        """
        Async generator that yields ReviewInput objects at the configured rate.

        Args:
            count: Total number of reviews to stream
            randomize_delay: Add jitter to delay (more realistic)

        Yields:
            ReviewInput objects
        """
        self._is_streaming = True

        for i in range(count):
            if not self._is_streaming:
                break

            text, category = random.choice(self._pool)

            yield ReviewInput(
                text=text,
                category=category,
                source="live_api",
                timestamp=datetime.now(),
            )

            self._streamed_count += 1

            # Delay between reviews
            base_delay = 1.0 / self._rate
            if randomize_delay:
                delay = base_delay * random.uniform(0.5, 1.5)
            else:
                delay = base_delay

            await asyncio.sleep(delay)

        self._is_streaming = False

    def stop(self):
        """Stop the stream."""
        self._is_streaming = False

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def streamed_count(self) -> int:
        return self._streamed_count


# Singleton
live_simulator = LiveReviewSimulator(rate=2.0)
