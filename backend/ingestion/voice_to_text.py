"""
Voice-to-Text Module — OpenAI Whisper integration for audio review ingestion.

Gracefully degrades if Whisper is not installed.
Supports: .wav, .mp3, .m4a audio files.
"""

from __future__ import annotations
import os
import tempfile
from typing import Optional
from models.schemas import ReviewInput


# Try to import Whisper (optional dependency)
_whisper_available = False
_whisper_model = None

try:
    import whisper
    _whisper_available = True
except ImportError:
    pass


class VoiceToText:
    """
    Audio transcription using OpenAI Whisper.

    Falls back gracefully if Whisper is not installed.
    Loads the model lazily on first use to avoid startup overhead.
    """

    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self._model_size = model_size
        self._model = None
        self._transcription_count = 0

    @property
    def is_available(self) -> bool:
        return _whisper_available

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None and _whisper_available:
            import whisper
            self._model = whisper.load_model(self._model_size)
            print(f"  [OK] Whisper model loaded: {self._model_size}")

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        category: str = "general",
    ) -> Optional[ReviewInput]:
        """
        Transcribe audio to text and return as ReviewInput.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (for format detection)
            category: Product category to assign

        Returns:
            ReviewInput with transcribed text, or None if failed
        """
        if not _whisper_available:
            print("  [WARN] Whisper not installed. Install with: pip install openai-whisper")
            return None

        # Validate format
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            print(f"  [WARN] Unsupported audio format: {ext}")
            return None

        try:
            self._load_model()

            # Write to temp file (Whisper needs a file path)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Transcribe
            result = self._model.transcribe(tmp_path)
            text = result.get("text", "").strip()

            # Cleanup
            os.unlink(tmp_path)

            if not text:
                return None

            self._transcription_count += 1

            detected_lang = result.get("language", "en")

            return ReviewInput(
                text=text,
                category=category,
                source="voice",
            )

        except Exception as e:
            print(f"  [WARN] Whisper transcription failed: {e}")
            return None

    @property
    def transcription_count(self) -> int:
        return self._transcription_count


# Singleton
voice_to_text = VoiceToText(model_size="base")
