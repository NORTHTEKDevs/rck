"""Multimodal interface stubs -- image / audio / video.

LLMs handle multimodal via integrated vision/audio encoders trained
jointly. RCK takes the modular approach: external models (Stable
Diffusion for images, Whisper for audio, etc.) plug in via provider
interfaces.

This module defines the PROTOCOL. External users implement the
providers. At runtime, the agent dispatches to whichever provider is
configured.

By default, the v6 stubs return informative "model not installed"
messages so the architecture is testable. v8 will ship real providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
#  Image generation
# ---------------------------------------------------------------------------

class ImageGenerator(Protocol):
    """Provider interface for image generation."""
    def generate(self, prompt: str, *, width: int = 1024, height: int = 1024,
                 **kwargs) -> dict:
        ...


@dataclass
class StubImageGenerator:
    """Default stub: returns a structured description of what WOULD be generated."""
    model_name: str = "stub-imagegen"

    def generate(self, prompt: str, *, width: int = 1024, height: int = 1024,
                 **kwargs) -> dict:
        return {
            "ok": False,
            "provider": self.model_name,
            "prompt": prompt,
            "size": [width, height],
            "message": (f"[stub] Would generate a {width}x{height} image of "
                        f"'{prompt}'. Install a real provider (e.g. via "
                        f"diffusers/stability-sdk) and register it."),
        }


# ---------------------------------------------------------------------------
#  Image understanding
# ---------------------------------------------------------------------------

class ImageUnderstander(Protocol):
    """Provider for image -> structured description."""
    def describe(self, image_path: str) -> dict:
        ...


@dataclass
class StubImageUnderstander:
    model_name: str = "stub-imagedescriber"

    def describe(self, image_path: str) -> dict:
        return {
            "ok": False,
            "provider": self.model_name,
            "path": image_path,
            "message": ("[stub] Would describe the image. Install a vision "
                        "model (e.g. CLIP/BLIP) and register it."),
        }


# ---------------------------------------------------------------------------
#  Audio
# ---------------------------------------------------------------------------

class AudioTranscriber(Protocol):
    def transcribe(self, audio_path: str) -> dict:
        ...


@dataclass
class StubAudioTranscriber:
    model_name: str = "stub-asr"

    def transcribe(self, audio_path: str) -> dict:
        return {
            "ok": False,
            "provider": self.model_name,
            "path": audio_path,
            "message": ("[stub] Would transcribe audio. Install Whisper "
                        "or similar and register it."),
        }


class TextToSpeech(Protocol):
    def speak(self, text: str, *, voice: str = "default") -> dict:
        ...


@dataclass
class StubTextToSpeech:
    model_name: str = "stub-tts"

    def speak(self, text: str, *, voice: str = "default") -> dict:
        return {
            "ok": False,
            "provider": self.model_name,
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "voice": voice,
            "message": ("[stub] Would synthesize speech. Install a TTS "
                        "model (e.g. Coqui, Bark) and register it."),
        }


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

@dataclass
class MultimodalRegistry:
    """Holds the active providers for each modality."""

    image_gen: ImageGenerator = field(default_factory=StubImageGenerator)
    image_understand: ImageUnderstander = field(default_factory=StubImageUnderstander)
    audio_transcribe: AudioTranscriber = field(default_factory=StubAudioTranscriber)
    tts: TextToSpeech = field(default_factory=StubTextToSpeech)

    def set_image_generator(self, provider: ImageGenerator) -> None:
        self.image_gen = provider

    def set_image_understander(self, provider: ImageUnderstander) -> None:
        self.image_understand = provider

    def set_audio_transcriber(self, provider: AudioTranscriber) -> None:
        self.audio_transcribe = provider

    def set_tts(self, provider: TextToSpeech) -> None:
        self.tts = provider

    def providers(self) -> dict[str, str]:
        return {
            "image_gen": getattr(self.image_gen, "model_name", "?"),
            "image_understand": getattr(self.image_understand, "model_name", "?"),
            "audio_transcribe": getattr(self.audio_transcribe, "model_name", "?"),
            "tts": getattr(self.tts, "model_name", "?"),
        }
