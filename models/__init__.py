"""Interchangeable SSL, representation, and decoder components."""

from models.interfaces import CTCDecoder, DecoderOutput, RepresentationAdapter
from models.interfaces import SSLExtractor, SpeechFeatures

__all__ = [
    "CTCDecoder",
    "DecoderOutput",
    "RepresentationAdapter",
    "SSLExtractor",
    "SpeechFeatures",
]
