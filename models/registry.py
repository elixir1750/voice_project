from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

from models.interfaces import CTCDecoder, RepresentationAdapter, SSLExtractor


SSL_REGISTRY: dict[str, type[SSLExtractor]] = {}
REPRESENTATION_REGISTRY: dict[str, type[RepresentationAdapter]] = {}
DECODER_REGISTRY: dict[str, type[CTCDecoder]] = {}

ComponentType = TypeVar("ComponentType")


def _register(
    registry: dict[str, type[ComponentType]],
    name: str,
) -> Callable[[type[ComponentType]], type[ComponentType]]:
    normalized = name.lower()

    def decorator(component: type[ComponentType]) -> type[ComponentType]:
        if normalized in registry:
            raise ValueError(f"Component is already registered: {normalized}")
        registry[normalized] = component
        return component

    return decorator


def register_ssl(name: str):
    return _register(SSL_REGISTRY, name)


def register_representation(name: str):
    return _register(REPRESENTATION_REGISTRY, name)


def register_decoder(name: str):
    return _register(DECODER_REGISTRY, name)


def _ensure_default_components() -> None:
    import models.ctc_decoder  # noqa: F401
    import models.representations  # noqa: F401
    import models.ssl_extractor  # noqa: F401


def _component_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if "type" not in config:
        raise ValueError("Component configuration requires a type")
    component_type = str(config["type"]).lower()
    arguments = dict(config)
    del arguments["type"]
    return component_type, arguments


def build_ssl(config: Mapping[str, Any]) -> SSLExtractor:
    _ensure_default_components()
    component_type, arguments = _component_config(config)
    try:
        component = SSL_REGISTRY[component_type]
    except KeyError as error:
        raise ValueError(f"Unknown SSL extractor: {component_type}") from error
    return component(**arguments)


def build_representation(
    config: Mapping[str, Any],
    input_dim: int,
) -> RepresentationAdapter:
    _ensure_default_components()
    component_type, arguments = _component_config(config)
    try:
        component = REPRESENTATION_REGISTRY[component_type]
    except KeyError as error:
        raise ValueError(f"Unknown representation: {component_type}") from error
    return component(input_dim=input_dim, **arguments)


def build_decoder(
    config: Mapping[str, Any],
    input_dim: int,
    vocab_size: int,
) -> CTCDecoder:
    _ensure_default_components()
    component_type, arguments = _component_config(config)
    try:
        component = DECODER_REGISTRY[component_type]
    except KeyError as error:
        raise ValueError(f"Unknown decoder: {component_type}") from error
    return component(input_dim=input_dim, vocab_size=vocab_size, **arguments)
