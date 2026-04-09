"""Shared utilities for the Landing AI ADE FiftyOne plugin."""

import os
from typing import Optional

import fiftyone.core.labels as fol
import fiftyone.operators.types as types

try:
    from landingai_ade import LandingAIADE as _LandingAIADE
except ImportError:
    _LandingAIADE = None


ADE_SUPPORTED_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc", ".odt",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
    ".gif", ".apng", ".dcx", ".dds", ".dib", ".gd", ".icns",
    ".jp2", ".pcx", ".ppm", ".psd", ".tga",
    ".xlsx", ".csv",
    ".ppt", ".pptx",
})


def get_api_key(ctx) -> str:
    """Resolve the Landing AI API key from FiftyOne secrets or environment.

    Priority: VISION_AGENT_API_KEY (secret → env) then LANDING_AI_API_KEY (secret → env).
    """
    api_key = (
        ctx.secrets.get("VISION_AGENT_API_KEY")
        or os.getenv("VISION_AGENT_API_KEY")
        or ctx.secrets.get("LANDING_AI_API_KEY")
        or os.getenv("LANDING_AI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "No Landing AI API key found. "
            "Set the VISION_AGENT_API_KEY environment variable or add it to FiftyOne secrets."
        )
    return api_key


def get_client(api_key: str, region: str = "us"):
    """Return an authenticated ``LandingAIADE`` client for the given region.

    Args:
        api_key: Landing AI / Vision Agent API key.
        region: ``"us"`` (default) or ``"eu"``.
    """
    if _LandingAIADE is None:
        raise ImportError(
            "The 'landingai-ade' package is required. "
            "Install it with:\n\n"
            "    pip install landingai-ade\n\n"
            "or run:\n\n"
            "    fiftyone plugins requirements @landingai/ade --install"
        )
    environment = "eu" if region == "eu" else "production"
    return _LandingAIADE(apikey=api_key, environment=environment)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def ade_box_to_fo(box) -> list:
    """Convert an ADE bounding box to FiftyOne ``[x, y, w, h]`` format.

    ADE uses ``[left, top, right, bottom]`` normalized 0–1.
    FiftyOne uses ``[x, y, width, height]`` from the top-left, normalized 0–1.
    """
    return [
        _clamp(box.left),
        _clamp(box.top),
        _clamp(box.right - box.left),
        _clamp(box.bottom - box.top),
    ]


def grounding_to_detections(grounding: dict) -> Optional[fol.Detections]:
    """Convert a ``parse_resp.grounding`` dict to ``fo.Detections``.

    Covers all element types — top-level chunks (text, table, figure) and
    individual table cells — each stored with ``chunk_id`` and ``page`` so
    bbox lookup works for every extracted field, including numeric ones.
    """
    detections = []
    for element_id, g in (grounding or {}).items():
        if not g or not g.box:
            continue
        detections.append(
            fol.Detection(
                label=g.type or "unknown",
                bounding_box=ade_box_to_fo(g.box),
                chunk_id=element_id,
                page=int(g.page) if g.page is not None else 0,
            )
        )
    return fol.Detections(detections=detections) if detections else None


def to_plain_data(value):
    """Convert SDK response objects into plain Python containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_plain_data(v) for v in value]

    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump())

    if hasattr(value, "dict"):
        return to_plain_data(value.dict())

    if hasattr(value, "__dict__"):
        return {
            k: to_plain_data(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }

    return str(value)


def filter_ade_samples(view) -> list:
    """Return samples whose filepath has an ADE-supported extension."""
    return [
        s for s in view
        if os.path.splitext(s.filepath)[1].lower() in ADE_SUPPORTED_EXTENSIONS
    ]


def check_api_key(inputs, ctx) -> bool:
    """Show an error notice and return ``False`` if no API key is configured.

    Call at the top of ``resolve_input``. If it returns ``False``, return
    ``types.Property(inputs, invalid=True)`` immediately.
    """
    key = (
        ctx.secrets.get("VISION_AGENT_API_KEY")
        or os.getenv("VISION_AGENT_API_KEY")
        or ctx.secrets.get("LANDING_AI_API_KEY")
        or os.getenv("LANDING_AI_API_KEY")
    )
    if not key:
        inputs.view(
            "no_api_key_error",
            types.Notice(
                label=(
                    "Landing AI API key not found. "
                    "Set the VISION_AGENT_API_KEY environment variable and restart FiftyOne, "
                    "or add it to your FiftyOne secrets config."
                )
            ),
        )
        return False
    return True


def add_model_input(inputs):
    """Add the model selector (dpt-2 vs dpt-2-mini)."""
    model_choices = types.RadioGroup()
    model_choices.add_choice("dpt-2", label="dpt-2  (Full featured — 3 credits/page)")
    model_choices.add_choice("dpt-2-mini", label="dpt-2-mini  (Simple docs, preview — 1.5 credits/page)")
    inputs.enum(
        "model",
        values=model_choices.values(),
        label="Model",
        default="dpt-2",
        required=True,
        view=model_choices,
    )


def add_extract_model_input(inputs):
    """Add the extraction model selector."""
    extract_choices = types.Dropdown()
    extract_choices.add_choice("extract-latest", label="extract-latest  (Latest extraction model)")
    extract_choices.add_choice("extract-20260314", label="extract-20260314  (March 2026)")
    inputs.enum(
        "extract_model",
        values=extract_choices.values(),
        label="Extraction model",
        default="extract-latest",
        required=True,
        view=extract_choices,
    )


def add_split_model_input(inputs):
    """Add the split model selector."""
    split_choices = types.Dropdown()
    split_choices.add_choice("split-latest", label="split-latest  (Latest split model)")
    split_choices.add_choice("split-20251105", label="split-20251105  (Pinned snapshot)")
    inputs.enum(
        "split_model",
        values=split_choices.values(),
        label="Split model",
        default="split-latest",
        required=True,
        view=split_choices,
    )


def add_password_input(inputs):
    """Add an optional password input for encrypted documents."""
    inputs.str(
        "password",
        label="Document password",
        description=(
            "Password for password-protected files. Requires a ZDR-enabled account. "
            "Ignored for unencrypted documents."
        ),
    )


def add_region_input(inputs):
    """Add the region selector (US vs EU endpoint)."""
    region_choices = types.Dropdown()
    region_choices.add_choice("us", label="US  (api.va.landing.ai)")
    region_choices.add_choice("eu", label="EU  (api.va.eu-west-1.landing.ai)")
    inputs.enum(
        "region",
        values=region_choices.values(),
        label="Region",
        default="us",
        view=region_choices,
    )
