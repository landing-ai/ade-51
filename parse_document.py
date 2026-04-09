"""Synchronous document parse operator for the LandingAI ADE FiftyOne plugin."""

from pathlib import Path

import fiftyone.operators as foo
import fiftyone.operators.types as types

try:
    from .utils import (
        add_model_input,
        add_password_input,
        add_region_input,
        check_api_key,
        grounding_to_detections,
        filter_ade_samples,
        get_api_key,
        get_client,
    )
except ImportError:
    from utils import (
        add_model_input,
        add_password_input,
        add_region_input,
        check_api_key,
        grounding_to_detections,
        filter_ade_samples,
        get_api_key,
        get_client,
    )


class ADEParseDocument(foo.Operator):
    """Parse selected documents to structured Markdown with spatial grounding.

    Converts PDFs, images, spreadsheets, and Office files into Markdown stored
    as a dataset field, with bounding box grounding as ``fo.Detections``.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="ade_parse_document",
            label="ADE: Parse Document",
            description=(
                "Convert selected documents to structured Markdown with spatial "
                "grounding using LandingAI ADE."
            ),
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not check_api_key(inputs, ctx):
            return types.Property(inputs, invalid=True)

        inputs.view_target(ctx)
        add_model_input(inputs)
        add_region_input(inputs)
        add_password_input(inputs)

        inputs.str(
            "result_field",
            label="Output field (Markdown)",
            description="Dataset field where parsed Markdown will be stored.",
            default="ade_parse",
            required=True,
        )
        inputs.bool(
            "store_grounding",
            label="Store spatial grounding",
            description=(
                "Store bounding box coordinates for each document element "
                "(text blocks, tables, figures) as fo.Detections."
            ),
            default=True,
        )

        if ctx.params.get("store_grounding", True):
            inputs.str(
                "grounding_field",
                label="Output field (Grounding Detections)",
                description="Dataset field where fo.Detections grounding boxes will be stored.",
                default="ade_grounding",
                required=True,
            )

        return types.Property(inputs)

    def execute(self, ctx):
        api_key = get_api_key(ctx)
        region = ctx.params.get("region", "us")
        model = ctx.params.get("model", "dpt-2-latest")
        password = (ctx.params.get("password") or "").strip()
        result_field = ctx.params.get("result_field", "ade_parse")
        store_grounding = ctx.params.get("store_grounding", True)
        grounding_field = ctx.params.get("grounding_field", "ade_grounding")

        client = get_client(api_key, region)
        samples = filter_ade_samples(ctx.target_view())
        total = len(samples)

        if total == 0:
            return {
                "processed": 0,
                "total": 0,
                "errors": [],
                "message": "No supported documents found among the selected samples.",
            }

        processed = 0
        errors = []

        for i, sample in enumerate(samples):
            ctx.set_progress(progress=i / total, label=f"Parsing {i + 1}/{total}…")
            try:
                parse_kwargs = {"document": Path(sample.filepath), "model": model}
                if password:
                    parse_kwargs["password"] = password

                response = client.parse(**parse_kwargs)
                version = getattr(response.metadata, "version", None)

                sample[result_field] = response.markdown
                sample[f"{result_field}_metadata"] = {
                    "page_count": response.metadata.page_count,
                    "credit_usage": float(response.metadata.credit_usage or 0),
                    "filename": response.metadata.filename,
                    "duration_ms": response.metadata.duration_ms,
                    "version": version,
                    "model_version": version,
                }

                if store_grounding and response.grounding:
                    detections = grounding_to_detections(response.grounding)
                    if detections:
                        sample[grounding_field] = detections

                sample.save()
                processed += 1

            except Exception as e:
                errors.append({"filepath": sample.filepath, "error": str(e)})

        ctx.trigger("reload_dataset")

        return {
            "processed": processed,
            "total": total,
            "errors": errors[:5],
            "error_count": len(errors),
            "result_field": result_field,
            "grounding_field": grounding_field if store_grounding else None,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        result = ctx.results or {}

        processed = result.get("processed", 0)
        total = result.get("total", 0)
        errors = result.get("errors", [])
        error_count = result.get("error_count", len(errors))
        result_field = result.get("result_field", "ade_parse")
        grounding_field = result.get("grounding_field")
        message = result.get("message", "")

        if message:
            outputs.view("notice", types.Notice(label=message))
            return types.Property(outputs)

        summary = f"Processed {processed}/{total} samples. Markdown stored in '{result_field}'."
        if grounding_field:
            summary += f" Grounding stored in '{grounding_field}'."
        if error_count:
            summary += f" {error_count} error(s) — see below."

        outputs.view("summary", types.Notice(label=summary))

        for i, err in enumerate(errors):
            outputs.str(
                f"error_{i}",
                label=f"Error: {err.get('filepath', 'unknown')}",
                default=err.get("error", ""),
            )

        return types.Property(outputs)
