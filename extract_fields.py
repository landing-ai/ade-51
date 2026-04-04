"""Schema-based field extraction operator for the Landing AI ADE FiftyOne plugin."""

import json
from pathlib import Path

import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types

try:
    from .utils import (
        add_model_input,
        add_region_input,
        ade_box_to_fo,
        check_api_key,
        grounding_to_detections,
        filter_ade_samples,
        get_api_key,
        get_client,
    )
except ImportError:
    from utils import (
        add_model_input,
        add_region_input,
        ade_box_to_fo,
        check_api_key,
        grounding_to_detections,
        filter_ade_samples,
        get_api_key,
        get_client,
    )


_DEFAULT_FIELDS = [
    {"name": "invoice_number", "description": "The unique invoice identifier",        "type": "string"},
    {"name": "vendor_name",    "description": "Name of the vendor or supplier",       "type": "string"},
    {"name": "total_amount",   "description": "Total amount due including taxes",     "type": "number"},
    {"name": "invoice_date",   "description": "Date the invoice was issued",          "type": "string"},
]

_SUPPORTED_TYPES = ["string", "number", "boolean"]

_FO_TYPE_MAP = {
    "string": fof.StringField,
    "number": fof.FloatField,
    "boolean": fof.BooleanField,
}


def _build_field_type_dropdown():
    d = types.Dropdown()
    d.add_choice("string",  label="Text (string)")
    d.add_choice("number",  label="Number (float)")
    d.add_choice("boolean", label="True / False (boolean)")
    return d


def _get_detections_fields(ctx) -> list:
    """Return names of all fo.Detections fields on the dataset."""
    result = []
    try:
        for name, field in ctx.dataset.get_field_schema().items():
            if isinstance(field, fof.EmbeddedDocumentField):
                if issubclass(field.document_type, fol.Detections):
                    result.append(name)
    except Exception:
        pass
    return result


class ADEExtractFields(foo.Operator):
    """Extract typed structured fields from documents using a form-based schema.

    Users define fields via a dynamic form (name + description + type). Values are
    stored as properly-typed top-level sample fields so FiftyOne infers the correct
    field type (StringField, FloatField, BooleanField) and shows matching filter
    widgets in the App sidebar.

    Grounding boxes correlating each extracted value to its document location are
    stored as ``fo.Detections`` in ``{result_field}_grounding``.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="ade_extract_fields",
            label="ADE: Extract Fields",
            description=(
                "Extract typed fields from documents using a form-based schema. "
                "Each field gets a proper FiftyOne type (text / number / boolean) "
                "with matching App filter widgets."
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
        add_region_input(inputs)

        inputs.bool(
            "parse_first",
            label="Parse document first",
            description=(
                "Parse each document before extracting. "
                "Disable if Markdown is already stored in a dataset field."
            ),
            default=True,
        )

        parse_first = ctx.params.get("parse_first", True)

        if parse_first:
            add_model_input(inputs)
            inputs.str(
                "save_parse_field",
                label="Save Markdown to field",
                description=(
                    "Save the parsed Markdown to this dataset field so you can "
                    "re-run Extract Fields later with different schemas without "
                    "paying parse credits again. Leave blank to skip."
                ),
                default="ade_parse",
            )
        else:
            inputs.str(
                "parse_field",
                label="Existing Markdown field",
                description="Dataset field that holds the parsed Markdown.",
                default="ade_parse",
                required=True,
            )

            det_fields = _get_detections_fields(ctx)
            if not det_fields:
                inputs.view(
                    "no_grounding_warning",
                    types.Notice(
                        label=(
                            "No detection fields found on this dataset. "
                            "Extracted values will be saved without grounding boxes. "
                            "Run 'ADE: Parse Document' first to enable bbox correlation."
                        )
                    ),
                )
            else:
                grounding_dropdown = types.Dropdown()
                for name in det_fields:
                    grounding_dropdown.add_choice(name, label=name)
                default_grounding = "ade_grounding" if "ade_grounding" in det_fields else det_fields[0]
                inputs.enum(
                    "grounding_field",
                    values=grounding_dropdown.values(),
                    label="Grounding field",
                    description=(
                        "Detection field with parse chunk IDs used to correlate "
                        "extracted values to their bounding boxes."
                    ),
                    default=default_grounding,
                    required=True,
                    view=grounding_dropdown,
                )

        field_type_dropdown = _build_field_type_dropdown()
        field_schema = types.Object()
        field_schema.str(
            "name",
            label="Field name",
            description="Snake-case name used as the dataset field suffix.",
            required=True,
        )
        field_schema.str(
            "description",
            label="Description",
            description="Tell the model what to extract (e.g. 'Total amount due including taxes').",
            required=True,
        )
        field_schema.enum(
            "type",
            values=field_type_dropdown.values(),
            label="Type",
            description="FiftyOne field type — determines which filter widget appears in the App.",
            default="string",
            required=True,
            view=field_type_dropdown,
        )

        inputs.list(
            "schema_fields",
            field_schema,
            label="Fields to extract",
            description=(
                "One row per field. Use + to add fields, − to remove. "
                "'number' fields get a range slider in the App; "
                "'boolean' fields get a true/false toggle."
            ),
            default=_DEFAULT_FIELDS,
        )

        inputs.str(
            "result_field",
            label="Output field prefix",
            description=(
                "Prefix for stored fields. "
                "E.g. 'ade_extraction' → ade_extraction_invoice_number, ade_extraction_total_amount."
            ),
            default="ade_extraction",
            required=True,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        api_key = get_api_key(ctx)
        region = ctx.params.get("region", "us")
        parse_first = ctx.params.get("parse_first", True)
        model = ctx.params.get("model", "dpt-2")
        parse_field = ctx.params.get("parse_field", "ade_parse")
        save_parse_field = (ctx.params.get("save_parse_field") or "").strip()
        grounding_field = ctx.params.get("grounding_field", "ade_grounding")
        result_field = ctx.params.get("result_field", "ade_extraction")
        schema_fields = ctx.params.get("schema_fields") or _DEFAULT_FIELDS

        properties = {}
        for f in schema_fields:
            name = (f.get("name") or "").strip().replace(" ", "_")
            if not name:
                continue
            field_type = f.get("type", "string")
            if field_type not in _SUPPORTED_TYPES:
                field_type = "string"
            properties[name] = {"type": field_type, "description": f.get("description", "")}

        if not properties:
            return {
                "error": "No fields defined. Add at least one field to extract.",
                "processed": 0,
                "total": 0,
            }

        json_schema_str = json.dumps({"type": "object", "properties": properties})

        for key, props in properties.items():
            field_name = f"{result_field}_{key}"
            expected_type = _FO_TYPE_MAP.get(props.get("type", "string"))
            existing_field = ctx.dataset.get_field(field_name)
            if existing_field is not None and expected_type and not isinstance(existing_field, expected_type):
                try:
                    ctx.dataset.delete_sample_field(field_name)
                except Exception:
                    pass

        client = get_client(api_key, region)

        if parse_first:
            samples = filter_ade_samples(ctx.target_view())
        else:
            samples = [s for s in ctx.target_view() if s.get_field(parse_field) is not None]

        total = len(samples)

        if total == 0:
            msg = (
                "No supported documents found in the target view."
                if parse_first
                else f"No samples with field '{parse_field}' found. Run 'ADE: Parse Document' first."
            )
            return {"processed": 0, "total": 0, "errors": [], "message": msg}

        processed = 0
        errors = []
        total_credits = 0.0

        for i, sample in enumerate(samples):
            ctx.set_progress(progress=i / total, label=f"Extracting fields: {i + 1}/{total}…")
            try:
                parse_resp = None
                if parse_first:
                    parse_resp = client.parse(document=Path(sample.filepath), model=model)
                    markdown_content = parse_resp.markdown

                    if save_parse_field:
                        sample[save_parse_field] = markdown_content
                        sample[f"{save_parse_field}_metadata"] = {
                            "page_count": parse_resp.metadata.page_count,
                            "credit_usage": float(parse_resp.metadata.credit_usage or 0),
                            "filename": parse_resp.metadata.filename,
                            "duration_ms": parse_resp.metadata.duration_ms,
                            "model_version": getattr(parse_resp.metadata, "version", None),
                        }
                        if parse_resp.grounding:
                            detections = grounding_to_detections(parse_resp.grounding)
                            if detections:
                                sample["ade_grounding"] = detections
                else:
                    markdown_content = sample.get_field(parse_field)
                    if not markdown_content:
                        errors.append({
                            "filepath": sample.filepath,
                            "error": f"Field '{parse_field}' is empty — skipped.",
                        })
                        continue

                extract_resp = client.extract(schema=json_schema_str, markdown=markdown_content)

                extraction = extract_resp.extraction
                if not isinstance(extraction, dict):
                    extraction = json.loads(json.dumps(extraction, default=str))

                for key, value in extraction.items():
                    field_type = properties.get(key, {}).get("type", "string")
                    if field_type == "number":
                        try:
                            clean = "".join(
                                c for c in str(value).replace(",", "").strip()
                                if c.isdigit() or c in ".+-"
                            )
                            value = float(clean) if clean else None
                        except (ValueError, TypeError):
                            value = None
                    elif field_type == "boolean":
                        value = value.lower() in ("true", "yes", "1") if isinstance(value, str) else bool(value)
                    if value is not None:
                        sample[f"{result_field}_{key}"] = value

                extraction_metadata = extract_resp.extraction_metadata
                if isinstance(extraction_metadata, dict):
                    chunk_map = _build_chunk_map(sample, parse_resp, grounding_field)
                    detections = _extraction_metadata_to_detections(extraction_metadata, chunk_map)
                    if detections:
                        sample[f"{result_field}_grounding"] = fol.Detections(detections=detections)

                sample[f"{result_field}_meta"] = {
                    "credit_usage": float(getattr(extract_resp.metadata, "credit_usage", 0) or 0),
                    "schema_violation_error": getattr(extract_resp.metadata, "schema_violation_error", None),
                    "model_version": getattr(extract_resp.metadata, "fallback_model_version", None),
                }
                total_credits += float(getattr(extract_resp.metadata, "credit_usage", 0) or 0)
                sample.save()
                processed += 1

            except Exception as e:
                errors.append({"filepath": sample.filepath, "error": str(e)})

        ctx.trigger("reload_dataset")

        return {
            "processed": processed,
            "total": total,
            "errors": errors[:5],
            "result_field": result_field,
            "field_count": len(properties),
            "total_credits": round(total_credits, 2),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        result = ctx.results or {}

        processed = result.get("processed", 0)
        total = result.get("total", 0)
        errors = result.get("errors", [])
        result_field = result.get("result_field", "ade_extraction")
        field_count = result.get("field_count", 0)
        total_credits = result.get("total_credits", 0)
        message = result.get("message", "")
        error_msg = result.get("error", "")

        if error_msg:
            outputs.view("error_notice", types.Notice(label=f"Error: {error_msg}"))
            return types.Property(outputs)

        if message:
            outputs.view("notice", types.Notice(label=message))
            return types.Property(outputs)

        summary = (
            f"Extracted {field_count} field(s) from {processed}/{total} samples. "
            f"Stored under '{result_field}_*'. "
            f"Total credits used: {total_credits}."
        )
        if errors:
            summary += f" {len(errors)} error(s) — see below."

        outputs.view("summary", types.Notice(label=summary))

        for i, err in enumerate(errors):
            outputs.str(
                f"error_{i}",
                label=f"Error: {err.get('filepath', 'unknown')}",
                default=err.get("error", ""),
            )

        return types.Property(outputs)


def _build_chunk_map(sample, parse_resp, grounding_field: str) -> dict:
    """Return ``{chunk_id: bounding_box}`` from a live parse response or stored grounding."""
    chunk_map = {}
    if parse_resp is not None:
        for element_id, g in (parse_resp.grounding or {}).items():
            if g and g.box:
                chunk_map[element_id] = ade_box_to_fo(g.box)
    else:
        stored = sample.get_field(grounding_field)
        if stored and stored.detections:
            for det in stored.detections:
                chunk_id = getattr(det, "chunk_id", None)
                if chunk_id:
                    chunk_map[chunk_id] = det.bounding_box
    return chunk_map


def _extraction_metadata_to_detections(extraction_metadata: dict, chunk_map: dict) -> list:
    """Convert extraction metadata and a chunk map into a ``fo.Detection`` list."""
    detections = []
    for field_name, meta in extraction_metadata.items():
        if not isinstance(meta, dict):
            continue
        value = meta.get("value", "")
        for ref in meta.get("references", []):
            bbox = chunk_map.get(ref)
            if bbox is None:
                continue
            detections.append(
                fol.Detection(label=field_name, bounding_box=bbox, value=str(value))
            )
    return detections
