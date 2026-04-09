"""Document splitting/classification operator for the LandingAI ADE FiftyOne plugin."""

from pathlib import Path

import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types

try:
    from .utils import (
        add_model_input,
        add_password_input,
        add_region_input,
        add_split_model_input,
        check_api_key,
        filter_ade_samples,
        get_api_key,
        get_client,
    )
except ImportError:
    from utils import (
        add_model_input,
        add_password_input,
        add_region_input,
        add_split_model_input,
        check_api_key,
        filter_ade_samples,
        get_api_key,
        get_client,
    )


_DEFAULT_SPLIT_CLASSES = [
    {"name": "Invoice",  "description": "A document requesting payment for goods or services"},
    {"name": "Contract", "description": "A legal agreement between two or more parties"},
    {"name": "Receipt",  "description": "Proof of payment for a transaction"},
]


class ADESplitDocument(foo.Operator):
    """Classify and split multi-document files by document type.

    Uses the LandingAI ADE Split API to identify and separate bundled documents
    (e.g. a PDF containing invoices, contracts, and receipts). Results are stored
    as a list of split summaries on each sample.

    .. note::
        The Split API is currently in **preview** and is not recommended for production use.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="ade_split_document",
            label="ADE: Split / Classify Document",
            description=(
                "Classify and split multi-document files by type using the "
                "LandingAI ADE Split API. NOTE: Split API is currently in preview."
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
        inputs.view(
            "preview_warning",
            types.Notice(
                label=(
                    "The ADE Split API is currently in preview and is not recommended "
                    "for production workloads. Results may be inconsistent."
                )
            ),
        )

        add_region_input(inputs)

        inputs.bool(
            "parse_first",
            label="Parse document first",
            description=(
                "Parse each document before splitting. "
                "Disable if Markdown is already stored in a dataset field."
            ),
            default=True,
        )

        if ctx.params.get("parse_first", True):
            add_model_input(inputs)
            add_password_input(inputs)
        else:
            inputs.str(
                "parse_field",
                label="Existing Markdown field",
                description="Name of the dataset field that holds parsed Markdown.",
                default="ade_parse",
                required=True,
            )

        add_split_model_input(inputs)

        class_schema = types.Object()
        class_schema.str(
            "name",
            label="Document type name",
            description='Short label for this type (e.g. "Invoice", "Receipt").',
            required=True,
        )
        class_schema.str(
            "description",
            label="Description",
            description="Help the model understand what this document type looks like.",
            required=True,
        )
        class_schema.str(
            "identifier",
            label="Identifier field",
            description=(
                "Optional field to distinguish multiple documents of the same type "
                '(e.g. "invoice_number" or "statement_date").'
            ),
        )

        inputs.list(
            "split_classes",
            class_schema,
            label="Document types to classify",
            description="Add one row per type. You can optionally define an identifier. Maximum 19 classes per call.",
            default=_DEFAULT_SPLIT_CLASSES,
        )

        inputs.str(
            "result_field",
            label="Output field",
            description="Dataset field where split classification results will be stored.",
            default="ade_splits",
            required=True,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        api_key = get_api_key(ctx)
        region = ctx.params.get("region", "us")
        parse_first = ctx.params.get("parse_first", True)
        model = ctx.params.get("model", "dpt-2")
        password = (ctx.params.get("password") or "").strip()
        parse_field = ctx.params.get("parse_field", "ade_parse")
        split_model = ctx.params.get("split_model", "split-latest")
        result_field = ctx.params.get("result_field", "ade_splits")
        raw_split_classes = ctx.params.get("split_classes") or _DEFAULT_SPLIT_CLASSES

        split_classes = []
        for c in raw_split_classes:
            name = (c.get("name") or "").strip()
            if not name:
                continue

            entry = {
                "name": name,
                "description": (c.get("description") or "").strip(),
            }
            identifier = (c.get("identifier") or "").strip()
            if identifier:
                entry["identifier"] = identifier

            split_classes.append(entry)

        if not split_classes:
            return {"error": "No document types defined. Add at least one class.", "processed": 0, "total": 0}
        if len(split_classes) > 19:
            return {"error": f"Too many classes ({len(split_classes)}). Maximum is 19.", "processed": 0, "total": 0}

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
        all_classifications = []

        for i, sample in enumerate(samples):
            ctx.set_progress(progress=i / total, label=f"Splitting: {i + 1}/{total}…")
            try:
                if parse_first:
                    parse_kwargs = {"document": Path(sample.filepath), "model": model}
                    if password:
                        parse_kwargs["password"] = password

                    parse_resp = client.parse(**parse_kwargs)
                    markdown_content = parse_resp.markdown
                else:
                    markdown_content = sample.get_field(parse_field)
                    if not markdown_content:
                        errors.append({
                            "filepath": sample.filepath,
                            "error": f"Field '{parse_field}' is empty — skipped.",
                        })
                        continue

                split_resp = client.split(
                    split_class=split_classes,
                    markdown=markdown_content,
                    model=split_model,
                )
                split_version = getattr(split_resp.metadata, "version", None)
                splits = list(split_resp.splits or [])

                splits_summary = []
                for s in splits:
                    splits_summary.append({
                        "classification": s.classification,
                        "identifier": getattr(s, "identifier", None),
                        "pages": list(s.pages) if s.pages else [],
                        "page_count": len(s.pages) if s.pages else 0,
                        "markdown_preview": s.markdowns[0][:300] if s.markdowns else "",
                    })
                    all_classifications.append(s.classification)

                sample[result_field] = splits_summary
                sample[f"{result_field}_count"] = len(splits)
                sample[f"{result_field}_type"] = (
                    fol.Classification(label=splits[0].classification)
                    if splits
                    else None
                )
                sample[f"{result_field}_all_types"] = sorted({
                    s.classification for s in splits
                })
                sample[f"{result_field}_metadata"] = {
                    "credit_usage": float(getattr(split_resp.metadata, "credit_usage", 0) or 0),
                    "filename": getattr(split_resp.metadata, "filename", None),
                    "page_count": getattr(split_resp.metadata, "page_count", None),
                    "duration_ms": getattr(split_resp.metadata, "duration_ms", None),
                    "job_id": getattr(split_resp.metadata, "job_id", None),
                    "version": split_version,
                    "model_version": split_version,
                }

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
            "unique_classifications": sorted(set(all_classifications)),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        result = ctx.results or {}

        processed = result.get("processed", 0)
        total = result.get("total", 0)
        errors = result.get("errors", [])
        error_count = result.get("error_count", len(errors))
        result_field = result.get("result_field", "ade_splits")
        unique_classifications = result.get("unique_classifications", [])
        message = result.get("message", "")
        error_msg = result.get("error", "")

        if error_msg:
            outputs.view("error_notice", types.Notice(label=f"Error: {error_msg}"))
            return types.Property(outputs)

        if message:
            outputs.view("notice", types.Notice(label=message))
            return types.Property(outputs)

        summary = f"Processed {processed}/{total} samples. Split results stored in '{result_field}'."
        if unique_classifications:
            summary += f" Types found: {', '.join(unique_classifications)}."
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
