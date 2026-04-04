# Landing AI ADE Plugin

A FiftyOne plugin that provides operators for parsing, extracting, and splitting
documents using [Landing AI](https://landing.ai)'s Agentic Document Extraction
(ADE) API. Converts PDFs, images, spreadsheets, and Office files into structured
Markdown with spatial bounding box grounding stored as native FiftyOne `Detections`.

## Installation

```shell
fiftyone plugins download https://github.com/landing-ai/computable-docs
```

Install the required dependencies:

```shell
fiftyone plugins requirements @landingai/ade --install
```

## Configuration

Set your Landing AI API key as the `VISION_AGENT_API_KEY` environment variable
or add it to [FiftyOne secrets](https://docs.voxel51.com/plugins/using_plugins.html#secrets).
You can obtain an API key from the [Landing AI dashboard](https://va.landing.ai).

```shell
export VISION_AGENT_API_KEY="your-api-key-here"
```

## Usage

1. Launch the App with a dataset that contains documents or images:

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "Voxel51/scanned_receipts",
    overwrite=True,
    persistent=True,
    name="scanned_receipts",
    max_samples=5,
)

session = fo.launch_app(dataset)
```

2. Press `` ` `` or click **Browse operations** to open the operator list.

3. Search for **ADE** to see all available operators.

## Operators

### `ade_parse_document`

Parse documents into structured Markdown with spatial bounding box grounding.

Calls the ADE synchronous Parse API. Each page element (text block, table,
figure, logo) becomes a `fo.Detection` with normalized coordinates visible in
the FiftyOne App grid and modal.

**How to use:**

1. Open the operator list and search for **ADE: Parse Document**
2. Choose a model, output field names, and whether to store grounding
3. Click **Execute**

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| Model | `dpt-2` (full, 3 credits/page) or `dpt-2-mini` (simple docs, 1.5 credits/page) | `dpt-2` |
| Region | `us` or `eu` endpoint | `us` |
| Output field (Markdown) | Field where the parsed Markdown text is stored | `ade_parse` |
| Store spatial grounding | Store element bounding boxes as `fo.Detections` | `True` |
| Output field (Grounding) | Field where grounding detections are stored | `ade_grounding` |
| Zero data retention | Prevent Landing AI from retaining document data (+1 credit/page) | `False` |

**What gets stored on each sample:**

| Field | Type | Content |
|-------|------|---------|
| `ade_parse` | `StringField` | Full parsed Markdown |
| `ade_grounding` | `Detections` | One detection per document element with label (type), bounding box, `chunk_id`, and `page` |
| `ade_parse_metadata` | `DictField` | `page_count`, `credit_usage`, `filename`, `duration_ms`, `model_version` |

---

### `ade_extract_fields`

Extract typed named fields from documents using a form-based schema.

Define each field with a name, description, and type (string, number, or boolean).
Values are stored as flat, properly-typed top-level sample fields so FiftyOne shows
the right filter widget in the App sidebar — range slider for numbers, toggle for
booleans, text search for strings.

When **Parse document first** is enabled, parsed Markdown and grounding boxes are
optionally saved so you can re-run with a different schema without paying parse
credits again.

**How to use:**

1. Open the operator list and search for **ADE: Extract Fields**
2. Choose whether to parse documents first or use an existing Markdown field
3. Define your fields in the form (name + description + type per row)
4. Click **Execute**

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| Parse document first | Call Parse API before extracting | `True` |
| Save Markdown to field | Persist parsed Markdown for future runs (only when parsing first) | `ade_parse` |
| Existing Markdown field | Field to read from when not parsing first | `ade_parse` |
| Grounding field | Detection field used for bbox correlation when not parsing first | `ade_grounding` |
| Model | Parse model (only used when parsing first) | `dpt-2` |
| Region | `us` or `eu` endpoint | `us` |
| Fields to extract | Form rows of `name` + `description` + `type` | Invoice example fields |
| Output field prefix | Prefix for all stored fields | `ade_extraction` |

**Default schema fields:**

| Name | Description | Type |
|------|-------------|------|
| `invoice_number` | The unique invoice identifier | string |
| `vendor_name` | Name of the vendor or supplier | string |
| `total_amount` | Total amount due including taxes | number |
| `invoice_date` | Date the invoice was issued | string |

**What gets stored on each sample:**

| Field | Type | Content |
|-------|------|---------|
| `ade_extraction_{field_name}` | `StringField` / `FloatField` / `BooleanField` | One field per schema entry, typed to match |
| `ade_extraction_grounding` | `Detections` | Bounding boxes correlating each extracted value to its document location |
| `ade_extraction_meta` | `DictField` | `credit_usage`, `schema_violation_error`, `model_version` |

When **Parse document first** is enabled and **Save Markdown to field** is non-empty, these additional fields are also written:

| Field | Type | Content |
|-------|------|---------|
| `ade_parse` | `StringField` | Parsed Markdown (reusable for future Extract or Split runs) |
| `ade_parse_metadata` | `DictField` | `page_count`, `credit_usage`, `filename`, `duration_ms`, `model_version` |
| `ade_grounding` | `Detections` | Grounding boxes for the parse chunks |

---

### `ade_split_document`

Classify and split multi-document files by document type.

Identify and separate bundled documents (e.g., a PDF that mixes invoices,
contracts, and receipts) by providing a list of document types to look for.

> **Note:** The ADE Split API is currently in **preview** and is not
> recommended for production workloads. Results may vary.

**How to use:**

1. Open the operator list and search for **ADE: Split / Classify Document**
2. Define your document types in the form (name + description per row)
3. Click **Execute**

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| Parse document first | Call Parse API before splitting | `False` |
| Existing Markdown field | Field to read from when not parsing first | `ade_parse` |
| Model | Parse model (only used when parsing first) | `dpt-2` |
| Region | `us` or `eu` endpoint | `us` |
| Document types to classify | Form rows of `name` + `description` (max 19) | Invoice / Contract / Receipt |
| Output field | Field where split results are stored | `ade_splits` |

**What gets stored on each sample:**

| Field | Type | Content |
|-------|------|---------|
| `ade_splits` | `ListField` | List of `{classification, identifier, pages, page_count, markdown_preview}` per split |
| `ade_splits_count` | `IntField` | Number of splits found |
| `ade_splits_type` | `Classification` | Primary document type (first split's classification) |
| `ade_splits_metadata` | `DictField` | `credit_usage`, `filename` |

---

## Models

| Model | Credits/page | Best for |
|-------|-------------|----------|
| `dpt-2` | 3 | Complex docs, scanned PDFs, tables, non-English, figures |
| `dpt-2-mini` | 1.5 | Simple digital docs, invoices, forms; not for scanned or complex tables |

---

## Supported File Types

| Category | Extensions |
|----------|-----------|
| Documents | `.pdf` `.docx` `.doc` `.odt` |
| Images | `.png` `.jpg` `.jpeg` `.bmp` `.tiff` `.tif` `.webp` `.gif` `.jp2` `.psd` |
| Spreadsheets | `.xlsx` `.xls` `.csv` |
| Presentations | `.ppt` `.pptx` |

Samples with unsupported extensions are silently skipped.

---
