"""
FiftyOne plugin for Landing AI Agentic Document Extraction (ADE).

Operators:
  - ade_parse_document   Sync parse: convert docs to Markdown + grounding Detections
  - ade_extract_fields   Schema-based field extraction from parsed content
  - ade_split_document   Classify and split multi-document files by type
"""

try:
    from .parse_document import ADEParseDocument
    from .extract_fields import ADEExtractFields
    from .split_document import ADESplitDocument
except ImportError:
    from parse_document import ADEParseDocument
    from extract_fields import ADEExtractFields
    from split_document import ADESplitDocument


def register(plugin):
    plugin.register(ADEParseDocument)
    plugin.register(ADEExtractFields)
    plugin.register(ADESplitDocument)
