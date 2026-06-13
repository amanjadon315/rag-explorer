import re
import unicodedata
from dataclasses import dataclass
from typing import List


@dataclass
class CleanResult:
    original: str
    cleaned: str
    changes: List[str]       # human-readable list of what was changed
    original_length: int
    cleaned_length: int
    reduction_pct: float


class TextCleaner:
    """
    Cleans raw text extracted from documents.
    Every transformation is tracked so the frontend can show exactly
    what changed at this pipeline stage.
    """

    def clean(self, text: str) -> CleanResult:
        changes = []
        result = text

        # 1. Normalize unicode (e.g. fancy quotes → straight quotes)
        normalized = unicodedata.normalize("NFKC", result)
        if normalized != result:
            changes.append("Unicode normalized (NFKC)")
        result = normalized

        # 2. Remove null bytes and control characters (except \n \t)
        cleaned_ctrl = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", result)
        if cleaned_ctrl != result:
            changes.append("Removed control characters")
        result = cleaned_ctrl

        # 3. Strip HTML tags
        stripped_html = re.sub(r"<[^>]+>", " ", result)
        if stripped_html != result:
            changes.append("Stripped HTML tags")
        result = stripped_html

        # 4. Collapse multiple blank lines → single blank line
        collapsed = re.sub(r"\n{3,}", "\n\n", result)
        if collapsed != result:
            changes.append("Collapsed multiple blank lines")
        result = collapsed

        # 5. Collapse multiple spaces → single space (within lines)
        spaces = re.sub(r"[ \t]{2,}", " ", result)
        if spaces != result:
            changes.append("Collapsed multiple spaces/tabs")
        result = spaces

        # 6. Strip leading/trailing whitespace per line
        lines = [line.strip() for line in result.splitlines()]
        stripped = "\n".join(lines)
        if stripped != result:
            changes.append("Stripped trailing whitespace per line")
        result = stripped

        # 7. Final strip
        result = result.strip()

        orig_len = len(text)
        clean_len = len(result)
        reduction = round((1 - clean_len / orig_len) * 100, 1) if orig_len > 0 else 0.0

        return CleanResult(
            original=text,
            cleaned=result,
            changes=changes if changes else ["No changes needed"],
            original_length=orig_len,
            cleaned_length=clean_len,
            reduction_pct=reduction,
        )
