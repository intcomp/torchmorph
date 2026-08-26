"""Small build-time enhancements for the generated API reference."""

import re

_API_HEADING = re.compile(r'(<h[2-6]\s+id="torchmorph\.([^"]+)"[^>]*>)')


def on_page_content(html: str, **kwargs) -> str:
    """Add a short anchor alias before every mkdocstrings API heading."""

    def add_short_anchor(match: re.Match[str]) -> str:
        short_name = match.group(2).rsplit(".", 1)[-1]
        anchor = f'<a id="{short_name}" class="doc-short-anchor" aria-hidden="true"></a>'
        return f"{anchor}\n{match.group(1)}"

    return _API_HEADING.sub(add_short_anchor, html)
