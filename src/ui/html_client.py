"""
html_client.py

Assembles _HTML_TEMPLATE at import time by reading the separate
CSS, JS, and HTML template files from the ui/ directory tree:

  ui/
    static/
      poker.css     — all styles
      state.js      — shared state and lookup tables
      utils.js      — fmt(), esc(), show(), addLog()
      render.js     — DOM rendering (cards, players, banner)
      actions.js    — action buttons and bet slider
      overlays.js   — hand-result and game-over overlays
      network.js    — WebSocket connection and message dispatch
    templates/
      index.html    — pure HTML markup with {{CSS}} / {{JS}} placeholders

The assembled string still contains {{WS_PORT}}, which UITableManager
replaces at serve time with the actual WebSocket port number.
"""

from pathlib import Path

_HERE = Path(__file__).parent

# JS modules in dependency order (each module assumes the previous ones
# have already been evaluated, so order matters).
_JS_MODULES = [
    "state.js",
    "utils.js",
    "render.js",
    "spectator.js",
    "actions.js",
    "overlays.js",
    "network.js",
]


def _build_template() -> str:
    static = _HERE / "static"
    templates = _HERE / "templates"

    css = (
        (static / "poker.css").read_text(encoding="utf-8")
        + "\n"
        + (static / "spectator.css").read_text(encoding="utf-8")
    )
    js = "\n\n".join(
        (static / name).read_text(encoding="utf-8") for name in _JS_MODULES
    )
    html = (templates / "index.html").read_text(encoding="utf-8")

    return html.replace("{{CSS}}", css).replace("{{JS}}", js)


_HTML_TEMPLATE = _build_template()
