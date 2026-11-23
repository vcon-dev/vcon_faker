"""A local index.html for listening to a run: one card per call, with its audio, transcript and script."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#66645e;--card:#fff;--line:#e4e1d8;--ok:#1f7a3f;--bad:#b3261e;--tag:#efece4}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a19a;--card:#21211e;--line:#34332f;--ok:#6cc58b;--bad:#f08a80;--tag:#2c2b27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}p.lede{color:var(--muted);margin:0 0 20px}
table.stats{border-collapse:collapse;margin:0 0 24px;font-size:14px}table.stats td{padding:3px 14px 3px 0;border-bottom:1px solid var(--line)}
table.stats td:first-child{color:var(--muted)}
article{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
article h2{font-size:16px;margin:0 0 6px}.meta{color:var(--muted);font-size:13px;margin:0 0 10px}
.pill{display:inline-block;background:var(--tag);border-radius:999px;padding:1px 9px;font-size:12px;margin-right:6px}
.ok{color:var(--ok)}.bad{color:var(--bad)}audio{width:100%;margin:6px 0 10px}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}
.turn{margin:4px 0}.who{font-weight:600}.transcript{white-space:pre-wrap}
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def write_index(out: Path, calls: list[dict[str, Any]], run: dict[str, Any]) -> Path:
    stats = run.get("stats", {})
    rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in stats.items())
    cards = []
    for c in calls:
        check = c["check"]
        verdict = (
            f'<span class="ok">passed</span> (WER {check["wer"]:.1%})'
            if check["passed"]
            else f'<span class="bad">failed</span> (WER {check["wer"]:.1%})'
        )
        screen = c.get("screen")
        screen_html = f'<span class="pill">screen: {_e(screen["status"])}</span>' if screen else ""
        turns = "".join(
            f'<div class="turn"><span class="who">{_e(c["speakers"][t["speaker"]])}:</span> {_e(t["text"])}</div>'
            for t in c["script"]["turns"]
        )
        files = (
            f'<a href="vcons/{_e(c["vcon_file"])}">vCon</a> · uuid {_e(c["uuid"])} · '
            if c.get("vcon_file") else "no vCon (rejected) · "
        ) + f'<a href="scripts/{_e(c["call_id"])}.json">script</a> · <a href="transcripts/{_e(c["call_id"])}.json">transcript check</a>'
        cards.append(
            f"""<article id="{_e(c['call_id'])}">
<h2>{c['index']:02d}. {_e(c['scenario_label'])} <span class="pill">{_e(c['business'])}, {_e(c['line'])} line</span>{screen_html}</h2>
<p class="meta">{_e(c['start_local'])} · {c['duration']:.0f} s · caller {_e(c['speakers']['caller'])} ({_e(c['voices']['caller'])}),
answered by {_e(c['speakers']['business'])} ({_e(c['voices']['business'])}) · transcript check {verdict}, attempt {c['audio_attempts']}</p>
<p class="meta">{_e(c['script']['summary'])} Outcome: {_e(c['script']['outcome'])}.</p>
<audio controls preload="none" src="{'audio' if check['passed'] else 'rejected'}/{_e(c['audio_file'])}"></audio>
<div class="transcript"><strong>Transcript (parakeet-mlx):</strong> {_e(check_text(c))}</div>
<details><summary>Script</summary>{turns}</details>
<details><summary>Files</summary>{files}</details>
</article>"""
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic calls pilot</title><style>{CSS}</style></head>
<body><main>
<h1>{_e(run['title'])}</h1>
<p class="lede">{_e(run['lede'])}</p>
<table class="stats">{rows}</table>
{''.join(cards)}
</main></body></html>
"""
    path = out / "index.html"
    path.write_text(page)
    return path


def check_text(c: dict[str, Any]) -> str:
    return c.get("transcript_text", "")
