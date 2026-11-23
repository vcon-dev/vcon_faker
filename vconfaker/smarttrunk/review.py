"""Local review pages: an index.html per run, with each call's audio, transcript and script, and an
overall page for a corpus of runs (one folder per industry)."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#66645e;--card:#fff;--line:#e4e1d8;--ok:#1f7a3f;--bad:#b3261e;--tag:#efece4}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a19a;--card:#21211e;--line:#34332f;--ok:#6cc58b;--bad:#f08a80;--tag:#2c2b27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}p.lede{color:var(--muted);margin:0 0 20px}
table{border-collapse:collapse;margin:0 0 24px;font-size:14px;width:100%}td,th{padding:4px 12px 4px 0;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
table.stats td:first-child{color:var(--muted);width:30%}
article{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
article h2{font-size:16px;margin:0 0 6px}.meta{color:var(--muted);font-size:13px;margin:0 0 10px}
.pill{display:inline-block;background:var(--tag);border-radius:999px;padding:1px 9px;font-size:12px;margin:0 6px 4px 0}
.ok{color:var(--ok)}.bad{color:var(--bad)}audio{width:100%;margin:6px 0 10px}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}
.turn{margin:4px 0}.who{font-weight:600}.transcript{white-space:pre-wrap}
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{CSS}</style></head>
<body><main>
{body}
</main></body></html>
"""


def write_index(out: Path, calls: list[dict[str, Any]], run: dict[str, Any]) -> Path:
    stats = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in run.get("stats", {}).items())
    cards = []
    for c in calls:
        check = c["check"]
        verdict = (
            f'<span class="ok">passed</span> (WER {check["wer"]:.1%})'
            if check["passed"]
            else f'<span class="bad">failed</span> (WER {check["wer"]:.1%}; {_e("; ".join(check.get("uncovered_lines", []) + check.get("stray_numbers", [])))})'
        )
        pills = [f'{_e(c["business"])}, {_e(c["line"].replace("_", " "))} line']
        pills += [f"{_e(k)}: {_e(v)}" if v != "true" else _e(k) for k, v in c.get("features", {}).items()]
        if c.get("screen"):
            pills.append(f'screen: {_e(c["screen"]["status"])}')
        speakers = c["speakers"]
        turns = "".join(
            f'<div class="turn"><span class="who">{_e(speakers.get(t["speaker"]) or t["speaker"])}:</span> '
            f'{"(cuts in) " if t.get("interrupts") else ""}{_e(t["text"])}</div>'
            for t in c["script"]["turns"]
        )
        files = (
            f'<a href="vcons/{_e(c["vcon_file"])}">vCon</a> · uuid {_e(c["uuid"])} · '
            if c.get("vcon_file") else "no vCon (rejected) · "
        ) + f'<a href="scripts/{_e(c["call_id"])}.json">script</a> · <a href="transcripts/{_e(c["call_id"])}.json">transcript check</a>'
        voices = ", ".join(f"{k} {v}" for k, v in c["voices"].items())
        cards.append(f"""<article id="{_e(c['call_id'])}">
<h2>{c['index']:02d}. {_e(c['scenario_label'])}</h2>
<p>{''.join(f'<span class="pill">{p}</span>' for p in pills)}</p>
<p class="meta">{_e(c['start_local'])} · rang {c.get('ring_seconds', 0):.0f} s, then {c['duration']:.0f} s recorded on {c.get('channels', 1)} channel(s) ·
caller {_e(speakers['caller'])} · answered by {_e(speakers['business'])}{' · transferred to ' + _e(speakers['colleague']) if speakers.get('colleague') else ''} ·
voices {_e(voices)} · transcript check {verdict}, attempt {c['audio_attempts']}</p>
<p class="meta">{_e(c['script']['summary'])} Outcome: {_e(c['script']['outcome'])}.</p>
<audio controls preload="none" src="{'audio' if check['passed'] else 'rejected'}/{_e(c['audio_file'])}"></audio>
<div class="transcript"><strong>Transcript (parakeet-mlx, both channels by time):</strong> {_e(c.get('transcript_text', ''))}</div>
<details><summary>Script</summary>{turns}</details>
<details><summary>Files</summary>{files}</details>
</article>""")
    body = f"""<h1>{_e(run['title'])}</h1>
<p class="lede">{_e(run['lede'])}</p>
<table class="stats">{stats}</table>
{''.join(cards)}"""
    path = out / "index.html"
    path.write_text(_page(run["title"], body))
    return path


def write_corpus_index(root: Path) -> Path:
    """One page over every run folder under ``root`` that has a manifest."""
    rows = []
    totals = {"calls": 0, "passed": 0, "audio": 0.0, "bytes": 0, "wall": 0.0}
    cost = None
    for manifest_path in sorted(root.glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        s = m["summary"]
        folder = manifest_path.parent.name
        totals["calls"] += s["calls"]
        totals["passed"] += s["passed"]
        totals["audio"] += s["audio_seconds"]
        totals["bytes"] += s["mp3_bytes"]
        totals["wall"] += m["wall_seconds"]
        cost = m["api_usage_total"]["estimated_cost_usd"]
        rows.append(
            f'<tr><td><a href="{_e(folder)}/index.html">{_e(folder.replace("_", " "))}</a></td>'
            f'<td>{s["passed"]} of {s["calls"]}</td><td>{s["mean_wer"]:.1%} / {s["worst_wer"]:.1%}</td>'
            f'<td>{s["short_calls"]}</td><td>{s["audio_seconds"] / 60:.1f} min</td><td>{s["mp3_bytes"] / 1e6:.1f} MB</td>'
            f'<td>{m["wall_seconds"] / 60:.0f} min</td></tr>'
        )
    body = f"""<h1>Smart Trunk synthetic calls: {_e(root.name)}</h1>
<p class="lede">One folder per industry. Invented businesses, people, places and numbers; scripts by Claude, local voices,
two-channel phone audio, transcripts per channel. Every vCon is marked synthetic.</p>
<table><tr><th>Industry</th><th>Passed</th><th>WER mean / worst</th><th>15 to 45 s</th><th>Audio</th><th>MP3</th><th>Run time</th></tr>
{''.join(rows)}
<tr><th>All</th><th>{totals['passed']} of {totals['calls']}</th><th></th><th></th><th>{totals['audio'] / 60:.1f} min</th>
<th>{totals['bytes'] / 1e6:.1f} MB</th><th>{totals['wall'] / 60:.0f} min</th></tr></table>
<p class="lede">Script tokens so far: {f'${cost:.2f}' if cost is not None else 'n/a'} at list prices.</p>"""
    path = root / "index.html"
    path.write_text(_page(f"Synthetic calls {root.name}", body))
    return path
