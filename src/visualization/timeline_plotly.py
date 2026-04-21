from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from PIL import Image

from src.utils import load_json


def _safe_load_json(path: Path) -> Any | None:
        if not path.exists():
                return None
        try:
                return load_json(path)
        except Exception:
                return None


def _image_to_data_uri(image_path: str | Path, max_width: int = 280) -> str | None:
        path = Path(image_path)
        if not path.exists():
                return None

        try:
                with Image.open(path) as image:
                        image = image.convert("RGB")
                        ratio = max_width / float(image.width)
                        new_size = (max_width, max(1, int(image.height * ratio))) if image.width > max_width else image.size
                        if new_size != image.size:
                                image = image.resize(new_size, Image.Resampling.LANCZOS)

                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=82, optimize=True)
                        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                        return f"data:image/jpeg;base64,{encoded}"
        except Exception:
                return None


def _topic_color_map(topic_ids: list[int]) -> dict[int, str]:
        palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Safe + px.colors.qualitative.Set3
        colors: dict[int, str] = {-1: "#8a8a8a"}
        unique_topics = [topic_id for topic_id in dict.fromkeys(topic_ids) if topic_id != -1]
        for index, topic_id in enumerate(unique_topics):
                colors[topic_id] = palette[index % len(palette)]
        return colors


def _figure_div(fig: go.Figure, include_js: bool = False) -> str:
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if include_js else False, config={"responsive": True})


def _segment_snippet(text: str, limit: int = 220) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
                return cleaned
        return cleaned[: limit - 1].rsplit(" ", 1)[0] + "…"


def _build_topic_card(
        topic_id: int,
        topic_records: dict[int, dict[str, Any]],
        topic_summary: str | None,
        segments: list[dict[str, Any]],
        frame_map: dict[int, list[str]],
        color: str,
) -> str:
        topic_info = topic_records.get(topic_id, {})
        representation = topic_info.get("Representation", []) or []
        title = topic_info.get("Name", f"Topic {topic_id}")
        count = topic_info.get("Count", len(segments))

        representative_segments = sorted(segments, key=lambda seg: (len(str(seg.get("text", ""))), float(seg.get("start_s", 0.0))), reverse=True)[:3]
        frame_cards: list[str] = []
        for segment in representative_segments:
                seg_id = int(segment.get("segment_id", -1))
                frames = frame_map.get(seg_id, [])[:3]
                frame_html = []
                for frame_path in frames:
                        data_uri = _image_to_data_uri(frame_path)
                        if data_uri:
                                frame_html.append(f'<img src="{data_uri}" alt="segment {seg_id} frame" />')
                frame_cards.append(
                        f'''
                        <article class="segment-card" data-topic-card="{topic_id}">
                            <div class="segment-card__meta">
                                <span class="segment-card__time">{segment.get("start_s", 0.0):.1f}s - {segment.get("end_s", 0.0):.1f}s</span>
                                <span class="segment-card__speaker">{escape(str(segment.get("speaker", "unknown")))}</span>
                            </div>
                            <p class="segment-card__text">{escape(_segment_snippet(str(segment.get("text", ""))))}</p>
                            <div class="frame-strip">{''.join(frame_html)}</div>
                        </article>
                        '''
                )

        topic_badges = " ".join(f'<span class="topic-word">{escape(str(word))}</span>' for word in representation[:8])
        summary_html = escape(topic_summary) if topic_summary else escape(" ".join(str(seg.get("text", "")) for seg in representative_segments[:2]))

        return f'''
        <section class="topic-card" data-topic-section="{topic_id}" style="--topic-color:{color};">
            <div class="topic-card__header">
                <div>
                    <div class="topic-card__eyebrow">Topic {topic_id}</div>
                    <h3>{escape(str(title))}</h3>
                </div>
                <div class="topic-card__count">{count} segments</div>
            </div>
            <div class="topic-card__summary">{summary_html}</div>
            <div class="topic-card__words">{topic_badges}</div>
            <div class="topic-card__frames">{''.join(frame_cards)}</div>
        </section>
        '''


def _build_topic_activity_heatmap(df: pd.DataFrame) -> go.Figure:
        if df.empty:
                fig = go.Figure()
                fig.update_layout(title="Topic activity over time")
                return fig

        max_end = float(df["end"].max()) or 1.0
        bins = min(18, max(6, int(len(df) / 20) or 6))
        df = df.copy()
        df["time_bin"] = pd.cut(df["start"], bins=bins, labels=False, include_lowest=True, duplicates="drop")
        pivot = df.pivot_table(index="topic", columns="time_bin", values="text", aggfunc="count", fill_value=0)
        pivot = pivot.sort_index()

        x_labels = []
        for idx in pivot.columns:
                left = max_end * (float(idx) / bins)
                right = max_end * (float(idx + 1) / bins)
                x_labels.append(f"{left:.0f}–{right:.0f}s")

        fig = go.Figure(
                data=go.Heatmap(
                        z=pivot.values,
                        x=x_labels,
                        y=[f"Topic {int(topic)}" for topic in pivot.index],
                        colorscale="Viridis",
                        hovertemplate="topic=%{y}<br>time bin=%{x}<br>count=%{z}<extra></extra>",
                )
        )
        fig.update_layout(
                title="Topic activity over time",
                xaxis_title="Time bins",
                yaxis_title="Topics",
                margin=dict(l=40, r=20, t=50, b=40),
                height=360,
        )
        return fig


def build_timeline(
    segments: list[dict[str, Any]],
    out_html: str | Path,
    title: str = "Topic Timeline by Speaker",
        processed_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
) -> Path:
        records: list[dict[str, Any]] = []
        for seg in segments:
                records.append(
                        {
                                "segment_id": int(seg.get("segment_id", len(records))),
                                "speaker": seg.get("speaker", "unknown"),
                                "topic": int(seg.get("topic", -1)),
                                "start": float(seg["start_s"]),
                                "end": float(seg["end_s"]),
                                "duration": float(seg["end_s"]) - float(seg["start_s"]),
                                "text": seg.get("text", ""),
                        }
                )

        df = pd.DataFrame(records)
        if df.empty:
                raise ValueError("No segments available to build the timeline visualization")

        processed_path = Path(processed_dir) if processed_dir else None
        output_path = Path(output_dir) if output_dir else None

        frame_map: dict[int, list[str]] = {}
        topic_info_records: list[dict[str, Any]] = []
        topic_summaries: dict[str, str] = {}
        if processed_path is not None:
                frame_map = {
                        int(topic_segment_id): paths
                        for topic_segment_id, paths in (_safe_load_json(processed_path / "segment_frames.json") or {}).items()
                }
        if output_path is not None:
                topic_info_records = _safe_load_json(output_path / "topic_info.json") or []
                topic_summaries = _safe_load_json(output_path / "topic_summaries.json") or {}

        topic_records = {int(row.get("Topic", -1)): row for row in topic_info_records}
        topic_ids = list(dict.fromkeys(int(topic) for topic in df["topic"].tolist()))
        colors = _topic_color_map(topic_ids)

        fig_timeline = px.timeline(
                df,
                x_start="start",
                x_end="end",
                y="speaker",
                color="topic",
                color_discrete_map={str(topic_id): color for topic_id, color in colors.items()},
                hover_data={"segment_id": True, "text": True, "start": ":.1f", "end": ":.1f", "duration": ":.1f"},
                title=title,
        )
        fig_timeline.update_yaxes(autorange="reversed")
        fig_timeline.update_layout(height=520, margin=dict(l=40, r=20, t=60, b=40))

        topic_counts = df.groupby("topic").size().reset_index(name="count").sort_values(["count", "topic"], ascending=[False, True])
        topic_counts["label"] = topic_counts["topic"].map(lambda topic_id: f"Topic {int(topic_id)}")
        topic_counts["color"] = topic_counts["topic"].map(colors)
        fig_counts = go.Figure(
                go.Bar(
                        x=topic_counts["count"],
                        y=topic_counts["label"],
                        orientation="h",
                        marker_color=topic_counts["color"],
                        text=topic_counts["count"],
                        textposition="outside",
                        hovertemplate="%{y}<br>count=%{x}<extra></extra>",
                )
        )
        fig_counts.update_layout(
                title="Topic distribution",
                xaxis_title="Segment count",
                yaxis_title="",
                height=max(320, 140 + 42 * len(topic_counts)),
                margin=dict(l=90, r=20, t=50, b=40),
                showlegend=False,
        )

        fig_activity = _build_topic_activity_heatmap(df)

        total_duration = float(df["end"].max())
        total_segments = int(len(df))
        total_topics = len([topic_id for topic_id in topic_ids if topic_id != -1])
        speaker_count = df["speaker"].nunique()
        noise_count = int((df["topic"] == -1).sum())

        ordered_topics = topic_counts["topic"].tolist()
        topic_filter_buttons = [
                '<button class="topic-filter is-active" type="button" data-topic-filter="all">All topics</button>'
        ]
        for topic_id in ordered_topics:
                topic_filter_buttons.append(
                        f'<button class="topic-filter" type="button" data-topic-filter="{int(topic_id)}">Topic {int(topic_id)}</button>'
                )

        topic_cards = []
        for topic_id in ordered_topics:
                topic_segments = df[df["topic"] == topic_id].sort_values("start")
                topic_segment_dicts = [segments[int(idx)] for idx in topic_segments.index]
                topic_summary = topic_summaries.get(str(int(topic_id)))
                topic_cards.append(
                        _build_topic_card(
                                int(topic_id),
                                topic_records,
                                topic_summary,
                                topic_segment_dicts,
                                frame_map,
                                colors[int(topic_id)],
                        )
                )

        legend_items = []
        for topic_id in ordered_topics:
                legend_items.append(
                        f'<span class="topic-legend__item" style="--topic-color:{colors[int(topic_id)]};">Topic {int(topic_id)}</span>'
                )
        if -1 in colors and -1 in df["topic"].values:
                legend_items.append('<span class="topic-legend__item" style="--topic-color:#8a8a8a;">Noise</span>')

        dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(title)}</title>
    <style>
        :root {{
            --bg: #0f1320;
            --bg-soft: #151b2d;
            --panel: #1a2237;
            --panel-2: #20283f;
            --text: #edf2ff;
            --muted: #aeb8d6;
            --border: rgba(255, 255, 255, 0.08);
            --shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(92, 124, 250, 0.2), transparent 32%),
                radial-gradient(circle at top right, rgba(15, 188, 155, 0.15), transparent 28%),
                linear-gradient(180deg, #0b1020 0%, #11182a 45%, #0d1322 100%);
            color: var(--text);
        }}
        .page {{ max-width: 1600px; margin: 0 auto; padding: 28px 24px 48px; }}
        .hero {{
            background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 28px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }}
        .hero h1 {{ margin: 0 0 10px; font-size: clamp(2rem, 3vw, 3.5rem); line-height: 1.02; }}
        .hero p {{ margin: 0; color: var(--muted); max-width: 960px; line-height: 1.6; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
        .metric {{ background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 18px; padding: 16px 18px; }}
        .metric .value {{ font-size: 1.8rem; font-weight: 700; }}
        .metric .label {{ color: var(--muted); margin-top: 6px; font-size: 0.95rem; }}
        .top-grid {{ display: grid; grid-template-columns: 1.6fr 1fr; gap: 18px; margin-top: 22px; }}
        .panel {{ background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 24px; padding: 18px; box-shadow: var(--shadow); }}
        .panel h2 {{ margin: 0 0 14px; font-size: 1.05rem; letter-spacing: 0.02em; text-transform: uppercase; color: var(--muted); }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
        .topic-legend__item {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.04); border: 1px solid var(--border);
        }}
        .topic-legend__item::before {{
            content: ''; width: 10px; height: 10px; border-radius: 999px; background: var(--topic-color);
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--topic-color) 22%, transparent);
        }}
        .topic-bar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
        .topic-filter {{
            appearance: none; border: 1px solid var(--border); background: rgba(255,255,255,0.05);
            color: var(--text); padding: 8px 12px; border-radius: 999px; cursor: pointer;
            transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
        }}
        .topic-filter:hover {{ transform: translateY(-1px); }}
        .topic-filter.is-active {{ background: rgba(118, 146, 255, 0.18); border-color: rgba(118, 146, 255, 0.5); }}
        .chart-stack {{ display: grid; gap: 18px; }}
        .topic-list {{ display: grid; gap: 18px; margin-top: 22px; }}
        .topic-card {{
            background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 24px;
            padding: 18px; box-shadow: var(--shadow); border-left: 6px solid var(--topic-color);
        }}
        .topic-card__header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }}
        .topic-card__eyebrow {{ color: var(--muted); font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.06em; }}
        .topic-card h3 {{ margin: 4px 0 0; font-size: 1.4rem; }}
        .topic-card__count {{ background: rgba(255,255,255,0.06); border: 1px solid var(--border); padding: 10px 14px; border-radius: 16px; color: var(--muted); }}
        .topic-card__summary {{ margin: 14px 0; color: #dae2ff; line-height: 1.7; }}
        .topic-card__words {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
        .topic-word {{ background: rgba(255,255,255,0.06); border: 1px solid var(--border); border-radius: 999px; padding: 6px 10px; color: var(--text); font-size: 0.9rem; }}
        .topic-card__frames {{ display: grid; gap: 14px; }}
        .segment-card {{ background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 18px; padding: 14px; }}
        .segment-card__meta {{ display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted); font-size: 0.88rem; margin-bottom: 8px; }}
        .segment-card__time, .segment-card__speaker {{ background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 999px; padding: 5px 9px; }}
        .segment-card__text {{ margin: 0 0 12px; line-height: 1.55; color: #ebefff; }}
        .frame-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
        .frame-strip img {{ width: 100%; border-radius: 14px; border: 1px solid var(--border); display: block; object-fit: cover; }}
        .muted {{ color: var(--muted); }}
        @media (max-width: 1100px) {{
            .metrics, .top-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <h1>{escape(title)}</h1>
            <p>Interactive topic timeline with speaker lanes, color-coded topics, frame snapshots, topic distributions, and a topic activity heatmap built from the saved pipeline outputs.</p>
            <div class="metrics">
                <div class="metric"><div class="value">{total_segments}</div><div class="label">segments</div></div>
                <div class="metric"><div class="value">{total_topics}</div><div class="label">topics</div></div>
                <div class="metric"><div class="value">{total_duration:.0f}s</div><div class="label">timeline duration</div></div>
                <div class="metric"><div class="value">{speaker_count}</div><div class="label">speakers · {noise_count} noise</div></div>
            </div>
            <div class="legend">{''.join(legend_items)}</div>
        </section>

        <section class="top-grid">
            <div class="panel">
                <h2>Timeline</h2>
                { _figure_div(fig_timeline, include_js=True) }
            </div>
            <div class="panel chart-stack">
                <div>
                    <h2>Topic Distribution</h2>
                    { _figure_div(fig_counts) }
                </div>
                <div>
                    <h2>Topic Activity</h2>
                    { _figure_div(fig_activity) }
                </div>
            </div>
        </section>

        <section class="panel" style="margin-top: 22px;">
            <h2>Topics + Frame Snaps</h2>
            <div class="topic-bar">{''.join(topic_filter_buttons)}</div>
            <div class="topic-list">{''.join(topic_cards)}</div>
        </section>
    </main>

    <script>
        (function() {{
            const buttons = Array.from(document.querySelectorAll('[data-topic-filter]'));
            const topicCards = Array.from(document.querySelectorAll('[data-topic-section]'));
            const segmentCards = Array.from(document.querySelectorAll('[data-topic-card]'));

            function setActive(topic) {{
                buttons.forEach((button) => button.classList.toggle('is-active', button.dataset.topicFilter === topic));
                topicCards.forEach((card) => {{
                    const visible = topic === 'all' || card.dataset.topicSection === topic;
                    card.style.display = visible ? '' : 'none';
                }});
                segmentCards.forEach((card) => {{
                    const visible = topic === 'all' || card.dataset.topicCard === topic;
                    card.style.display = visible ? '' : 'none';
                }});
            }}

            buttons.forEach((button) => button.addEventListener('click', () => setActive(button.dataset.topicFilter)));
            window.addEventListener('hashchange', () => {{
                const hash = window.location.hash.replace('#topic-', '');
                if (hash) {{ setActive(hash); }}
            }});

            const initial = window.location.hash.replace('#topic-', '') || 'all';
            setActive(initial);
        }})();
    </script>
</body>
</html>"""

        target = Path(out_html)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dashboard_html, encoding="utf-8")
        return target
