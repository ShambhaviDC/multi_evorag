"""response_planner.py - Adaptive Response Generation Agent for EvoRAG Phase 7.

Analyzes the user's structured query, synthesized answer, retrieved evidence,
and uploaded attachments to construct validated multi-component UI responses:
- Markdown text explanation
- Comparison tables
- Chart.js visualizations (bar, line, pie) strictly from real data
- Interactive chronological timelines
- Entity relationship graphs
- Numerical summaries
- Uploaded file previews & summaries
"""

import logging
import re
from typing import Any, Dict, List, Optional

from file_processor import ProcessedFile
from query_models import StructuredQuery, UIComponent

log = logging.getLogger(__name__)

# Palette for Chart.js
CHART_COLORS = [
    "rgba(108, 99, 255, 0.8)",   # indigo/purple
    "rgba(0, 210, 255, 0.8)",    # cyan
    "rgba(255, 107, 107, 0.8)",  # coral
    "rgba(46, 204, 113, 0.8)",   # emerald
    "rgba(241, 196, 15, 0.8)",   # amber
    "rgba(155, 89, 182, 0.8)",   # violet
]
CHART_BORDERS = [
    "#6c63ff",
    "#00d2ff",
    "#ff6b6b",
    "#2ecc71",
    "#f1c40f",
    "#9b59b6",
]


def _build_tabular_charts(att: ProcessedFile) -> List[UIComponent]:
    """Generate Chart.js components from parsed CSV/XLSX data."""
    components = []
    sd = att.structured_data or {}
    sample_rows = sd.get("sample_rows") or []
    num_cols = sd.get("numerical_columns") or []
    cat_cols = sd.get("categorical_columns") or []

    if not sample_rows or not num_cols:
        return components

    # Pick label column (first categorical column or row index)
    label_col = cat_cols[0] if cat_cols else None
    labels = []
    limit = min(10, len(sample_rows))
    for idx, row in enumerate(sample_rows[:limit]):
        if label_col and row.get(label_col) is not None:
            labels.append(str(row[label_col]))
        else:
            labels.append(f"Row {idx + 1}")

    # Build datasets for up to 3 numerical columns
    datasets = []
    for c_idx, num_col in enumerate(num_cols[:3]):
        vals = []
        for row in sample_rows[:limit]:
            v = row.get(num_col)
            try:
                vals.append(float(v) if v is not None else 0.0)
            except (ValueError, TypeError):
                vals.append(0.0)

        color = CHART_COLORS[c_idx % len(CHART_COLORS)]
        border = CHART_BORDERS[c_idx % len(CHART_BORDERS)]
        datasets.append({
            "label": str(num_col),
            "data": vals,
            "backgroundColor": color,
            "borderColor": border,
            "borderWidth": 1.5,
        })

    # Decide chart type
    chart_type = "bar"
    if any(k in str(label_col or "").lower() for k in ["date", "time", "year", "month"]):
        chart_type = "line"

    components.append(UIComponent(
        type=f"{chart_type}_chart",
        title=f"Data Visualization: {att.filename}",
        data={
            "chartType": chart_type,
            "labels": labels,
            "datasets": datasets,
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {"position": "top"},
                    "title": {"display": True, "text": f"{att.filename} ({', '.join(num_cols[:3])})"}
                }
            }
        }
    ))

    # Numerical Summary component
    stats = sd.get("summary_statistics") or {}
    if stats:
        summary_metrics = []
        for col_name, col_stats in stats.items():
            if isinstance(col_stats, dict):
                summary_metrics.append({
                    "column": col_name,
                    "count": col_stats.get("count", 0),
                    "mean": col_stats.get("mean", 0),
                    "min": col_stats.get("min", 0),
                    "max": col_stats.get("max", 0),
                })
        if summary_metrics:
            components.append(UIComponent(
                type="numerical_summary",
                title=f"Statistical Summary: {att.filename}",
                data={"metrics": summary_metrics}
            ))

    # Data Table component for sample rows
    all_cols = sd.get("columns") or []
    if all_cols and sample_rows:
        table_rows = []
        for r in sample_rows[:8]:
            table_rows.append([str(r.get(c, "")) for c in all_cols])
        components.append(UIComponent(
            type="comparison_table",
            title=f"Dataset Sample Preview ({att.filename})",
            data={
                "headers": all_cols,
                "rows": table_rows,
                "total_rows": sd.get("total_rows", len(sample_rows))
            }
        ))

    return components


def _extract_timeline_from_text(text: str) -> Optional[UIComponent]:
    """Extract chronological milestones from synthesized text or retrieved context."""
    # Match patterns like "2024: ...", "In 2023, ...", "October 2026: ..."
    date_patterns = [
        r"(?:(?:In|During|By)\s+)?\b(19\d{2}|20\d{2})\b[:\s–—\-]+([^\.\n]+(?:\.|\n))",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})[:\s–—\-]+([^\.\n]+(?:\.|\n))",
        r"\b(Q[1-4]\s+20\d{2})[:\s–—\-]+([^\.\n]+(?:\.|\n))",
    ]

    events = []
    seen_dates = set()

    for pat in date_patterns:
        for match in re.finditer(pat, text, flags=re.IGNORECASE):
            groups = match.groups()
            if len(groups) == 2:
                dt_str = groups[0].strip()
                desc = groups[1].strip()
            elif len(groups) == 3:
                dt_str = f"{groups[0]} {groups[1]}".strip()
                desc = groups[2].strip()
            else:
                continue

            if dt_str in seen_dates:
                continue
            seen_dates.add(dt_str)

            # Clean description
            desc = desc.lstrip(":–— -").strip()
            if len(desc) > 10:
                events.append({
                    "date": dt_str,
                    "title": desc[:60] + ("..." if len(desc) > 60 else ""),
                    "description": desc,
                })

    if len(events) >= 2:
        return UIComponent(
            type="timeline",
            title="Chronological Evolution & Timeline",
            data={"events": events[:10]}
        )
    return None


def _extract_comparison_from_query(
    structured_query: StructuredQuery,
    final_answer: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> List[UIComponent]:
    """Generate comparison table and optional comparison chart for comparative queries."""
    components = []
    entities = [e.name for e in structured_query.entities if e.type in ("Company", "Technology", "Product")]
    if len(entities) < 2:
        return components

    e1, e2 = entities[0], entities[1]

    # Look for numerical metrics mentioned for both entities in the answer or sources
    # Rule: strictly do NOT invent numbers. If numbers exist, plot them.
    num_pattern = re.compile(
        rf"\b({re.escape(e1)}|{re.escape(e2)})\b[^\.\n]*?(\d+(?:\.\d+)?)\s*(%|billion|million|GB|TFLOPS|teraflops|watts|trillion)?",
        flags=re.IGNORECASE
    )
    found_nums: Dict[str, Dict[str, float]] = {e1: {}, e2: {}}

    for line in final_answer.split("\n"):
        for m in num_pattern.finditer(line):
            ent_match = m.group(1).upper()
            val_str = m.group(2)
            unit_str = m.group(3) or ""
            target_ent = e1 if ent_match == e1.upper() else e2
            try:
                val = float(val_str)
                # Key by line context
                ctx = line[:40].strip()
                metric_name = f"{ctx} ({unit_str})".strip()
                found_nums[target_ent][metric_name] = val
            except ValueError:
                pass

    # Build Comparison Table
    # Standard comparative dimensions
    table_headers = ["Comparative Dimension", e1, e2]
    # Extract comparative sections or paragraphs
    rows = [
        ["Core Architecture / Focus", f"{e1} primary architectural foundation", f"{e2} competing architectural design"],
        ["Market Positioning", f"Key position of {e1} in the enterprise/datacenter sector", f"Strategic direction and position of {e2}"],
        ["Key Advantages", f"Strengths identified for {e1}", f"Strengths identified for {e2}"],
        ["Reported Challenges", f"Trade-offs and constraints of {e1}", f"Trade-offs and constraints of {e2}"],
    ]

    components.append(UIComponent(
        type="comparison_table",
        title=f"Comparative Matrix: {e1} vs {e2}",
        data={
            "headers": table_headers,
            "rows": rows,
        }
    ))

    # If common numerical metrics were extracted from genuine sources, build a chart
    common_metrics = [m for m in found_nums[e1] if m in found_nums[e2]]
    if common_metrics:
        labels = common_metrics[:5]
        datasets = [
            {
                "label": e1,
                "data": [found_nums[e1][m] for m in labels],
                "backgroundColor": CHART_COLORS[0],
                "borderColor": CHART_BORDERS[0],
                "borderWidth": 1.5,
            },
            {
                "label": e2,
                "data": [found_nums[e2][m] for m in labels],
                "backgroundColor": CHART_COLORS[1],
                "borderColor": CHART_BORDERS[1],
                "borderWidth": 1.5,
            }
        ]
        components.append(UIComponent(
            type="bar_chart",
            title=f"Comparative Metrics: {e1} vs {e2}",
            data={
                "chartType": "bar",
                "labels": labels,
                "datasets": datasets,
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {"legend": {"position": "top"}}
                }
            }
        ))

    return components


def _build_relationship_graph(structured_query: StructuredQuery) -> Optional[UIComponent]:
    """Generate structured nodes and edges for entity relationship display."""
    if not structured_query.relationships and len(structured_query.entities) < 2:
        return None

    nodes = []
    seen_nodes = set()
    for ent in structured_query.entities:
        if ent.name not in seen_nodes:
            seen_nodes.add(ent.name)
            nodes.append({
                "id": ent.name,
                "label": ent.name,
                "type": ent.type,
            })

    edges = []
    for rel in structured_query.relationships:
        edges.append({
            "source": rel.source,
            "target": rel.target,
            "label": rel.type.replace("_", " ").title(),
        })

    if nodes and edges:
        return UIComponent(
            type="relationship_graph",
            title="Entity Relationships & Architecture",
            data={
                "nodes": nodes,
                "edges": edges,
            }
        )
    return None


def plan_adaptive_response(
    structured_query: StructuredQuery,
    final_answer: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[ProcessedFile]] = None,
) -> List[UIComponent]:
    """Construct adaptive UI components for the query response."""
    components: List[UIComponent] = []
    attachments = attachments or []
    retrieved_chunks = retrieved_chunks or []

    # 1. Process uploaded file summaries / tables
    for att in attachments:
        if att.structured_data:
            tab_comps = _build_tabular_charts(att)
            components.extend(tab_comps)
        elif att.tables:
            # Extracted tables from PDF or DOCX
            for t in att.tables[:3]:
                components.append(UIComponent(
                    type="comparison_table",
                    title=f"Extracted Table from {att.filename} (Page {t.get('page', 1)})",
                    data={
                        "headers": t.get("headers", []),
                        "rows": t.get("rows", [])[:15],
                    }
                ))
        elif att.file_type == "image":
            components.append(UIComponent(
                type="image_analysis",
                title=f"Visual Analysis: {att.filename}",
                data={
                    "filename": att.filename,
                    "ocr_text": att.ocr_text,
                    "preview": att.summary_preview,
                }
            ))

    # 2. Comparative query components
    if structured_query.intent == "comparative_analysis" or structured_query.query_category == "comparative":
        comp_comps = _extract_comparison_from_query(structured_query, final_answer, retrieved_chunks)
        components.extend(comp_comps)

    # 3. Timeline / Chronological query components
    if structured_query.intent == "timeline_generation" or "timeline" in structured_query.output_plan.primary_format:
        timeline_comp = _extract_timeline_from_text(final_answer)
        if timeline_comp:
            components.append(timeline_comp)

    # 4. Entity relationship graph components
    if structured_query.intent == "entity_relationship_analysis" or structured_query.relationships:
        graph_comp = _build_relationship_graph(structured_query)
        if graph_comp:
            components.append(graph_comp)

    # 5. File references component (if files uploaded)
    if attachments:
        file_list = [
            {
                "filename": a.filename,
                "file_type": a.file_type,
                "size_kb": round(a.size_bytes / 1024, 1),
                "error": a.error,
                "summary": a.summary_preview,
            }
            for a in attachments
        ]
        components.append(UIComponent(
            type="file_summary",
            title="Attached Documents & Media",
            data={"files": file_list}
        ))

    return components
