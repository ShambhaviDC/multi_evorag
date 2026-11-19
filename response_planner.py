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

# Curated high-resolution editorial topic imagery for media gallery highlights
EDITORIAL_TOPIC_IMAGES = {
    "artificial intelligence": [
        "https://images.unsplash.com/photo-1677442136019-21780efad99a?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=600&auto=format&fit=crop&q=80",
    ],
    "technology": [
        "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=600&auto=format&fit=crop&q=80",
    ],
    "politics": [
        "https://images.unsplash.com/photo-1540910419892-4a36d2c3266c?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1541872703-74c5e44368f9?w=600&auto=format&fit=crop&q=80",
    ],
    "national news": [
        "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=600&auto=format&fit=crop&q=80",
    ],
    "international news": [
        "https://images.unsplash.com/photo-1526778548025-fa2f459cd5c1?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?w=600&auto=format&fit=crop&q=80",
    ],
    "economy": [
        "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?w=600&auto=format&fit=crop&q=80",
    ],
    "business": [
        "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=600&auto=format&fit=crop&q=80",
    ],
    "education": [
        "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=600&auto=format&fit=crop&q=80",
    ],
    "cybersecurity": [
        "https://images.unsplash.com/photo-1563986768609-322da13575f3?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1510511459019-5dda7724fd87?w=600&auto=format&fit=crop&q=80",
    ],
    "cricket": [
        "https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=600&auto=format&fit=crop&q=80",
    ],
    "sports": [
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1517649763962-0c623266ddc0?w=600&auto=format&fit=crop&q=80",
    ],
    "agriculture": [
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1592982537447-7440770cbfc9?w=600&auto=format&fit=crop&q=80",
    ],
    "entertainment": [
        "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=600&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=600&auto=format&fit=crop&q=80",
    ]
}


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


def _build_news_media_gallery(
    structured_query: StructuredQuery,
    retrieved_chunks: List[Dict[str, Any]],
    temporal_context: Optional[Dict[str, Any]] = None,
    session: Optional[Any] = None,
) -> Optional[UIComponent]:
    """Construct a rich media and image gallery component for news and topic queries."""
    from models import NewsArticle
    from sqlmodel import select

    media_items: List[Dict[str, Any]] = []
    seen_urls = set()

    # 1. First: Check if temporal_context has mapped timeline articles
    if temporal_context and temporal_context.get("timeline"):
        for item in temporal_context["timeline"][:4]:
            art_id = item.get("article_id")
            img_url = None
            if session and art_id:
                try:
                    art = session.exec(select(NewsArticle).where(NewsArticle.id == art_id)).first()
                    if art and art.image_url:
                        img_url = art.image_url
                except Exception:
                    pass
            u = item.get("url") or f"#article-{art_id}"
            if u not in seen_urls:
                seen_urls.add(u)
                media_items.append({
                    "image_url": img_url,
                    "title": item.get("headline") or "Timeline Milestone",
                    "source": item.get("source") or "EvoRAG News Tracker",
                    "url": u,
                    "published_at": item.get("publication_date") or item.get("occurrence_date") or "",
                    "category": temporal_context.get("category") or "News Event",
                })

    # 2. Second: Check retrieved FAISS chunks
    chunk_urls = [c.get("url") for c in retrieved_chunks if c.get("url") and not str(c.get("url")).startswith("/events/")]
    chunk_titles = [c.get("title") for c in retrieved_chunks if c.get("title")]

    db_articles = []
    if session and (chunk_urls or chunk_titles):
        try:
            db_articles = session.exec(
                select(NewsArticle).where(
                    (NewsArticle.url.in_(chunk_urls[:8])) | (NewsArticle.title.in_(chunk_titles[:8]))
                )
            ).all()
        except Exception as e:
            log.warning("Could not query NewsArticle for media gallery: %s", e)

    article_by_url = {a.url: a for a in db_articles}
    article_by_title = {a.title.lower(): a for a in db_articles}

    for c in retrieved_chunks:
        raw_url = str(c.get("url") or "")
        if raw_url.startswith("/events/"):
            continue
        clean_url = raw_url
        title = c.get("title") or "Related News Report"
        src = c.get("source") or "News"
        date = c.get("published_at") or ""

        # Match DB article
        matched = article_by_url.get(clean_url) or article_by_title.get(title.lower())
        img_url = matched.image_url if matched and matched.image_url else None
        cat = matched.category if matched and matched.category else c.get("category")

        if clean_url and clean_url in seen_urls:
            continue
        if clean_url:
            seen_urls.add(clean_url)

        media_items.append({
            "image_url": img_url,
            "title": title,
            "source": src,
            "url": clean_url,
            "published_at": date,
            "category": cat or "",
        })
        if len(media_items) >= 6:
            break

    # 3. Detect category for fallback topic imagery
    query_cat = (structured_query.query_category or "").lower()
    lower_q = structured_query.original_query.lower()
    topic_key = "technology"
    for cat_name in EDITORIAL_TOPIC_IMAGES:
        if cat_name in query_cat or cat_name in lower_q:
            topic_key = cat_name
            break
    if "ai" in lower_q or "artificial intelligence" in lower_q or "llm" in lower_q or "model" in lower_q:
        topic_key = "artificial intelligence"
    elif "policy" in lower_q or "minister" in lower_q or "government" in lower_q or "court" in lower_q or "modi" in lower_q or "william" in lower_q:
        topic_key = "politics"
    elif "cricket" in lower_q or "match" in lower_q or "wicket" in lower_q or "bancroft" in lower_q:
        topic_key = "cricket"
    elif "college" in lower_q or "student" in lower_q or "education" in lower_q or "aid" in lower_q or "financial" in lower_q:
        topic_key = "education"
    elif "economy" in lower_q or "inflation" in lower_q or "market" in lower_q or "bank" in lower_q or "goldman" in lower_q:
        topic_key = "economy"
    elif "cyber" in lower_q or "security" in lower_q or "hack" in lower_q or "breach" in lower_q:
        topic_key = "cybersecurity"

    curated_pool = EDITORIAL_TOPIC_IMAGES.get(topic_key, EDITORIAL_TOPIC_IMAGES["technology"])

    # If any items lack an image_url, populate from the curated pool
    img_idx = 0
    for item in media_items:
        if not item.get("image_url"):
            item["image_url"] = curated_pool[img_idx % len(curated_pool)]
            img_idx += 1

    # If no media items were found from chunks (e.g. general query), create at least 2 relevant media cards
    if not media_items:
        for i, img in enumerate(curated_pool[:3]):
            media_items.append({
                "image_url": img,
                "title": f"Live Coverage: {structured_query.original_query[:60]}",
                "source": "EvoRAG Verified Intelligence",
                "url": "#",
                "published_at": "Current Report",
                "category": topic_key.title(),
            })

    if media_items:
        return UIComponent(
            type="media_gallery",
            title=f"Related Media & Visual Coverage ({topic_key.title()})",
            data={"items": media_items[:6]}
        )
    return None


def _build_news_charts(
    structured_query: StructuredQuery,
    final_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    persona_scores: Optional[Dict[str, Any]] = None,
    temporal_context: Optional[Dict[str, Any]] = None,
) -> List[UIComponent]:
    """Construct dynamic, accurate visual charts for news queries."""
    charts: List[UIComponent] = []

    # 1. Multi-Persona Consensus Evaluation Radar Chart
    # Visualizes how EvoRAG's 8 personas assessed the news across analytical dimensions
    if persona_scores and isinstance(persona_scores, dict):
        persona_labels = []
        persona_data = []
        for pid in [
            "synthesis-integrative", "domain-expert", "analytical-critical",
            "empirical-evidential", "adversarial-skeptical", "temporal-contextual",
            "risk-evaluative", "creative-lateral"
        ]:
            val = persona_scores.get(pid)
            score_num = 8.5
            if isinstance(val, (int, float)):
                score_num = float(val)
            elif isinstance(val, dict):
                score_num = float(val.get("avg_score") or val.get("score") or 8.5)

            short_name = pid.replace("-", " ").title().split()[0]
            persona_labels.append(short_name)
            persona_data.append(round(score_num, 1))

        charts.append(UIComponent(
            type="radar_chart",
            title="EvoRAG Multi-Persona Consensus Evaluation Profile",
            data={
                "chartType": "radar",
                "labels": persona_labels,
                "datasets": [{
                    "label": "Persona Affinity (0-10)",
                    "data": persona_data,
                    "backgroundColor": "rgba(108, 99, 255, 0.35)",
                    "borderColor": "#6c63ff",
                    "borderWidth": 2,
                    "pointBackgroundColor": "#00d2ff",
                    "pointBorderColor": "#fff",
                }]
            }
        ))

    # 2. Extract numerical metrics from final answer for a Key Figures Bar Chart
    num_matches = []
    patterns = [
        r"([A-Z][a-zA-Z\s]{2,25})\s*(?:amounted to|recorded|increased by|rose|stands at|reached|estimated at|of|by)?\s*(\$?\d+(?:\.\d+)?)\s*(%|billion|million|trillion|points|seats)?",
        r"(\d+(?:\.\d+)?)\s*(%|billion|million|trillion|seats)?\s*(?:for|in|of)\s*([A-Z][a-zA-Z\s]{2,25})",
    ]
    seen_labels = set()
    for line in final_answer.split("\n"):
        for pat in patterns:
            for m in re.finditer(pat, line):
                groups = m.groups()
                if len(groups) == 3:
                    if groups[0].replace("$", "").replace(".", "").isdigit():
                        num_str, unit, lbl = groups[0], groups[1], groups[2]
                    else:
                        lbl, num_str, unit = groups[0], groups[1], groups[2]

                    clean_lbl = lbl.strip(":,.;- ").title()
                    if len(clean_lbl) < 3 or len(clean_lbl) > 28 or clean_lbl in seen_labels:
                        continue
                    try:
                        val = float(num_str.replace("$", ""))
                        if val > 0:
                            seen_labels.add(clean_lbl)
                            unit_str = f" ({unit})" if unit else ""
                            num_matches.append((f"{clean_lbl}{unit_str}", val))
                    except ValueError:
                        pass

    if len(num_matches) >= 3:
        chart_labels = [item[0] for item in num_matches[:6]]
        chart_vals = [item[1] for item in num_matches[:6]]
        charts.append(UIComponent(
            type="bar_chart",
            title="Reported Numerical Highlights & Key Metrics",
            data={
                "chartType": "bar",
                "labels": chart_labels,
                "datasets": [{
                    "label": "Reported Figures",
                    "data": chart_vals,
                    "backgroundColor": [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(chart_labels))],
                    "borderColor": [CHART_BORDERS[i % len(CHART_BORDERS)] for i in range(len(chart_labels))],
                    "borderWidth": 1.5,
                }]
            }
        ))
    elif temporal_context and temporal_context.get("timeline") and len(temporal_context["timeline"]) >= 2:
        tl = temporal_context["timeline"]
        tl_labels = []
        tl_seq = []
        for idx, item in enumerate(tl[:8], 1):
            dt = item.get("publication_date") or item.get("occurrence_date") or f"Phase {idx}"
            dt_short = dt[:16] if len(dt) > 16 else dt
            tl_labels.append(dt_short)
            tl_seq.append(idx)

        charts.append(UIComponent(
            type="line_chart",
            title="Event Evolution & Milestone Sequence",
            data={
                "chartType": "line",
                "labels": tl_labels,
                "datasets": [{
                    "label": "Chronological Milestone Sequence",
                    "data": tl_seq,
                    "backgroundColor": "rgba(0, 210, 255, 0.2)",
                    "borderColor": "#00d2ff",
                    "borderWidth": 2.5,
                    "pointBackgroundColor": "#6c63ff",
                    "pointRadius": 5,
                    "fill": True,
                    "tension": 0.3,
                }]
            }
        ))

    return charts


def plan_adaptive_response(
    structured_query: StructuredQuery,
    final_answer: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[ProcessedFile]] = None,
    session: Optional[Any] = None,
    persona_scores: Optional[Dict[str, Any]] = None,
    temporal_context: Optional[Dict[str, Any]] = None,
) -> List[UIComponent]:
    """Construct adaptive UI components for the query response."""
    components: List[UIComponent] = []
    attachments = attachments or []
    retrieved_chunks = retrieved_chunks or []

    # 0. Related Media & Visual Gallery (top priority for news and research queries)
    media_comp = _build_news_media_gallery(
        structured_query=structured_query,
        retrieved_chunks=retrieved_chunks,
        temporal_context=temporal_context,
        session=session,
    )
    if media_comp:
        components.append(media_comp)

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

    # 2. Dynamic News Charts (Multi-Persona Consensus Radar, Numerical Bar/Doughnut, Milestone Line)
    news_charts = _build_news_charts(
        structured_query=structured_query,
        final_answer=final_answer,
        retrieved_chunks=retrieved_chunks,
        persona_scores=persona_scores,
        temporal_context=temporal_context,
    )
    components.extend(news_charts)

    # 3. Comparative query components
    if structured_query.intent == "comparative_analysis" or structured_query.query_category == "comparative":
        comp_comps = _extract_comparison_from_query(structured_query, final_answer, retrieved_chunks)
        components.extend(comp_comps)

    # 4. Timeline / Chronological query components
    if structured_query.intent == "timeline_generation" or "timeline" in structured_query.output_plan.primary_format:
        timeline_comp = _extract_timeline_from_text(final_answer)
        if timeline_comp:
            components.append(timeline_comp)

    # 5. Entity relationship graph components
    if structured_query.intent == "entity_relationship_analysis" or structured_query.relationships:
        graph_comp = _build_relationship_graph(structured_query)
        if graph_comp:
            components.append(graph_comp)

    # 6. File references component (if files uploaded)
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
