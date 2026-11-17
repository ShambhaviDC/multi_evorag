"""input_understanding_agent.py - Intelligent Input Understanding Agent for EvoRAG Phase 7.

Analyzes incoming user queries and attachments to extract:
1. User intent & query category
2. Named entities & entity types
3. Inter-entity relationships
4. Requested information
5. Temporal constraints & freshness requirements
6. Routing flags (news, document, image, numerical)
7. Adaptive output format plan
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from file_processor import ProcessedFile
from query_models import (
    EntityItem,
    OutputPlan,
    QueryConstraints,
    RelationshipItem,
    StructuredQuery,
)
from utils import call_gemini_async, safe_json_loads

log = logging.getLogger(__name__)

# Known technology and corporate entities for fast deterministic extraction
KNOWN_COMPANIES = {
    "nvidia": "NVIDIA",
    "amd": "AMD",
    "intel": "Intel",
    "microsoft": "Microsoft",
    "google": "Google",
    "deepmind": "DeepMind",
    "openai": "OpenAI",
    "apple": "Apple",
    "meta": "Meta",
    "amazon": "Amazon",
    "tsmc": "TSMC",
    "anthropic": "Anthropic",
    "tesla": "Tesla",
    "ibm": "IBM",
    "oracle": "Oracle",
    "qualcomm": "Qualcomm",
    "arm": "Arm",
    "samsung": "Samsung",
}

KNOWN_TECHNOLOGIES = {
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "llm": "LLM",
    "large language model": "LLM",
    "transformers": "Transformers",
    "transformer": "Transformers",
    "self-attention": "Self-Attention",
    "attention mechanism": "Attention Mechanism",
    "ai safety": "AI Safety",
    "gpu": "GPU",
    "quantum computing": "Quantum Computing",
    "h100": "H100 GPU",
    "b200": "Blackwell B200",
    "mi300": "Instinct MI300",
    "faiss": "FAISS",
    "vector database": "Vector Database",
    "neural network": "Neural Network",
    "reinforcement learning": "Reinforcement Learning",
    "deep learning": "Deep Learning",
    "robotics": "Robotics",
}

KNOWN_RELATIONSHIPS = [
    (("NVIDIA", "AMD"), "competes_with"),
    (("NVIDIA", "Intel"), "competes_with"),
    (("AMD", "Intel"), "competes_with"),
    (("Google", "DeepMind"), "acquired"),
    (("DeepMind", "Google"), "part_of"),
    (("Microsoft", "OpenAI"), "partner_of"),
    (("OpenAI", "Microsoft"), "uses"),
    (("Apple", "Google"), "competes_with"),
    (("Meta", "OpenAI"), "competes_with"),
    (("Anthropic", "OpenAI"), "competes_with"),
]


def _deterministic_entity_extraction(query: str) -> Tuple[List[EntityItem], List[RelationshipItem]]:
    """Fast rule-based entity and relationship extraction without LLM overhead."""
    lower_q = query.lower()
    entities: List[EntityItem] = []
    found_names: set[str] = set()

    # Match known companies
    for key, canonical in KNOWN_COMPANIES.items():
        if re.search(r"\b" + re.escape(key) + r"\b", lower_q):
            entities.append(EntityItem(name=canonical, type="Company"))
            found_names.add(canonical)

    # Match known technologies
    for key, canonical in KNOWN_TECHNOLOGIES.items():
        if re.search(r"\b" + re.escape(key) + r"\b", lower_q):
            entities.append(EntityItem(name=canonical, type="Technology"))
            found_names.add(canonical)

    # Extract capitalized proper names not yet found (e.g. "Sam Altman", "Jensen Huang")
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", query)
    stop_words = {"What", "How", "Why", "When", "Where", "Compare", "Explain", "Analyze", "Show", "Give", "Tell", "Summarize", "Latest", "Recent", "Please", "Can", "Could", "The"}
    for pn in proper_nouns:
        pn_clean = pn.strip()
        if pn_clean not in stop_words and pn_clean not in found_names and len(pn_clean) > 2:
            # Check if it looks like a person or organization
            if " " in pn_clean:
                entities.append(EntityItem(name=pn_clean, type="Person"))
            else:
                entities.append(EntityItem(name=pn_clean, type="Concept"))
            found_names.add(pn_clean)

    # Detect relationships between discovered entities
    relationships: List[RelationshipItem] = []
    names_list = list(found_names)
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            n1, n2 = names_list[i], names_list[j]
            # Check known pairings
            for (p1, p2), rel_type in KNOWN_RELATIONSHIPS:
                if (n1 == p1 and n2 == p2) or (n1 == p2 and n2 == p1):
                    relationships.append(RelationshipItem(source=p1, target=p2, type=rel_type))

    # Detect explicit relationship keywords in query
    if "compet" in lower_q or "vs" in lower_q or "versus" in lower_q:
        if len(entities) >= 2 and not relationships:
            relationships.append(RelationshipItem(
                source=entities[0].name,
                target=entities[1].name,
                type="competes_with"
            ))

    return entities, relationships


def _deterministic_intent_classification(
    query: str,
    attachments: Optional[List[ProcessedFile]] = None
) -> Tuple[str, str, OutputPlan, QueryConstraints, Dict[str, bool], List[str]]:
    """Determine intent, category, output formats, and routing flags via deterministic heuristics."""
    lower_q = query.lower()
    attachments = attachments or []

    # Check attachment types
    has_image = any(a.file_type == "image" for a in attachments)
    has_tabular = any(a.file_type in ("csv", "xlsx") for a in attachments)
    has_doc = any(a.file_type in ("pdf", "docx", "txt") for a in attachments)

    # Freshness / temporal checks
    requires_latest = any(w in lower_q for w in [
        "latest", "recent", "today", "breaking", "current", "newest", "2026", "2025", "this week", "this month"
    ])
    temporal_match = re.search(r"\b(20\d{2}|q[1-4]\s*20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december)\b", lower_q)
    temporal_focus = temporal_match.group(0) if temporal_match else ("latest" if requires_latest else None)

    # Requested information items
    requested_info = []
    if "development" in lower_q or "update" in lower_q:
        requested_info.append("recent developments")
    if "market" in lower_q or "revenue" in lower_q or "share" in lower_q:
        requested_info.append("market position")
    if "spec" in lower_q or "architecture" in lower_q or "benchmark" in lower_q:
        requested_info.append("technical specifications")
    if "challenge" in lower_q or "risk" in lower_q or "limitation" in lower_q:
        requested_info.append("challenges and risks")
    if not requested_info:
        requested_info.append("general explanation")

    # Routing flags
    flags = {
        "requires_news_retrieval": False,
        "requires_document_analysis": has_doc,
        "requires_image_analysis": has_image,
        "requires_numerical_analysis": has_tabular,
    }

    # Intent routing rules
    # 1. Image upload
    if has_image:
        intent = "image_understanding"
        category = "image"
        flags["requires_image_analysis"] = True
        output_plan = OutputPlan(
            primary_format="image_analysis",
            additional_formats=["text_summary", "entity_relationship_diagram"]
        )
    # 2. Spreadsheet / Tabular upload
    elif has_tabular:
        intent = "numerical_analysis"
        category = "numerical"
        flags["requires_numerical_analysis"] = True
        output_plan = OutputPlan(
            primary_format="bar_chart",
            additional_formats=["comparison_table", "numerical_summary", "text_summary"]
        )
    # 3. Document QA (PDF, DOCX)
    elif has_doc:
        intent = "document_qa"
        category = "document"
        flags["requires_document_analysis"] = True
        output_plan = OutputPlan(
            primary_format="document_summary",
            additional_formats=["comparison_table", "text_summary"]
        )
    # 4. Comparative query
    elif any(k in lower_q for k in ["compare", "vs", "versus", "difference between", "better than", "contrast", "comparison"]):
        intent = "comparative_analysis"
        category = "comparative"
        flags["requires_news_retrieval"] = True
        output_plan = OutputPlan(
            primary_format="comparison_table",
            additional_formats=["comparison_chart", "text_summary", "source_list"]
        )
    # 5. Timeline / Chronological query
    elif any(k in lower_q for k in ["timeline", "history", "chronology", "evolution of", "milestones", "over time", "roadmap"]):
        intent = "timeline_generation"
        category = "historical"
        flags["requires_news_retrieval"] = True
        output_plan = OutputPlan(
            primary_format="timeline",
            additional_formats=["text_summary", "source_list"]
        )
    # 6. Trend / Numerical in text
    elif any(k in lower_q for k in ["trend", "growth rate", "percentage", "market share", "statistics", "metrics", "chart"]):
        intent = "trend_analysis"
        category = "numerical"
        flags["requires_news_retrieval"] = True
        flags["requires_numerical_analysis"] = True
        output_plan = OutputPlan(
            primary_format="bar_chart",
            additional_formats=["comparison_table", "numerical_summary", "text_summary"]
        )
    # 7. Relationship analysis
    elif any(k in lower_q for k in ["relationship between", "how do .* relate", "compete with", "network", "ecosystem"]):
        intent = "entity_relationship_analysis"
        category = "relational"
        flags["requires_news_retrieval"] = True
        output_plan = OutputPlan(
            primary_format="relationship_graph",
            additional_formats=["comparison_table", "text_summary"]
        )
    # 8. Latest news
    elif requires_latest or "news" in lower_q:
        intent = "latest_news_retrieval"
        category = "news"
        flags["requires_news_retrieval"] = True
        output_plan = OutputPlan(
            primary_format="text_summary",
            additional_formats=["source_list", "comparison_table"]
        )
    # 9. Default: Factual QA
    else:
        intent = "factual_question_answering"
        category = "factual"
        flags["requires_news_retrieval"] = True
        output_plan = OutputPlan(
            primary_format="text_summary",
            additional_formats=["source_list"]
        )

    constraints = QueryConstraints(
        requires_latest_information=requires_latest,
        requires_source_citations=True,
        temporal_focus=temporal_focus,
    )

    return intent, category, output_plan, constraints, flags, requested_info


async def _llm_refine_understanding(
    query: str,
    preliminary_struct: StructuredQuery,
    attachments: Optional[List[ProcessedFile]] = None
) -> StructuredQuery:
    """Invoke LLM structured extraction when query is complex, ambiguous, or multi-faceted."""
    prompt = f"""You are the EvoRAG Input Understanding Agent. Analyze the user's query and return a valid JSON object.

User Query: "{query}"

Detected Preliminary Entities: {[e.name for e in preliminary_struct.entities]}
Has Attachments: {bool(attachments)}

Strict Output JSON Schema:
{{
  "original_query": "{query}",
  "intent": "comparative_analysis | latest_news_retrieval | factual_question_answering | timeline_generation | numerical_analysis | document_qa | image_understanding | entity_relationship_analysis | explanation | summarization",
  "query_category": "comparative | news | factual | historical | numerical | document | image | relational",
  "entities": [
    {{"name": "...", "type": "Company | Technology | Person | Organization | Event | Location | Concept | Metric | Product"}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "competes_with | acquired | works_for | located_in | uses | part_of | occurred_at | related_to"}}
  ],
  "requested_information": ["..."],
  "constraints": {{
    "requires_latest_information": true | false,
    "requires_source_citations": true,
    "temporal_focus": "string or null"
  }},
  "output_plan": {{
    "primary_format": "comparison_table | comparison_chart | bar_chart | line_chart | timeline | numerical_summary | relationship_graph | text_summary | image_analysis",
    "additional_formats": ["..."]
  }}
}}

RULES:
- Return ONLY valid JSON.
- If an entity or relationship is uncertain, omit it. Do NOT hallucinate.
- Preserve original_query exactly.
"""
    try:
        raw_resp = await call_gemini_async(
            prompt,
            max_output_tokens=700,
            temperature=0.1,
            timeout_seconds=5,
            max_attempts=1
        )
        if raw_resp:
            clean_json = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip())
            clean_json = re.sub(r"\s*```$", "", clean_json)
            parsed = json.loads(clean_json)

            entities = [
                EntityItem(name=e.get("name", ""), type=e.get("type", "Concept"), attributes=e.get("attributes"))
                for e in parsed.get("entities", [])
                if e.get("name")
            ]
            relationships = [
                RelationshipItem(source=r.get("source", ""), target=r.get("target", ""), type=r.get("type", "related_to"))
                for r in parsed.get("relationships", [])
                if r.get("source") and r.get("target")
            ]
            constraints_data = parsed.get("constraints", {})
            constraints = QueryConstraints(
                requires_latest_information=bool(constraints_data.get("requires_latest_information")),
                requires_source_citations=True,
                temporal_focus=constraints_data.get("temporal_focus"),
            )
            output_plan_data = parsed.get("output_plan", {})
            output_plan = OutputPlan(
                primary_format=output_plan_data.get("primary_format", preliminary_struct.output_plan.primary_format),
                additional_formats=output_plan_data.get("additional_formats", preliminary_struct.output_plan.additional_formats),
            )

            return StructuredQuery(
                original_query=query,
                intent=parsed.get("intent", preliminary_struct.intent),
                query_category=parsed.get("query_category", preliminary_struct.query_category),
                entities=entities or preliminary_struct.entities,
                relationships=relationships or preliminary_struct.relationships,
                requested_information=parsed.get("requested_information") or preliminary_struct.requested_information,
                constraints=constraints,
                output_plan=output_plan,
                requires_news_retrieval=preliminary_struct.requires_news_retrieval,
                requires_document_analysis=preliminary_struct.requires_document_analysis,
                requires_image_analysis=preliminary_struct.requires_image_analysis,
                requires_numerical_analysis=preliminary_struct.requires_numerical_analysis,
                attachment_filenames=preliminary_struct.attachment_filenames,
            )
    except Exception as exc:
        log.warning("LLM refinement failed (%s) — using deterministic structured query", exc)

    return preliminary_struct


async def understand_user_query(
    query: str,
    attachments: Optional[List[ProcessedFile]] = None,
    force_llm_refinement: bool = False
) -> StructuredQuery:
    """Main entrypoint: parses query and attachments into a validated StructuredQuery."""
    attachments = attachments or []
    attachment_filenames = [a.filename for a in attachments]

    # Step 1: Fast deterministic extraction
    intent, category, output_plan, constraints, flags, requested_info = _deterministic_intent_classification(
        query, attachments
    )
    entities, relationships = _deterministic_entity_extraction(query)

    preliminary = StructuredQuery(
        original_query=query,
        intent=intent,
        query_category=category,
        entities=entities,
        relationships=relationships,
        requested_information=requested_info,
        constraints=constraints,
        output_plan=output_plan,
        requires_news_retrieval=flags["requires_news_retrieval"],
        requires_document_analysis=flags["requires_document_analysis"],
        requires_image_analysis=flags["requires_image_analysis"],
        requires_numerical_analysis=flags["requires_numerical_analysis"],
        attachment_filenames=attachment_filenames,
    )

    # Step 2: Use LLM refinement only if requested or if query is complex without obvious entities
    # (CPU optimization: skip LLM call for straightforward queries with clear entities or attachments)
    should_refine = (
        force_llm_refinement
        or (len(query.split()) > 10 and not entities and not attachments)
        or ("relation" in query.lower() and not relationships)
    )

    if should_refine:
        return await _llm_refine_understanding(query, preliminary, attachments)

    return preliminary
