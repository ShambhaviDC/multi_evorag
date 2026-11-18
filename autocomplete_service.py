"""
autocomplete_service.py — Smart Context-Aware Autocomplete and Topic Suggestion Engine.

Features:
  - Prefix matching on categories, subcategories, key entities, and stored article headlines
  - Context-aware ranking influenced by user preferences without suppressing general discovery
  - Token-level replacement detection to preserve preceding and succeeding query text
  - Sub-millisecond in-memory prefix indexing + SQLite article title lookups
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlmodel import Session, select

from models import NewsArticle, PersonalizationProfile, UserPreference
from news_service import ALL_NEWS_CATEGORIES, NEWS_TAXONOMY

log = logging.getLogger(__name__)

# Predefined high-frequency search entities and topic expansions
CURATED_SUGGESTIONS: List[Dict[str, str]] = [
    # Cricket & Sports
    {"text": "Cricket", "category": "Cricket", "type": "category"},
    {"text": "Cricket World Cup", "category": "Cricket", "type": "trending"},
    {"text": "Cricket News", "category": "Cricket", "type": "news"},
    {"text": "Cricket Rankings", "category": "Cricket", "type": "topic"},
    {"text": "Cricket Developments", "category": "Cricket", "type": "topic"},
    {"text": "MS Dhoni", "category": "Cricket", "type": "entity"},
    {"text": "Dhoni IPL Performance", "category": "Cricket", "type": "topic"},
    {"text": "Dhoni Latest News", "category": "Cricket", "type": "trending"},
    {"text": "Dhoni Career Developments", "category": "Cricket", "type": "topic"},
    {"text": "Virat Kohli", "category": "Cricket", "type": "entity"},
    {"text": "Rohit Sharma", "category": "Cricket", "type": "entity"},
    {"text": "Jasprit Bumrah", "category": "Cricket", "type": "entity"},
    {"text": "IPL", "category": "Cricket", "type": "category"},
    {"text": "IPL Developments", "category": "Cricket", "type": "topic"},
    {"text": "IPL Auction", "category": "Cricket", "type": "trending"},
    {"text": "IPL Team Updates", "category": "Cricket", "type": "topic"},
    {"text": "IPL News", "category": "Cricket", "type": "news"},
    {"text": "IPL Player Developments", "category": "Cricket", "type": "topic"},
    {"text": "Indian Cricket Team", "category": "Cricket", "type": "entity"},
    {"text": "BCCI", "category": "Cricket", "type": "entity"},
    {"text": "Sports News", "category": "Sports", "type": "news"},
    {"text": "Football Champions League", "category": "Sports", "type": "topic"},
    {"text": "Olympics 2028", "category": "Sports", "type": "topic"},
    {"text": "Formula 1 Grand Prix", "category": "Sports", "type": "topic"},
    {"text": "Badminton World Tour", "category": "Sports", "type": "topic"},

    # Artificial Intelligence & Technology
    {"text": "Artificial Intelligence", "category": "Artificial Intelligence", "type": "category"},
    {"text": "AI Developments", "category": "Artificial Intelligence", "type": "trending"},
    {"text": "AI Research", "category": "Artificial Intelligence", "type": "topic"},
    {"text": "AI Startups", "category": "Artificial Intelligence", "type": "topic"},
    {"text": "AI Regulations", "category": "Artificial Intelligence", "type": "topic"},
    {"text": "AI Safety and Governance", "category": "Artificial Intelligence", "type": "topic"},
    {"text": "AI Agents and Reasoning", "category": "Artificial Intelligence", "type": "topic"},
    {"text": "Technology", "category": "Technology", "type": "category"},
    {"text": "Tech Developments", "category": "Technology", "type": "topic"},
    {"text": "Tech News", "category": "Technology", "type": "news"},
    {"text": "Tech Startups", "category": "Technology", "type": "topic"},
    {"text": "Semiconductor Chips", "category": "Technology", "type": "topic"},
    {"text": "Generative AI", "category": "Artificial Intelligence", "type": "subcategory"},
    {"text": "OpenAI", "category": "Artificial Intelligence", "type": "entity"},
    {"text": "Google DeepMind Gemini", "category": "Artificial Intelligence", "type": "entity"},
    {"text": "Anthropic Claude", "category": "Artificial Intelligence", "type": "entity"},
    {"text": "NVIDIA AI Chips", "category": "Technology", "type": "entity"},
    {"text": "AMD Processors", "category": "Technology", "type": "entity"},
    {"text": "Cybersecurity", "category": "Cybersecurity", "type": "category"},
    {"text": "Ransomware Attacks", "category": "Cybersecurity", "type": "topic"},
    {"text": "Data Breach Vulnerabilities", "category": "Cybersecurity", "type": "topic"},

    # Agriculture
    {"text": "Agriculture", "category": "Agriculture", "type": "category"},
    {"text": "Agricultural Technology", "category": "Agriculture", "type": "subcategory"},
    {"text": "Agricultural Schemes", "category": "Agriculture", "type": "topic"},
    {"text": "Agritech Startups", "category": "Agriculture", "type": "topic"},
    {"text": "Farming Developments", "category": "Agriculture", "type": "topic"},
    {"text": "Crop Developments & Yield", "category": "Agriculture", "type": "topic"},
    {"text": "Monsoon & Irrigation Impact", "category": "Agriculture", "type": "topic"},
    {"text": "Rural Economy & MSP", "category": "Agriculture", "type": "topic"},

    # Business, Economy, Startups
    {"text": "Business", "category": "Business", "type": "category"},
    {"text": "Economy", "category": "Economy", "type": "category"},
    {"text": "Inflation and Interest Rates", "category": "Economy", "type": "topic"},
    {"text": "GDP Growth Forecast", "category": "Economy", "type": "topic"},
    {"text": "Stock Market Trends", "category": "Business", "type": "topic"},
    {"text": "Startups", "category": "Startups", "type": "category"},
    {"text": "Venture Capital Funding", "category": "Startups", "type": "topic"},
    {"text": "Unicorn Startups", "category": "Startups", "type": "topic"},
    {"text": "Employment & Hiring Trends", "category": "Employment", "type": "topic"},
    {"text": "Job Market & Layoffs", "category": "Employment", "type": "topic"},

    # Science, Environment, Healthcare, Politics
    {"text": "Science", "category": "Science", "type": "category"},
    {"text": "Space Exploration & NASA", "category": "Science", "type": "topic"},
    {"text": "ISRO Space Missions", "category": "Science", "type": "topic"},
    {"text": "Quantum Computing", "category": "Science", "type": "topic"},
    {"text": "Environment", "category": "Environment", "type": "category"},
    {"text": "Climate Change Solutions", "category": "Environment", "type": "topic"},
    {"text": "Renewable Energy & Solar", "category": "Environment", "type": "topic"},
    {"text": "Healthcare", "category": "Healthcare", "type": "category"},
    {"text": "Medical Research Breakthroughs", "category": "Healthcare", "type": "topic"},
    {"text": "Vaccine Clinical Trials", "category": "Healthcare", "type": "topic"},
    {"text": "Politics", "category": "Politics", "type": "category"},
    {"text": "Indian Politics", "category": "Politics", "type": "subcategory"},
    {"text": "Election Developments", "category": "Politics", "type": "topic"},
    {"text": "Government Policy", "category": "National News", "type": "topic"},
    {"text": "International Relations", "category": "International News", "type": "topic"},
    {"text": "Education & Universities", "category": "Education", "type": "topic"},
    {"text": "Entertainment & Box Office", "category": "Entertainment", "type": "topic"},
]

# Build in-memory fast index from taxonomy
for cat, subs in NEWS_TAXONOMY.items():
    if not any(s["text"].lower() == cat.lower() for s in CURATED_SUGGESTIONS):
        CURATED_SUGGESTIONS.append({"text": cat, "category": cat, "type": "category"})
    for sub in subs:
        if not any(s["text"].lower() == sub.lower() for s in CURATED_SUGGESTIONS):
            CURATED_SUGGESTIONS.append({"text": sub, "category": cat, "type": "subcategory"})


def _extract_active_token(query: str) -> Tuple[str, int, int]:
    """
    Extract the trailing active word currently being typed by user.
    Returns: (token_str, start_index, end_index)
    """
    if not query:
        return "", 0, 0

    # Match trailing single word being typed (e.g. "cri" at end of sentence)
    match_single = re.search(r"([a-zA-Z0-9_\-]+)\s*$", query)
    if match_single:
        token = match_single.group(1).strip()
        start = match_single.start(1)
        end = match_single.end(1)
        return token, start, end

    stripped = query.strip()
    return stripped, 0, len(stripped)


def get_autocomplete_suggestions(
    query_text: str,
    user_id: Optional[int] = None,
    session: Optional[Session] = None,
    max_suggestions: int = 7,
) -> Dict[str, Any]:
    """
    Compute smart autocomplete suggestions based on query text.
    Combines:
      - Prefix matches on taxonomy & entities
      - Stored article headlines matching query
      - User preference weighting (promotes preferred domains without hiding others)
      - Precise replacement boundaries for seamless text insertion
    """
    token, start_idx, end_idx = _extract_active_token(query_text)
    if not token or len(token) < 2:
        return {
            "query": query_text,
            "token": token,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "suggestions": [],
        }

    token_lower = token.lower()
    results: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()

    # User preferred categories
    user_cats: Set[str] = set()
    if user_id and session:
        try:
            profile = session.exec(
                select(PersonalizationProfile).where(PersonalizationProfile.user_id == user_id)
            ).first()
            if profile:
                import json
                user_cats = {c.lower() for c in json.loads(profile.preferred_categories_json or "[]")}
        except Exception:
            pass

    # 1. Search curated suggestions & taxonomy
    for item in CURATED_SUGGESTIONS:
        cand_text = item["text"]
        cand_lower = cand_text.lower()

        # Prefix match or word boundary match
        is_prefix = cand_lower.startswith(token_lower)
        is_word_match = any(w.startswith(token_lower) for w in cand_lower.split())

        if is_prefix or is_word_match:
            if cand_lower not in seen_texts:
                seen_texts.add(cand_lower)

                # Relevance score
                score = 100 if is_prefix else 70
                # Extra boost if exact starting acronym (e.g. "IPL", "AI")
                if cand_lower == token_lower:
                    score += 50
                # Personalization boost
                if item["category"].lower() in user_cats:
                    score += 25

                results.append({
                    "text": cand_text,
                    "category": item["category"],
                    "type": item["type"],
                    "score": score,
                })

    # 2. Search stored news article headlines from SQLite (if session provided)
    if session and len(results) < max_suggestions * 2:
        try:
            like_pattern = f"%{token}%"
            articles = session.exec(
                select(NewsArticle)
                .where(NewsArticle.title.ilike(like_pattern))
                .order_by(NewsArticle.created_at.desc())
                .limit(8)
            ).all()

            for art in articles:
                t = art.title.strip()
                t_lower = t.lower()
                if t_lower not in seen_texts:
                    seen_texts.add(t_lower)
                    # Shorten long headlines for autocomplete readability
                    words = t.split()
                    short_t = " ".join(words[:7]) + ("..." if len(words) > 7 else "")
                    score = 40
                    if art.category.lower() in user_cats:
                        score += 20
                    results.append({
                        "text": short_t,
                        "category": art.category,
                        "type": "headline",
                        "score": score,
                    })
        except Exception as e:
            log.debug("Article headline lookup error: %s", e)

    # Sort results by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:max_suggestions]

    # Clean out internal score field before returning
    for r in top_results:
        r.pop("score", None)

    return {
        "query": query_text,
        "token": token,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "suggestions": top_results,
    }


def apply_autocomplete_token(query: str, suggestion: str, start_idx: int, end_idx: int) -> str:
    """Helper to replace active token in a query with suggestion preserving the rest of the sentence."""
    before = query[:start_idx]
    after = query[end_idx:]
    return f"{before}{suggestion}{after}"

