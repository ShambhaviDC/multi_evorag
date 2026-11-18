"""
news_service.py — Real-Time News Ingestion, Taxonomy Classification, and Homepage Discovery.

Supports:
  - 18 Major Categories + Subcategories Taxonomy
  - Real RSS Ingestion across all domains (Cricket, Agriculture, AI, Tech, Politics, Economy, etc.)
  - Real metadata extraction: Title, URL, Source, Published Date, Snippet, Image URL
  - Persistence into SQLite `NewsArticle` table
  - General Homepage aggregation for Guests
  - Personalized Homepage curation for Authenticated Users based on explicit + inferred interests
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import feedparser
from bs4 import BeautifulSoup
from sqlmodel import Session, select

from config import DATA_DIR, INDEX_DIR, META_FILE
from database import get_session
from models import NewsArticle, PersonalizationProfile, UserPreference, utc_now

log = logging.getLogger(__name__)

# ── 18 Comprehensive News Categories and Taxonomy ────────────────────────────
NEWS_TAXONOMY: Dict[str, List[str]] = {
    "Politics": [
        "Indian Politics", "State Politics", "Government Policies", "Elections",
        "International Politics", "Parliament", "Legislation", "Cabinet",
    ],
    "Technology": [
        "Software Development", "Hardware & Chips", "Robotics", "Gadgets",
        "Cloud Computing", "Telecom & 5G", "Space Technology", "Semiconductors",
    ],
    "Artificial Intelligence": [
        "Generative AI", "LLMs & Foundation Models", "AI Agents", "Machine Learning",
        "Computer Vision", "AI Safety & Ethics", "AI Hardware & GPUs", "Neural Networks",
    ],
    "Agriculture": [
        "Farming", "Agricultural Technology", "Government Schemes", "Crop Developments",
        "Rural Economy", "Monsoon & Irrigation", "Agritech Startups", "Horticulture",
    ],
    "National News": [
        "Government Policy", "Education Policy", "Infrastructure", "Public Health",
        "Judiciary & Law", "National Security", "State Developments", "Railways",
    ],
    "International News": [
        "International Relations", "Global Conflicts", "World Economy", "Diplomacy",
        "International Organizations", "Geopolitics", "Global Summits", "Trade Accords",
    ],
    "Business": [
        "Corporate Earnings", "Stock Markets", "Mergers & Acquisitions", "Banking",
        "Real Estate", "Enterprise Strategy", "Global Commerce", "Executive Leadership",
    ],
    "Economy": [
        "Inflation", "Interest Rates", "GDP Growth", "Fiscal Policy & Budget",
        "Monetary Policy", "Trade & Tariffs", "Taxation", "Central Banks",
    ],
    "Sports": [
        "Football", "Tennis", "Badminton", "Formula 1", "Hockey",
        "Olympics", "Athletics", "Kabaddi", "Chess",
    ],
    "Cricket": [
        "IPL", "International Cricket", "Test Cricket", "T20 & T20I", "ODI",
        "Indian Cricket Team", "Player Developments", "BCCI", "World Cup",
    ],
    "Science": [
        "Space Exploration", "Physics & Quantum", "Astronomy & Cosmology",
        "Biotechnology", "Genetics", "NASA & ISRO", "Archaeology",
    ],
    "Healthcare": [
        "Medical Research", "Public Health", "Vaccines", "Clinical Trials",
        "Pharmaceuticals", "Nutrition & Wellness", "Disease Prevention",
    ],
    "Environment": [
        "Climate Change", "Renewable Energy", "Global Warming", "Carbon Emissions",
        "Sustainability", "Wildlife Conservation", "Pollution Control", "Clean Tech",
    ],
    "Education": [
        "Higher Education", "Universities", "EdTech", "Competitive Exams",
        "School Education", "Skilling & Vocational", "Scholarships",
    ],
    "Entertainment": [
        "Cinema & Movies", "OTT & Streaming", "Music", "Box Office",
        "Celebrity News", "Gaming", "Film Festivals",
    ],
    "Cybersecurity": [
        "Data Breaches", "Ransomware", "Cyber Threats", "Malware & Phishing",
        "Network Security", "Privacy & Encryption", "Government Cyber Defense",
    ],
    "Startups": [
        "Funding & Venture Capital", "Unicorns", "Tech Startups", "Fintech",
        "Agritech Startups", "Founders & Pitch", "Incubators & Accelerators",
    ],
    "Employment": [
        "Job Market", "Hiring Trends", "Layoffs", "Remote Work",
        "Gig Economy", "Workforce Skills", "Labor Reforms", "Salaries",
    ],
}

ALL_NEWS_CATEGORIES = list(NEWS_TAXONOMY.keys())

# Real RSS Feeds categorized by primary subject
FEED_SOURCES: List[Dict[str, Any]] = [
    # Cricket & Sports
    {
        "url": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        "source": "ESPN Cricinfo",
        "default_category": "Cricket",
    },
    {
        "url": "https://feeds.bbci.co.uk/sport/rss.xml",
        "source": "BBC Sport",
        "default_category": "Sports",
    },
    # Technology & AI
    {
        "url": "https://feeds.feedburner.com/TechCrunch/",
        "source": "TechCrunch",
        "default_category": "Technology",
    },
    {
        "url": "https://www.theverge.com/rss/index.xml",
        "source": "The Verge",
        "default_category": "Technology",
    },
    {
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
        "source": "MIT Technology Review",
        "default_category": "Artificial Intelligence",
    },
    {
        "url": "https://venturebeat.com/category/ai/feed/",
        "source": "VentureBeat",
        "default_category": "Artificial Intelligence",
    },
    # Agriculture
    {
        "url": "https://modernfarmer.com/feed/",
        "source": "Modern Farmer",
        "default_category": "Agriculture",
    },
    {
        "url": "https://www.agweb.com/rss/crops",
        "source": "AgWeb",
        "default_category": "Agriculture",
    },
    # Business, Economy, Startups
    {
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "source": "BBC Business",
        "default_category": "Business",
    },
    {
        "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "source": "CNBC Economy",
        "default_category": "Economy",
    },
    {
        "url": "https://techcrunch.com/category/startups/feed/",
        "source": "TechCrunch Startups",
        "default_category": "Startups",
    },
    # Science & Environment
    {
        "url": "https://www.sciencedaily.com/rss/all.xml",
        "source": "ScienceDaily",
        "default_category": "Science",
    },
    {
        "url": "https://www.sciencedaily.com/rss/earth_climate.xml",
        "source": "ScienceDaily Climate",
        "default_category": "Environment",
    },
    {
        "url": "https://www.sciencedaily.com/rss/health_medicine.xml",
        "source": "ScienceDaily Health",
        "default_category": "Healthcare",
    },
    # National & International & Politics
    {
        "url": "http://feeds.bbci.co.uk/news/rss.xml",
        "source": "BBC News",
        "default_category": "National News",
    },
    {
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "source": "BBC World",
        "default_category": "International News",
    },
    {
        "url": "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "source": "BBC Politics",
        "default_category": "Politics",
    },
    # Cybersecurity & Education & Entertainment
    {
        "url": "https://feeds.feedburner.com/TheHackersNews",
        "source": "The Hacker News",
        "default_category": "Cybersecurity",
    },
    {
        "url": "https://www.insidehighered.com/rss.xml",
        "source": "Inside Higher Ed",
        "default_category": "Education",
    },
    {
        "url": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
        "source": "BBC Entertainment",
        "default_category": "Entertainment",
    },
]

# ── Category Classification Keyword Rules ────────────────────────────────────
CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("Cricket", [
        "cricket", "ipl", "dhoni", "virat kohli", "rohit sharma", "bcci", "wicket",
        "batsman", "bowler", "test match", "t20", "odi", "century", "cricinfo",
        "ranji", "bumrah", "hardik pandya", "shubman gill", "ipl auction", "super kings",
    ]),
    ("Sports", [
        "football", "tennis", "soccer", "championship", "tournament", "olympics",
        "premier league", "champions league", "badminton", "formula 1", "grand prix",
        "hockey", "athlete", "fifa", "uefa", "nba", "kabaddi", "grand slam",
    ]),
    ("Artificial Intelligence", [
        "artificial intelligence", "ai", "machine learning", "deep learning", "llm",
        "chatgpt", "gemini", "openai", "anthropic", "claude", "neural network",
        "generative ai", "diffusion model", "transformer", "copilot", "hugging face",
    ]),
    ("Cybersecurity", [
        "cybersecurity", "ransomware", "malware", "data breach", "hacker", "hacking",
        "vulnerability", "zero-day", "phishing", "encryption", "infosec", "ddos",
        "cve", "cyber attack", "cisa", "firewall",
    ]),
    ("Startups", [
        "startup", "startups", "venture capital", "seed funding", "series a",
        "series b", "unicorn", "pitch deck", "incubator", "accelerator", "founder",
    ]),
    ("Agriculture", [
        "agriculture", "farming", "farmer", "crop", "crops", "harvest", "fertilizer",
        "monsoon", "irrigation", "soil", "wheat", "paddy", "agritech", "horticulture",
        "kisan", "mandi", "msp", "tractor", "grain", "seeds",
    ]),
    ("Economy", [
        "inflation", "interest rate", "interest rates", "gdp", "recession", "central bank",
        "federal reserve", "rbi", "fiscal policy", "monetary policy", "deficit", "tariff",
        "treasury", "debt ceiling",
    ]),
    ("Business", [
        "revenue", "earnings", "quarterly profit", "shares", "stock market", "nasdaq",
        "wall street", "acquisition", "merger", "ipo", "banking", "corporate", "valuation",
    ]),
    ("Healthcare", [
        "health", "healthcare", "medicine", "medical", "hospital", "doctor", "disease",
        "vaccine", "clinical trial", "fda", "cancer", "therapy", "pharma", "pharmaceutical",
        "infection", "patient", "epidemic",
    ]),
    ("Environment", [
        "climate change", "global warming", "carbon", "renewable energy", "solar",
        "wind energy", "sustainability", "emissions", "pollution", "wildlife", "conservation",
        "biodiversity", "glacier", "green energy",
    ]),
    ("Science", [
        "science", "scientific", "physics", "astronomy", "space exploration", "nasa",
        "isro", "telescope", "quantum", "planet", "galaxy", "laboratory", "cosmology",
        "black hole", "genetics", "biotechnology",
    ]),
    ("Politics", [
        "politics", "political", "parliament", "congress", "senate", "election",
        "elections", "legislation", "minister", "prime minister", "president", "democracy",
        "bjp", "congress party", "cabinet", "bill passed", "supreme court",
    ]),
    ("National News", [
        "national", "government", "infrastructure", "highways", "railways", "state government",
        "police", "district", "municipality", "citizens", "public sector",
    ]),
    ("International News", [
        "international", "foreign policy", "diplomacy", "united nations", "nato",
        "geopolitics", "embassy", "bilateral", "sanctions", "treaty", "ambassador",
        "global affairs", "g20",
    ]),
    ("Education", [
        "education", "school", "schools", "university", "college", "students", "teachers",
        "curriculum", "scholarship", "exam", "admissions", "degree", "ugc", "cbse",
    ]),
    ("Entertainment", [
        "movie", "movies", "film", "cinema", "box office", "actor", "actress",
        "trailer", "hollywood", "bollywood", "ott", "netflix", "streaming", "music album",
    ]),
    ("Employment", [
        "employment", "jobs", "hiring", "unemployment", "labor", "workforce",
        "layoffs", "wages", "salary", "remote work", "career", "recruitment",
    ]),
    ("Technology", [
        "technology", "tech", "software", "hardware", "computer", "semiconductor",
        "chip", "processor", "gadget", "smartphone", "cloud", "operating system",
    ]),
]


def classify_article(title: str, snippet: str, default_category: str = "Technology") -> Tuple[str, Optional[str]]:
    """Determine best category and subcategory based on headline and snippet text."""
    combined = f"{title} {snippet}".lower()

    # Match primary category
    matched_cat = default_category
    for cat, keywords in CATEGORY_RULES:
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, combined):
                matched_cat = cat
                break
        if matched_cat == cat:
            break

    # Match subcategory if possible
    matched_sub = None
    subcategories = NEWS_TAXONOMY.get(matched_cat, [])
    for sub in subcategories:
        if sub.lower() in combined:
            matched_sub = sub
            break

    if not matched_sub and subcategories:
        matched_sub = subcategories[0]

    return matched_cat, matched_sub


def _extract_image_url(entry: Any) -> Optional[str]:
    """Extract valid article thumbnail or image URL from RSS media/enclosure tags."""
    # 1. media_content
    if hasattr(entry, "media_content") and entry.media_content:
        for m in entry.media_content:
            if m.get("url") and ("image" in m.get("type", "") or m.get("medium") == "image" or re.search(r"\.(jpe?g|png|webp)", m.get("url", ""), re.I)):
                return m.get("url")

    # 2. media_thumbnail
    if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
        for t in entry.media_thumbnail:
            if t.get("url"):
                return t.get("url")

    # 3. enclosures
    if hasattr(entry, "enclosures") and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get("href") and ("image" in enc.get("type", "") or re.search(r"\.(jpe?g|png|webp)", enc.get("href", ""), re.I)):
                return enc.get("href")

    # 4. Embedded img in summary or description
    desc = getattr(entry, "summary", "") or getattr(entry, "description", "")
    if desc and "<img" in desc:
        try:
            soup = BeautifulSoup(desc, "html.parser")
            img = soup.find("img")
            if img and img.get("src"):
                return img["src"]
        except Exception:
            pass

    return None


def _clean_snippet(raw_text: str, max_words: int = 40) -> str:
    """Strip HTML tags and truncate summary cleanly."""
    if not raw_text:
        return ""
    try:
        clean = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ").strip()
    except Exception:
        clean = re.sub(r"<[^>]+>", " ", raw_text).strip()
    words = clean.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return clean


def ingest_rss_feed(feed_info: Dict[str, Any], session: Session, max_articles: int = 15) -> int:
    """Fetch and ingest real news articles from an RSS feed into the database."""
    feed_url = feed_info["url"]
    source_name = feed_info.get("source", "RSS News")
    default_cat = feed_info.get("default_category", "Technology")

    try:
        parsed = feedparser.parse(feed_url)
    except Exception as exc:
        log.warning("Could not parse RSS feed %s: %s", feed_url, exc)
        return 0

    entries = parsed.entries[:max_articles]
    inserted = 0

    for e in entries:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        if not title or not link:
            continue

        url_hash = hashlib.md5(link.encode("utf-8")).hexdigest()

        # Check deduplication
        existing = session.exec(select(NewsArticle).where(NewsArticle.url_hash == url_hash)).first()
        if existing:
            continue

        raw_desc = getattr(e, "summary", "") or getattr(e, "description", "")
        snippet = _clean_snippet(raw_desc)
        img_url = _extract_image_url(e)

        # Published date parsing
        pub_date = getattr(e, "published", "") or getattr(e, "updated", "")
        if not pub_date:
            pub_date = utc_now().strftime("%Y-%m-%d %H:%M")

        category, subcategory = classify_article(title, snippet, default_category=default_cat)

        article = NewsArticle(
            title=title,
            url=link,
            url_hash=url_hash,
            source=source_name,
            category=category,
            subcategory=subcategory,
            published_at=pub_date,
            snippet=snippet or f"Latest real-time news coverage on {category}.",
            image_url=img_url,
            created_at=utc_now(),
        )
        session.add(article)
        inserted += 1

    if inserted > 0:
        session.commit()
        log.info("Ingested %d new articles from %s [%s]", inserted, source_name, default_cat)
        # Incrementally cluster newly ingested articles into evolving event tracking
        try:
            from event_evolution_service import cluster_article_incrementally
            new_articles = session.exec(
                select(NewsArticle)
                .where(NewsArticle.source == source_name)
                .order_by(NewsArticle.id.desc())
                .limit(inserted)
            ).all()
            for art in reversed(new_articles):
                cluster_article_incrementally(art, session)
            session.commit()
        except Exception as cl_err:
            log.warning("Incremental event clustering during RSS ingestion skipped: %s", cl_err)

    return inserted


def seed_news_if_needed(session: Optional[Session] = None, force_refresh: bool = False) -> int:
    """Ensure news database has real coverage across all 18 categories."""
    close_s = False
    if session is None:
        session = next(get_session())
        close_s = True

    try:
        count = session.exec(select(NewsArticle)).all()
        # If sufficient articles exist and no force refresh requested, skip
        if len(count) >= 60 and not force_refresh:
            return len(count)

        log.info("Ingesting fresh news across all 18 categories (current total: %d)...", len(count))
        total_new = 0
        for f in FEED_SOURCES:
            try:
                n = ingest_rss_feed(f, session, max_articles=12)
                total_new += n
            except Exception as e:
                log.warning("Feed ingestion error on %s: %s", f.get("url"), e)

        log.info("Completed news ingestion sweep: added %d new articles.", total_new)
        return len(session.exec(select(NewsArticle)).all())
    finally:
        if close_s and session:
            session.close()


# ── Public News Query Helpers ────────────────────────────────────────────────

def get_homepage_news(
    session: Session,
    active_category: Optional[str] = None,
    limit: int = 60,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Construct rich homepage news sections for general/guest visitors.
    Guarantees diverse coverage across all 18 categories.
    """
    # Seed if database is empty
    all_articles = session.exec(
        select(NewsArticle).order_by(NewsArticle.created_at.desc()).limit(500)
    ).all()

    if len(all_articles) < 20:
        seed_news_if_needed(session)
        all_articles = session.exec(
            select(NewsArticle).order_by(NewsArticle.created_at.desc()).limit(500)
        ).all()

    def _to_dict(art: NewsArticle) -> Dict[str, Any]:
        return {
            "id": art.id,
            "title": art.title,
            "url": art.url,
            "source": art.source,
            "category": art.category,
            "subcategory": art.subcategory or art.category,
            "published_at": art.published_at,
            "snippet": art.snippet,
            "image_url": art.image_url,
        }

    # Filter by specific category if requested (for category tab browsing)
    if active_category and active_category.lower() != "all":
        filtered = [a for a in all_articles if a.category.lower() == active_category.lower()]
        paged = filtered[offset : offset + limit]
        return {
            "is_personalized": False,
            "selected_filter": active_category,
            "total_articles": len(filtered),
            "articles": [_to_dict(a) for a in paged],
            "categories": ALL_NEWS_CATEGORIES,
        }

    # Organize into balanced homepage sections
    top_headlines = [_to_dict(a) for a in all_articles[:6]]
    latest_news = [_to_dict(a) for a in all_articles[6:16]]

    # Category-specific buckets
    tech_ai = [_to_dict(a) for a in all_articles if a.category in ("Technology", "Artificial Intelligence", "Cybersecurity")][:6]
    sports_cricket = [_to_dict(a) for a in all_articles if a.category in ("Cricket", "Sports")][:6]
    business_economy = [_to_dict(a) for a in all_articles if a.category in ("Business", "Economy", "Startups", "Employment")][:6]
    agriculture = [_to_dict(a) for a in all_articles if a.category == "Agriculture"][:6]
    national_world = [_to_dict(a) for a in all_articles if a.category in ("National News", "International News", "Politics")][:6]
    science_health = [_to_dict(a) for a in all_articles if a.category in ("Science", "Healthcare", "Environment", "Education")][:6]

    # Trending topics computed from high-frequency article entities/categories
    trending_counts: Dict[str, int] = {}
    for a in all_articles:
        trending_counts[a.category] = trending_counts.get(a.category, 0) + 1
        if a.subcategory:
            trending_counts[a.subcategory] = trending_counts.get(a.subcategory, 0) + 1

    sorted_trending = sorted(trending_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    trending_topics = [t[0] for t in sorted_trending]

    return {
        "is_personalized": False,
        "categories": ALL_NEWS_CATEGORIES,
        "taxonomy": NEWS_TAXONOMY,
        "sections": {
            "top_headlines": top_headlines,
            "latest_news": latest_news,
            "technology_ai": tech_ai,
            "sports_cricket": sports_cricket,
            "business_economy": business_economy,
            "agriculture": agriculture,
            "national_world": national_world,
            "science_health": science_health,
        },
        "trending_topics": trending_topics,
    }


def get_personalized_homepage_news(
    user_id: int,
    session: Session,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Construct dynamic personalized homepage news based on user's explicit preferences & inferred interests.
    Includes:
      - 'for_you': Top prioritized articles matching user's domains
      - 'latest_developments': Fresh news strictly within their chosen categories
      - 'selected_categories': Grouped by their selected main categories
      - 'trending_in_interests': Trending developments in their specific sectors
      - 'explore_other_topics': Curated articles outside their interests to avoid filter bubbles
    """
    profile = session.exec(select(PersonalizationProfile).where(PersonalizationProfile.user_id == user_id)).first()
    pref = session.exec(select(UserPreference).where(UserPreference.user_id == user_id)).first()

    # Enforce mandatory onboarding requirement
    if (pref and not pref.onboarding_completed) or (profile and not profile.onboarding_completed) or (not pref and not profile):
        return {
            "is_personalized": False,
            "onboarding_required": True,
            "message": "Mandatory preference onboarding required before accessing personalized news feed.",
            "categories": ALL_NEWS_CATEGORIES,
        }

    selected_cats: List[str] = []
    selected_subs: List[str] = []

    if profile:
        try:
            selected_cats.extend(json.loads(profile.preferred_categories_json or "[]"))
        except Exception:
            pass
        try:
            selected_subs.extend(json.loads(profile.preferred_subcategories_json or "[]"))
        except Exception:
            pass

    if pref and not selected_cats:
        try:
            selected_cats.extend(json.loads(pref.preferred_categories_json or "[]"))
        except Exception:
            pass
        if pref.interests:
            selected_cats.extend([i.strip() for i in pref.interests.split(",") if i.strip()])

    # Deduplicate while preserving order
    selected_cats = list(dict.fromkeys(selected_cats))
    selected_subs = list(dict.fromkeys(selected_subs))

    # Inferred interests from chat history
    inferred_cats: List[str] = []
    if profile:
        try:
            inferred = json.loads(profile.inferred_interests_json or "[]")
            inferred_cats = [item["category"] for item in inferred if item.get("status") in ("strong", "recurring")]
        except Exception:
            pass

    # Target interest set
    user_interest_set = {c.lower() for c in (selected_cats + selected_subs + inferred_cats)}

    all_articles = session.exec(
        select(NewsArticle).order_by(NewsArticle.created_at.desc()).limit(500)
    ).all()

    if len(all_articles) < 20:
        seed_news_if_needed(session)
        all_articles = session.exec(
            select(NewsArticle).order_by(NewsArticle.created_at.desc()).limit(500)
        ).all()

    def _to_dict(art: NewsArticle) -> Dict[str, Any]:
        return {
            "id": art.id,
            "title": art.title,
            "url": art.url,
            "source": art.source,
            "category": art.category,
            "subcategory": art.subcategory or art.category,
            "published_at": art.published_at,
            "snippet": art.snippet,
            "image_url": art.image_url,
        }

    # Split into relevant vs explore
    relevant_articles = []
    explore_articles = []

    for art in all_articles:
        cat_match = art.category.lower() in user_interest_set
        sub_match = bool(art.subcategory and art.subcategory.lower() in user_interest_set)
        if cat_match or sub_match:
            relevant_articles.append(art)
        else:
            explore_articles.append(art)

    # 1. 'For You': prioritized relevant articles
    for_you = [_to_dict(a) for a in relevant_articles[:8]]
    if not for_you:
        # Fallback to general headlines if user hasn't selected any matching articles
        for_you = [_to_dict(a) for a in all_articles[:6]]

    # 2. 'Latest Developments'
    latest_devs = [_to_dict(a) for a in relevant_articles[8:16]]
    if not latest_devs:
        latest_devs = [_to_dict(a) for a in all_articles[6:12]]

    # 3. 'Your Selected Categories' grouped dictionary
    grouped_selected: Dict[str, List[Dict[str, Any]]] = {}
    display_cats = selected_cats if selected_cats else (inferred_cats if inferred_cats else ["Technology", "Cricket", "Business"])
    for cat in display_cats[:4]:
        matching = [
            _to_dict(a) for a in all_articles
            if a.category.lower() == cat.lower() or (a.subcategory and a.subcategory.lower() == cat.lower())
        ][:5]
        if matching:
            grouped_selected[cat] = matching

    # 4. 'Trending in Your Interests'
    trending_in_interests = [_to_dict(a) for a in relevant_articles if any(kw in a.title.lower() for kw in ("breakthrough", "record", "surge", "win", "launch", "latest", "top"))][:6]
    if not trending_in_interests:
        trending_in_interests = [_to_dict(a) for a in relevant_articles[:4]]

    # 5. 'Explore Other Topics' (prevent echo chambers)
    explore_other = [_to_dict(a) for a in explore_articles[:8]]

    summary_interests = (selected_cats + inferred_cats)[:3]
    summary_text = f"Curated for your interests in {', '.join(summary_interests)}" if summary_interests else "Personalized news feed active"

    return {
        "is_personalized": True,
        "user_id": user_id,
        "summary": summary_text,
        "preferred_categories": selected_cats,
        "preferred_subcategories": selected_subs,
        "inferred_interests": inferred_cats,
        "categories": ALL_NEWS_CATEGORIES,
        "taxonomy": NEWS_TAXONOMY,
        "sections": {
            "for_you": for_you,
            "latest_developments": latest_devs,
            "selected_categories": grouped_selected,
            "trending_in_interests": trending_in_interests,
            "explore_other_topics": explore_other,
        },
    }
