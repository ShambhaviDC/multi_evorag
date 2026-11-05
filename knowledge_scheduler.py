"""
knowledge_scheduler.py — EvoRAG Phase 6: Automated Knowledge Ingestion and Orchestration

Features:
- APScheduler-based periodic news ingestion with a 3-day interval.
- Update-due verification: if updated < 3 days ago, reuses existing FAISS index.
- Incremental updates: only embeds newly discovered chunks via all-MiniLM-L6-v2.
- Crash-safe atomic persistence for FAISS index and metadata sidecar.
- Records ingestion status and timestamp in index/ingestion_status.json.
- Zero persona execution in background news ingestion (personas only run for user queries).
- Clear progress logging at every stage (fetch, clean, chunk, filter, embed, update).

Usage:
    python knowledge_scheduler.py            # Starts scheduler (runs check/ingestion then schedules every 3 days)
    python knowledge_scheduler.py --once     # Runs single check/update and exits
    python knowledge_scheduler.py --force    # Forces immediate fetch and incremental update
    python knowledge_scheduler.py --check-only # Only reports whether an update is due
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import (
    ROOT_DIR,
    INDEX_DIR,
    FAISS_INDEX_FILE,
    META_FILE,
    AI_COVERAGE_QUERIES,
    MAX_ARTICLES_PER_SOURCE,
)
from fetcher import fetch
from embedder import add_chunks_to_index, release_model, load_index

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evorag_scheduler")

INGESTION_STATUS_FILE = INDEX_DIR / "ingestion_status.json"
UPDATE_INTERVAL_DAYS = 3


def load_ingestion_status() -> Dict:
    """Load the latest recorded ingestion status from disk."""
    if not INGESTION_STATUS_FILE.exists():
        return {}
    try:
        with open(INGESTION_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not read %s: %s", INGESTION_STATUS_FILE, exc)
        return {}


def save_ingestion_status(status_data: Dict) -> None:
    """Safely persist ingestion status using an atomic swap."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = INGESTION_STATUS_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, INGESTION_STATUS_FILE)
    log.info("Saved ingestion status → %s", INGESTION_STATUS_FILE)


def is_update_due(interval_days: int = UPDATE_INTERVAL_DAYS) -> Tuple[bool, str]:
    """
    Check whether a knowledge base update is due.

    Returns:
        (is_due: bool, reason: str)
    """
    if not FAISS_INDEX_FILE.exists() or not META_FILE.exists():
        return True, "FAISS index or metadata sidecar not found on disk. Initial build required."

    status = load_ingestion_status()
    last_update_str = status.get("last_successful_update")
    if not last_update_str:
        return True, "No previous successful ingestion recorded in status file."

    try:
        last_dt = datetime.fromisoformat(last_update_str.replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        elapsed = now - last_dt

        if elapsed >= timedelta(days=interval_days):
            return True, (
                f"Previous update was {elapsed.days} days ago "
                f"(>= {interval_days} day interval threshold). Update is due."
            )
        else:
            remaining = timedelta(days=interval_days) - elapsed
            hrs_remaining = int(remaining.total_seconds() // 3600)
            return False, (
                f"Previous update was {elapsed.days}d {elapsed.seconds // 3600}h ago "
                f"(< {interval_days} day interval threshold). Index is fresh. "
                f"Next update due in ~{hrs_remaining} hours."
            )
    except Exception as exc:
        return True, f"Error parsing last update timestamp ({last_update_str}): {exc}. Update required."


def run_ingestion_pipeline(queries: Optional[List[str]] = None, force: bool = False) -> Dict:
    """
    Execute news ingestion, chunk deduplication, and incremental FAISS update.
    Does NOT execute any personas (keeps memory and CPU focused strictly on ingestion).
    """
    print("\n" + "=" * 65)
    print("  EvoRAG Automated News Ingestion Pipeline")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 65)

    due, reason = is_update_due()
    log.info("Update check: due=%s (%s)", due, reason)

    if not due and not force:
        print(f"\n[INFO] {reason}")
        if FAISS_INDEX_FILE.exists() and META_FILE.exists():
            idx, meta = load_index()
            print(f"[OK]   Existing FAISS index reused: {idx.ntotal} vectors | {len(meta)} metadata entries.")
            return {
                "status": "reused",
                "reason": reason,
                "vectors": idx.ntotal,
                "metadata_count": len(meta),
            }
        return {"status": "reused", "reason": reason}

    if force and not due:
        log.info("Update forced by user flag (--force).")

    # Step 1: Determine queries to fetch
    target_queries = queries or AI_COVERAGE_QUERIES[:3]  # Focus sweep on top topics
    log.info("Starting ingestion across %d query topics: %s", len(target_queries), target_queries)

    all_fetched_chunks: List[Dict] = []
    t_start = time.time()

    for idx, query in enumerate(target_queries, start=1):
        log.info("[%d/%d] Fetching news for: '%s'...", idx, len(target_queries), query)
        try:
            chunks = fetch(query, max_per_source=min(5, MAX_ARTICLES_PER_SOURCE))
            log.info("[%d/%d] Obtained %d candidate chunks for '%s'.", idx, len(target_queries), len(chunks), query)
            all_fetched_chunks.extend(chunks)
        except Exception as exc:
            log.error("[%d/%d] Fetch failed for '%s': %s", idx, len(target_queries), query, exc)

    log.info("Total raw candidate chunks collected across all queries: %d", len(all_fetched_chunks))

    # Step 2: Incremental FAISS update (only embeds newly discovered chunks)
    print("\n[...] Comparing with existing index and embedding new chunks...")
    try:
        index, metadata, new_added = add_chunks_to_index(all_fetched_chunks)
        release_model()
    except Exception as exc:
        log.exception("Incremental FAISS update failed: %s", exc)
        # Record failure status
        err_status = {
            "last_attempt": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": str(exc),
            "update_interval_days": UPDATE_INTERVAL_DAYS,
        }
        save_ingestion_status(err_status)
        return {"status": "failed", "error": str(exc)}

    elapsed_sec = round(time.time() - t_start, 1)

    # Step 3: Record successful update status
    success_status = {
        "last_successful_update": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "total_vectors": index.ntotal if index else 0,
        "metadata_count": len(metadata),
        "new_chunks_added": new_added,
        "candidate_chunks_evaluated": len(all_fetched_chunks),
        "update_interval_days": UPDATE_INTERVAL_DAYS,
        "elapsed_seconds": elapsed_sec,
        "last_error": None,
    }
    save_ingestion_status(success_status)

    print("\n" + "=" * 65)
    print(f"  [OK] Ingestion complete in {elapsed_sec}s")
    print(f"       New chunks embedded : {new_added}")
    print(f"       Total FAISS vectors : {index.ntotal if index else 0}")
    print(f"       Metadata entries    : {len(metadata)}")
    print(f"       Status recorded in  : {INGESTION_STATUS_FILE.name}")
    print("=" * 65 + "\n")

    return success_status


def start_scheduler():
    """Start APScheduler in blocking mode with a 3-day interval."""
    print("\n" + "=" * 65)
    print("  EvoRAG News Ingestion & Knowledge-Base Scheduler")
    print(f"  Update Frequency: Every {UPDATE_INTERVAL_DAYS} days")
    print(f"  Target Model: sentence-transformers/all-MiniLM-L6-v2")
    print("=" * 65)

    # 1. Perform immediate startup check/ingestion if due
    log.info("Performing initial check on startup...")
    run_ingestion_pipeline()

    # 2. Configure APScheduler
    scheduler = BlockingScheduler(timezone="UTC")
    trigger = IntervalTrigger(days=UPDATE_INTERVAL_DAYS)

    scheduler.add_job(
        run_ingestion_pipeline,
        trigger=trigger,
        id="evorag_news_ingestion",
        name="EvoRAG 3-Day News Ingestion",
        replace_existing=True,
    )

    next_run = datetime.now(timezone.utc) + timedelta(days=UPDATE_INTERVAL_DAYS)
    log.info(
        "APScheduler running in background. Next scheduled update: %s",
        next_run.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    print("\n[Scheduler Active] Press Ctrl+C at any time to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler received shutdown signal. Exiting cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="EvoRAG Knowledge-Base Scheduler & Ingestion Orchestration."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run check/ingestion once and exit immediately (no background scheduler loop).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force an immediate fetch, chunk, and incremental embedding run even if not due.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check whether an index update is due and exit.",
    )
    args = parser.parse_args()

    if args.check_only:
        due, reason = is_update_due()
        print(f"\nUpdate Due : {due}")
        print(f"Reason     : {reason}\n")
        sys.exit(0 if not due else 1)

    if args.once or args.force:
        run_ingestion_pipeline(force=args.force)
        sys.exit(0)

    start_scheduler()


if __name__ == "__main__":
    main()

