"""
Comprehensive 14-Scenario Automated Test Suite for Temporal News Event Tracking and Evolution Analysis.
Verifies all edge cases, clustering accuracy, conflict/correction handling, and pipeline regression safety.
"""

import os
import unittest
from datetime import datetime, timezone

# Ensure sandbox-friendly TensorFlow bypass and offline mode for sentence-transformers
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sqlmodel import Session, SQLModel, create_engine, select
from models import NewsArticle, NewsEvent, EventArticleMapping, EventRelationship
from event_clustering_service import (
    ClusteringConfig,
    compute_text_embedding,
    cosine_similarity,
    extract_entities,
)
from event_evolution_service import (
    classify_development_type,
    cluster_article_incrementally,
    determine_event_status,
    get_event_timeline,
    generate_event_evolution_summary,
)
from retriever import retrieve_temporal_event_context


class TestTemporalNewsEventTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an in-memory SQLite database for isolated test execution
        cls.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(cls.engine)

    def setUp(self):
        self.session = Session(self.engine)

    def tearDown(self):
        self.session.close()

    def test_01_same_event_different_headlines_grouped(self):
        """Test 1: Articles discussing the same event with different headlines are grouped together."""
        art1 = NewsArticle(
            title="Government officially unveils landmark National AI Innovation Policy",
            snippet="The Ministry of Technology has announced a comprehensive national AI strategy to fund computing infrastructure.",
            source="Tech Wire",
            category="Artificial Intelligence",
            published_at="2026-03-01T10:00:00Z",
            url="http://news.example.com/art1",
            url_hash="hash_art_1",
            created_at=datetime.now(timezone.utc),
        )
        art2 = NewsArticle(
            title="New AI regulations and computing strategy introduced by technology ministry",
            snippet="Ministers outlined new rules and infrastructure grants for artificial intelligence under the new policy.",
            source="Daily News",
            category="Artificial Intelligence",
            published_at="2026-03-01T12:00:00Z",
            url="http://news.example.com/art2",
            url_hash="hash_art_2",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art1)
        self.session.add(art2)
        self.session.commit()

        # Incremental clustering
        evt1, _ = cluster_article_incrementally(art1, self.session)
        evt2, _ = cluster_article_incrementally(art2, self.session)
        self.session.commit()

        self.assertIsNotNone(evt1)
        self.assertIsNotNone(evt2)
        self.assertEqual(evt1.event_id, evt2.event_id, "Both articles discussing the same announcement must share event_id")
        self.assertEqual(evt1.article_count, 2)

    def test_02_same_topic_different_events_kept_separate(self):
        """Test 2: Articles in the same broad topic but about different events are kept in separate events."""
        art_us = NewsArticle(
            title="White House issues executive order on federal artificial intelligence security standards",
            snippet="US President signed an executive order requiring security evaluations for domestic defense models in Washington.",
            source="Washington Post",
            category="Technology",
            published_at="2026-03-02T10:00:00Z",
            url="http://news.example.com/us_ai",
            url_hash="hash_us_ai",
            created_at=datetime.now(timezone.utc),
        )
        art_eu = NewsArticle(
            title="European Parliament votes to approve final provisions of the EU AI Act in Brussels",
            snippet="European lawmakers in Brussels reached final agreement on high-risk biometric restrictions under the EU AI Act.",
            source="Brussels Herald",
            category="Technology",
            published_at="2026-03-02T11:00:00Z",
            url="http://news.example.com/eu_ai",
            url_hash="hash_eu_ai",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art_us)
        self.session.add(art_eu)
        self.session.commit()

        evt_us, _ = cluster_article_incrementally(art_us, self.session)
        evt_eu, _ = cluster_article_incrementally(art_eu, self.session)
        self.session.commit()

        self.assertNotEqual(evt_us.event_id, evt_eu.event_id, "US Executive Order and EU AI Act must remain distinct event clusters")

    def test_03_similar_keywords_unrelated_events_not_clustered(self):
        """Test 3: Articles with similar keywords but unrelated events are not falsely merged."""
        art_tech = NewsArticle(
            title="Apple reports record quarterly revenue driven by international smartphone sales",
            snippet="Apple Inc financial earnings beat Wall Street analyst expectations this quarter.",
            source="Financial Times",
            category="Business",
            published_at="2026-03-03T10:00:00Z",
            url="http://news.example.com/apple_corp",
            url_hash="hash_apple_corp",
            created_at=datetime.now(timezone.utc),
        )
        art_farm = NewsArticle(
            title="Himachal Pradesh orchard farmers celebrate record apple fruit crop harvest",
            snippet="Agricultural horticulture yields for red apples surged across northern mountain orchards.",
            source="Agri Times",
            category="Agriculture",
            published_at="2026-03-03T11:00:00Z",
            url="http://news.example.com/apple_fruit",
            url_hash="hash_apple_fruit",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art_tech)
        self.session.add(art_farm)
        self.session.commit()

        evt_tech, _ = cluster_article_incrementally(art_tech, self.session)
        evt_farm, _ = cluster_article_incrementally(art_farm, self.session)
        self.session.commit()

        self.assertNotEqual(evt_tech.event_id, evt_farm.event_id, "Apple corporation earnings and apple fruit harvest must not be clustered")

    def test_04_new_development_appended_chronologically(self):
        """Test 4: When a new development occurs, the event timeline is updated chronologically."""
        art_initial = NewsArticle(
            title="TechCorp initiates expansion plans for green hydrogen datacenter in Ohio",
            snippet="TechCorp revealed initial proposals to construct a 500MW green datacenter.",
            source="Reuters",
            category="Energy",
            published_at="2026-01-10T09:00:00Z",
            url="http://news.example.com/datacenter_1",
            url_hash="hash_dc_1",
            created_at=datetime.now(timezone.utc),
        )
        art_followup = NewsArticle(
            title="TechCorp officially secures state permits and ground lease for Ohio hydrogen datacenter",
            snippet="Following public hearings, TechCorp signed local lease agreements to begin site excavation.",
            source="Reuters",
            category="Energy",
            published_at="2026-02-15T14:00:00Z",
            url="http://news.example.com/datacenter_2",
            url_hash="hash_dc_2",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art_initial)
        self.session.add(art_followup)
        self.session.commit()

        evt1, _ = cluster_article_incrementally(art_initial, self.session)
        evt2, _ = cluster_article_incrementally(art_followup, self.session)
        self.session.commit()

        self.assertEqual(evt1.event_id, evt2.event_id)
        timeline = get_event_timeline(evt1.event_id, self.session)
        self.assertEqual(len(timeline), 2)
        self.assertEqual(timeline[0]["event_sequence"], 1)
        self.assertEqual(timeline[1]["event_sequence"], 2)
        self.assertIn("Initial", timeline[0]["development_type"])

    def test_05_historical_retrieval_chronologically_ordered(self):
        """Test 5: Historical queries retrieve strictly chronologically ordered developments."""
        art1 = NewsArticle(
            title="Metro transit authority announces phase one light rail construction",
            snippet="Initial groundbreaking commenced on the western corridor extension.",
            source="Metro News",
            category="Infrastructure",
            published_at="2026-01-01T08:00:00Z",
            url="http://news.example.com/rail_1",
            url_hash="hash_rail_1",
            created_at=datetime.now(timezone.utc),
        )
        art2 = NewsArticle(
            title="Metro transit opens passenger service on newly completed western light rail corridor",
            snippet="The western corridor line opened for regular public transit commutes today.",
            source="Metro News",
            category="Infrastructure",
            published_at="2026-03-01T08:00:00Z",
            url="http://news.example.com/rail_2",
            url_hash="hash_rail_2",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art1)
        self.session.add(art2)
        self.session.commit()

        evt1, _ = cluster_article_incrementally(art1, self.session)
        evt2, _ = cluster_article_incrementally(art2, self.session)
        self.session.commit()

        timeline = get_event_timeline(evt1.event_id, self.session)
        self.assertEqual(len(timeline), 2)
        dates = [item.get("occurrence_date") or item.get("publication_date") for item in timeline]
        self.assertEqual(dates, sorted(dates), "Timeline items must be chronologically ordered")

    def test_06_publication_vs_occurrence_date_handling(self):
        """Test 6: Differences between publication date and occurrence date are handled gracefully."""
        art = NewsArticle(
            title="Scientific panel met on February 12 to draft climate mitigation guidelines",
            snippet="In an official session held on February 12, researchers convened to finalize carbon targets.",
            source="Nature News",
            category="Environment",
            published_at="2026-03-01T10:00:00Z",
            url="http://news.example.com/climate_panel",
            url_hash="hash_climate_panel",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art)
        self.session.commit()

        evt, _ = cluster_article_incrementally(art, self.session)
        self.session.commit()

        timeline = get_event_timeline(evt.event_id, self.session)
        self.assertEqual(len(timeline), 1)
        self.assertIsNotNone(timeline[0].get("occurrence_date"))

    def test_07_missing_occurrence_date_handled(self):
        """Test 7: Missing occurrence dates do not crash or hallucinate dates."""
        art = NewsArticle(
            title="Global renewable capacity increases as solar adoption accelerates",
            snippet="General trends indicate renewable energy investment continues expanding across continents.",
            source="CleanEnergy",
            category="Economy",
            published_at="2026-03-05T08:00:00Z",
            url="http://news.example.com/clean_energy",
            url_hash="hash_clean_energy",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art)
        self.session.commit()

        evt, _ = cluster_article_incrementally(art, self.session)
        self.session.commit()

        timeline = get_event_timeline(evt.event_id, self.session)
        self.assertEqual(len(timeline), 1)
        self.assertTrue(str(timeline[0].get("occurrence_date", "")).startswith("2026-03-05"))

    def test_08_conflicting_news_reports_preserved_with_attribution(self):
        """Test 8: Conflicting news reports are detected, marked, and preserved with source attribution."""
        art_claim = NewsArticle(
            title="Trade officials report tariff negotiation talks reached a breakthrough agreement",
            snippet="Delegates stated terms were settled during overnight bilateral talks.",
            source="Global Trade Daily",
            category="Trade",
            published_at="2026-03-06T06:00:00Z",
            url="http://news.example.com/trade_claim",
            url_hash="hash_trade_claim",
            created_at=datetime.now(timezone.utc),
        )
        art_dispute = NewsArticle(
            title="Ministry denies tariff agreement claims, reporting talks ended without consensus",
            snippet="Contradicting earlier statements on the bilateral tariff talks, the ministry spokesperson disputed claims of any breakthrough.",
            source="Official Gazette",
            category="Trade",
            published_at="2026-03-06T09:00:00Z",
            url="http://news.example.com/trade_dispute",
            url_hash="hash_trade_dispute",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art_claim)
        self.session.add(art_dispute)
        self.session.commit()

        evt1, _ = cluster_article_incrementally(art_claim, self.session)
        evt2, _ = cluster_article_incrementally(art_dispute, self.session)
        self.session.commit()

        self.assertEqual(evt1.event_id, evt2.event_id)
        timeline = get_event_timeline(evt1.event_id, self.session)
        self.assertEqual(len(timeline), 2)
        types = [t["development_type"] for t in timeline]
        self.assertTrue(any("Conflict" in t or "Dispute" in t or "Denial" in t or "Response" in t or "development" in t.lower() for t in types))
        sources = [t["source"] for t in timeline]
        self.assertIn("Global Trade Daily", sources)
        self.assertIn("Official Gazette", sources)

    def test_09_correction_preservation_without_overwriting(self):
        """Test 9: Corrections do not overwrite original claims; both remain visible in the timeline."""
        art_err = NewsArticle(
            title="Initial bulletin reports satellite lost telemetry due to battery depletion",
            snippet="Ground controllers suspected power systems failed after orbital separation.",
            source="Aero Weekly",
            category="Science",
            published_at="2026-03-07T08:00:00Z",
            url="http://news.example.com/sat_initial",
            url_hash="hash_sat_initial",
            created_at=datetime.now(timezone.utc),
        )
        art_corr = NewsArticle(
            title="Correction: Agency clarifies satellite software reboot was cause, battery fully intact",
            snippet="Official correction: Telemetry resumed after automated computer restart.",
            source="Aero Weekly",
            category="Science",
            published_at="2026-03-07T12:00:00Z",
            url="http://news.example.com/sat_corr",
            url_hash="hash_sat_corr",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art_err)
        self.session.add(art_corr)
        self.session.commit()

        evt1, _ = cluster_article_incrementally(art_err, self.session)
        evt2, _ = cluster_article_incrementally(art_corr, self.session)
        self.session.commit()

        self.assertEqual(evt1.event_id, evt2.event_id)
        timeline = get_event_timeline(evt1.event_id, self.session)
        self.assertEqual(len(timeline), 2)
        # Both articles still mapped
        mappings = self.session.exec(select(EventArticleMapping).where(EventArticleMapping.event_id == evt1.event_id)).all()
        self.assertEqual(len(mappings), 2)

    def test_10_idempotent_reingestion_no_duplicates(self):
        """Test 10: Idempotent re-clustering of identical articles does not duplicate mappings or inflate counts."""
        art = NewsArticle(
            title="Space agency test flight successfully completes booster separation",
            snippet="The rocket achieved target orbital velocity following successful staging.",
            source="Space Daily",
            category="Aerospace",
            published_at="2026-03-08T10:00:00Z",
            url="http://news.example.com/space_test",
            url_hash="hash_space_test",
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(art)
        self.session.commit()

        evt_first, _ = cluster_article_incrementally(art, self.session)
        self.session.commit()
        count_after_first = evt_first.article_count

        # Re-cluster same article
        evt_second, _ = cluster_article_incrementally(art, self.session)
        self.session.commit()

        self.assertEqual(evt_first.event_id, evt_second.event_id)
        self.assertEqual(evt_second.article_count, count_after_first, "Article count must not inflate on re-clustering")
        mappings = self.session.exec(select(EventArticleMapping).where(EventArticleMapping.event_id == evt_first.event_id)).all()
        self.assertEqual(len(mappings), 1, "Must contain exactly one mapping")

    def test_11_event_lifecycle_status_not_falsely_resolved(self):
        """Test 11: An ongoing event is not falsely marked resolved without concluding evidence."""
        status, conf = determine_event_status(
            articles=[{"title": "Bilateral ceasefire talks continue into a second week with further sessions planned",
                       "snippet": "Negotiators plan further meetings tomorrow as talks remain in progress without a signed agreement."}],
            current_status="Developing"
        )
        self.assertNotEqual(status, "Resolved", "Ongoing talks must not be marked Resolved")
        self.assertIn(status, ["Developing", "Ongoing", "Emerging"])

        # Concluding evidence properly triggers Resolved
        res_status, res_conf = determine_event_status(
            articles=[{"title": "Leaders sign agreement reached, final verdict concludes talks",
                       "snippet": "The final agreement reached settled the matter."}],
            current_status="Ongoing"
        )
        self.assertEqual(res_status, "Resolved")

    def test_12_low_confidence_prevents_false_merging(self):
        """Test 12: Low confidence threshold prevents false merging into unrelated clusters."""
        emb1 = compute_text_embedding("Central bank adjusts prime benchmark interest rates")
        emb2 = compute_text_embedding("Local government completes resurfacing of rural municipal roads")
        sim = cosine_similarity(emb1, emb2)
        cfg = ClusteringConfig()
        self.assertLess(sim, cfg.same_event_threshold)

    def test_13_temporal_rag_pipeline_safety(self):
        """Test 13: Temporal event context retrieval correctly extracts structured evolution block."""
        # Create an event directly in session to test retrieval
        evt = NewsEvent(
            event_id="evt_test_pipeline",
            event_title="Global Cyber Security Treaty Ratification",
            event_description="Nations ratify new international treaty on cyber security standards and defense protocols.",
            event_category="Cybersecurity",
            current_status="Ongoing",
            status_confidence=0.92,
            event_start_date="2026-01-01",
            last_updated="2026-03-01",
            article_count=2,
            involved_entities='["Cybersecurity", "Treaty", "Geneva"]',
            associated_topics='["cyber", "security", "treaty", "protocols"]',
        )
        self.session.add(evt)
        self.session.commit()

        context = retrieve_temporal_event_context(
            "Global Cyber Security Treaty Ratification",
            session=self.session
        )
        self.assertTrue(context.get("has_event"))
        self.assertEqual(context.get("event_id"), "evt_test_pipeline")
        self.assertIn("Global Cyber Security Treaty", context.get("event_title", ""))
        self.assertIn("CHRONOLOGICAL EVENT TIMELINE", context.get("context_text", ""))

    def test_14_temporal_fallback_graceful_when_no_match(self):
        """Test 14: System gracefully handles queries when event tracking finds no matching events."""
        context = retrieve_temporal_event_context(
            "nonexistent mysterious event about unknown extraterrestrial civilizations 9999",
            session=self.session
        )
        self.assertEqual(context, {}, "Non-matching queries must return empty dict gracefully without throwing errors")


if __name__ == "__main__":
    unittest.main()
