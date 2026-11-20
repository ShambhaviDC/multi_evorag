"""
test_feedback_intelligence.py — Automated Test Suite for EvoRAG Feedback Intelligence & Continuous Personalization.

Uses Python standard library unittest to verify all 12 core evaluation criteria:
  1. User submitting a like
  2. User submitting a dislike with optional reason and comment
  3. Changing feedback from like to dislike and vice versa
  4. Removing feedback by clicking the active button again
  5. Preventing duplicate records (one active feedback per user/session + message)
  6. Correct message and conversation association
  7. Guest feedback recorded with session_id without creating user profile
  8. Authenticated feedback correctly linked to user_id
  9. Repeated positive feedback updating learned preferences (Dimension A)
  10. Explicit onboarding preferences retaining priority over learned preferences
  11. Dislike reasons correctly classified into pipeline components (Dimension B)
  12. Feedback analytics computing real metrics accurately
"""

import json
import os
import sys
import unittest
import uuid
from datetime import datetime
from sqlmodel import Session, select

from database import engine, create_db_and_tables
from models import (
    User,
    Conversation,
    Message,
    PersonalizationProfile,
    ResponseFeedback,
    FeedbackCreate,
)
from feedback_intelligence import (
    save_or_update_feedback,
    delete_feedback,
    get_feedback_for_message,
    get_user_feedback_history,
    learn_user_preferences_from_feedback,
    analyze_response_quality_issues,
    analyze_persona_feedback_performance,
    generate_system_improvement_recommendations,
    get_feedback_analytics,
    REASON_TO_PIPELINE_COMPONENT,
)
from personalization_agent import (
    build_personalization_context,
    compute_personalization_scores,
)


class TestFeedbackIntelligence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_db_and_tables()

    def setUp(self):
        self.session = Session(engine)

    def tearDown(self):
        self.session.close()

    def test_01_submit_like(self):
        """1. User submitting a like."""
        msg_id = f"test_msg_{uuid.uuid4().hex[:8]}"
        fb, action = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="like",
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            query_text="What are recent quantum computing milestones?",
            execution_metadata={"top_personas": ["domain-expert", "analytical-critical"]}
        )

        self.assertIsNotNone(fb)
        self.assertEqual(fb.feedback_type, "like")
        self.assertEqual(action, "created")
        self.assertEqual(fb.message_id, msg_id)
        self.assertIsNotNone(fb.feedback_id)

    def test_02_submit_dislike_with_reason_and_comment(self):
        """2. User submitting a dislike with reason and comment."""
        msg_id = f"test_msg_{uuid.uuid4().hex[:8]}"
        fb, action = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="dislike",
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            reason="factually_incorrect",
            comment="The dates mentioned in paragraph 2 contradict official reports.",
            query_text="Summarize the budget announcement.",
            execution_metadata={"winning_persona": "empirical-evidential"}
        )

        self.assertIsNotNone(fb)
        self.assertEqual(fb.feedback_type, "dislike")
        self.assertEqual(fb.reason, "factually_incorrect")
        self.assertIn("contradict official reports", fb.comment)
        self.assertEqual(action, "created")

    def test_03_change_feedback_toggle(self):
        """3. Changing feedback from like to dislike and vice versa."""
        msg_id = f"test_msg_{uuid.uuid4().hex[:8]}"
        sess_id = f"sess_{uuid.uuid4().hex[:8]}"

        # Submit Like first
        fb1, action1 = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="like",
            session_id=sess_id,
        )
        self.assertEqual(action1, "created")
        self.assertEqual(fb1.feedback_type, "like")

        # Change to Dislike
        fb2, action2 = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="dislike",
            session_id=sess_id,
            reason="incomplete_answer"
        )
        self.assertEqual(action2, "updated")
        self.assertEqual(fb2.feedback_id, fb1.feedback_id)
        self.assertEqual(fb2.feedback_type, "dislike")
        self.assertEqual(fb2.reason, "incomplete_answer")

        # Change back to Like
        fb3, action3 = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="like",
            session_id=sess_id,
        )
        self.assertEqual(action3, "updated")
        self.assertEqual(fb3.feedback_type, "like")
        self.assertIsNone(fb3.reason)

    def test_04_remove_feedback(self):
        """4. Removing feedback by clicking the active button again."""
        msg_id = f"test_msg_{uuid.uuid4().hex[:8]}"
        sess_id = f"sess_{uuid.uuid4().hex[:8]}"

        fb, _ = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="like",
            session_id=sess_id,
        )
        self.assertIsNotNone(fb)

        # Delete
        deleted = delete_feedback(
            session=self.session,
            message_id=msg_id,
            session_id=sess_id
        )
        self.assertTrue(deleted)

        # Verify not found
        retrieved = get_feedback_for_message(
            session=self.session,
            message_id=msg_id,
            session_id=sess_id
        )
        self.assertIsNone(retrieved)

    def test_05_prevent_duplicate_records(self):
        """5. Preventing duplicate records (one active record per user/session + message)."""
        msg_id = f"test_msg_{uuid.uuid4().hex[:8]}"
        sess_id = f"sess_{uuid.uuid4().hex[:8]}"

        # Submit 3 times consecutively
        for _ in range(3):
            save_or_update_feedback(
                session=self.session,
                message_id=msg_id,
                feedback_type="like",
                session_id=sess_id,
            )

        # Count records in database
        records = self.session.exec(
            select(ResponseFeedback).where(
                ResponseFeedback.message_id == msg_id,
                ResponseFeedback.session_id == sess_id
            )
        ).all()
        self.assertEqual(len(records), 1, "Expected exactly 1 record, duplicate prevention failed!")

    def test_06_message_and_conversation_association(self):
        """6. Correct message and conversation association."""
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        sess_id = f"sess_{uuid.uuid4().hex[:8]}"

        fb, _ = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="like",
            session_id=sess_id,
            conversation_id=conv_id,
            query_text="Test query for message association"
        )

        self.assertEqual(fb.conversation_id, conv_id)
        self.assertEqual(fb.message_id, msg_id)
        self.assertEqual(fb.query_text, "Test query for message association")

    def test_07_guest_feedback_isolation(self):
        """7. Guest feedback recorded with session_id without creating user profile."""
        guest_sess = f"guest_session_{uuid.uuid4().hex[:12]}"
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"

        fb, _ = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="like",
            session_id=guest_sess,
            user=None  # Guest!
        )

        self.assertIsNone(fb.user_id)
        self.assertEqual(fb.session_id, guest_sess)

        # Verify no User was fabricated for guest
        user = self.session.exec(select(User).where(User.email == f"{guest_sess}@example.com")).first()
        self.assertIsNone(user, "Guest session must not create a User account!")

    def test_08_authenticated_feedback_linked_to_user(self):
        """8. Authenticated feedback correctly linked to user_id."""
        uname = f"testuser_{uuid.uuid4().hex[:6]}"
        user = User(
            email=f"{uname}@example.com",
            full_name=uname,
            hashed_password="fakehashfortest"
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        msg_id = f"auth_msg_{uuid.uuid4().hex[:8]}"
        fb, _ = save_or_update_feedback(
            session=self.session,
            message_id=msg_id,
            feedback_type="like",
            user=user
        )

        self.assertEqual(fb.user_id, user.id)
        history = get_user_feedback_history(session=self.session, user_id=user.id)
        self.assertTrue(any(h.message_id == msg_id for h in history))

    def test_09_repeated_positive_feedback_updates_learned_preferences(self):
        """9. Repeated positive feedback updating learned preferences (Dimension A)."""
        uname = f"pro_researcher_{uuid.uuid4().hex[:6]}"
        user = User(
            email=f"{uname}@example.com",
            full_name=uname,
            hashed_password="fakehashfortest"
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        # Create PersonalizationProfile
        profile = PersonalizationProfile(
            user_id=user.id,
            preferred_categories_json=json.dumps(["Technology"]),
            preferred_perspectives_json=json.dumps(["academic"]),
            learned_preferences_json="{}"
        )
        self.session.add(profile)
        self.session.commit()

        # User repeatedly likes answers with technical depth and empirical evidence
        for i in range(3):
            msg_id = f"tech_msg_{i}_{uuid.uuid4().hex[:6]}"
            save_or_update_feedback(
                session=self.session,
                message_id=msg_id,
                feedback_type="like",
                user=user,
                query_text=f"Explain technical deep-dive into LLM attention mechanisms step {i}",
                execution_metadata={
                    "top_personas": ["domain-expert", "analytical-critical", "empirical-evidential"],
                    "winning_persona": "domain-expert",
                    "complexity_level": "technical"
                }
            )

        # Run preference learning
        learned_dict = learn_user_preferences_from_feedback(user_id=user.id, session=self.session)
        self.assertIsNotNone(learned_dict)
        self.assertIn("total_signals_analyzed", learned_dict)
        self.assertGreaterEqual(learned_dict["total_signals_analyzed"], 3)
        self.assertTrue(learned_dict.get("prefer_technical_depth") is True)
        favored = learned_dict.get("favored_personas", [])
        self.assertIn("domain-expert", favored)

        # Refresh database profile and verify persisted JSON
        self.session.refresh(profile)
        persisted = json.loads(profile.learned_preferences_json)
        self.assertEqual(persisted["prefer_technical_depth"], True)
        self.assertIn("domain-expert", persisted["favored_personas"])

    def test_10_explicit_preferences_priority_over_learned_feedback(self):
        """10. Explicit onboarding preferences retaining priority over learned preferences."""
        uname = f"priority_user_{uuid.uuid4().hex[:6]}"
        user = User(
            email=f"{uname}@example.com",
            full_name=uname,
            hashed_password="fakehashfortest"
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        # Explicit onboarding preferences: Science & Health, strictly layperson
        explicit_cats = ["Science", "Health"]
        explicit_persp = ["layperson"]

        profile = PersonalizationProfile(
            user_id=user.id,
            enabled=True,
            preferred_categories_json=json.dumps(explicit_cats),
            preferred_perspectives_json=json.dumps(explicit_persp),
            learned_preferences_json="{}"
        )
        self.session.add(profile)
        self.session.commit()

        # User gives feedback on sports question
        save_or_update_feedback(
            session=self.session,
            message_id=f"msg_prio_{uuid.uuid4().hex[:6]}",
            feedback_type="like",
            user=user,
            query_text="Summarize football cup results",
            execution_metadata={"winning_persona": "domain-expert"}
        )

        # Trigger learning
        learn_user_preferences_from_feedback(user_id=user.id, session=self.session)

        # Verify explicit categories and perspectives were NEVER overwritten
        self.session.refresh(profile)
        self.assertEqual(json.loads(profile.preferred_categories_json), explicit_cats)
        self.assertEqual(json.loads(profile.preferred_perspectives_json), explicit_persp)

        # Build personalization context and verify explicit preferences are prominent
        ctx = build_personalization_context(profile, query="What is the latest health news?")
        self.assertIsNotNone(ctx)
        self.assertIn("Science", ctx)
        self.assertIn("Health", ctx)
        self.assertIn("layperson", ctx)

    def test_11_dislike_reasons_pipeline_classification(self):
        """11. Dislike reasons correctly classified into pipeline components (Dimension B)."""
        self.assertEqual(REASON_TO_PIPELINE_COMPONENT["factually_incorrect"], "Factual Grounding & Verification")
        self.assertEqual(REASON_TO_PIPELINE_COMPONENT["outdated_information"], "Temporal Event Tracking & Freshness")
        self.assertEqual(REASON_TO_PIPELINE_COMPONENT["irrelevant_query"], "Query Understanding & Routing")
        self.assertEqual(REASON_TO_PIPELINE_COMPONENT["missing_sources"], "Retrieval & FAISS Search")
        self.assertEqual(REASON_TO_PIPELINE_COMPONENT["incomplete_answer"], "Response Synthesis & Output Limits")

        # Add 2 outdated_information dislikes
        for i in range(2):
            save_or_update_feedback(
                session=self.session,
                message_id=f"stale_msg_{i}_{uuid.uuid4().hex[:6]}",
                feedback_type="dislike",
                reason="outdated_information",
                session_id=f"sess_{uuid.uuid4().hex[:6]}",
            )

        quality = analyze_response_quality_issues(session=self.session)
        self.assertGreaterEqual(quality["total_negative_reports"], 2)
        self.assertIn("Temporal Event Tracking & Freshness", quality["component_issues"])

    def test_12_feedback_analytics_real_metrics(self):
        """12. Feedback analytics computing real metrics accurately."""
        analytics = get_feedback_analytics(session=self.session)

        summary = analytics["summary"]
        self.assertIn("total_submissions", summary)
        self.assertIn("likes_count", summary)
        self.assertIn("dislikes_count", summary)
        self.assertIn("positive_percentage", summary)

        # Verify math
        total = summary["likes_count"] + summary["dislikes_count"]
        self.assertEqual(total, summary["total_submissions"])
        if total > 0:
            expected_pct = round((summary["likes_count"] / total) * 100, 1)
            self.assertAlmostEqual(summary["positive_percentage"], expected_pct, places=1)

        # Verify Chart.js structure
        charts = analytics["charts"]
        self.assertIn("satisfaction_doughnut", charts)
        self.assertIn("dislike_reasons_bar", charts)
        self.assertIn("trends_line", charts)

        # Verify recommendations
        recs = analytics["recommendations"]
        self.assertIsInstance(recs, list)
        self.assertGreaterEqual(len(recs), 1)
        self.assertIn("component", recs[0])
        self.assertIn("recommendation", recs[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
