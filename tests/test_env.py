"""
Tests for Email Triage OpenEnv.
Run: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env import Action, EmailTriageEnv, Observation, Reward, State


# ── Fixtures ──────────────────────────────────

@pytest.fixture
def env_easy():
    return EmailTriageEnv(task_id="classify_urgency")

@pytest.fixture
def env_medium():
    return EmailTriageEnv(task_id="triage_and_route")

@pytest.fixture
def env_hard():
    return EmailTriageEnv(task_id="inbox_zero")


# ── Spec compliance ───────────────────────────

class TestSpecCompliance:
    def test_reset_returns_observation(self, env_easy):
        obs = env_easy.reset()
        assert isinstance(obs, Observation)

    def test_reset_clean_state(self, env_easy):
        env_easy.reset()
        env_easy.reset()  # Second reset should also work
        obs = env_easy.reset()
        assert obs.step_number == 0
        assert obs.processed_count == 0
        assert not obs.done

    def test_step_returns_correct_types(self, env_easy):
        obs = env_easy.reset()
        email_id = obs.inbox[0].id
        action = Action(
            action_type="classify",
            email_id=email_id,
            urgency="critical",
            category="engineering",
        )
        obs2, reward, done, info = env_easy.step(action)
        assert isinstance(obs2, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_reward_in_01_range(self, env_medium):
        obs = env_medium.reset()
        for email in obs.inbox:
            action = Action(
                action_type="route",
                email_id=email.id,
                urgency="medium",
                category="support",
                department="support",
            )
            _, reward, done, _ = env_medium.step(action)
            assert 0.0 <= reward.value <= 1.0
            if done:
                break

    def test_state_returns_state(self, env_easy):
        env_easy.reset()
        state = env_easy.state()
        assert isinstance(state, State)
        assert state.task_id == "classify_urgency"

    def test_done_after_all_emails_processed(self, env_easy):
        obs = env_easy.reset()
        email_id = obs.inbox[0].id
        action = Action(
            action_type="classify",
            email_id=email_id,
            urgency="critical",
            category="engineering",
        )
        obs2, reward, done, info = env_easy.step(action)
        assert done  # only 1 email → done after 1 step


# ── Grading logic ─────────────────────────────

class TestGrading:
    def test_correct_critical_escalation_scores_high(self, env_easy):
        """Correctly handling a critical email should score well."""
        obs = env_easy.reset()
        email_id = obs.inbox[0].id  # e001 is critical/escalate
        action = Action(
            action_type="escalate",
            email_id=email_id,
            urgency="critical",
            category="engineering",
            department="engineering",
            reason="Production outage"
        )
        _, reward, _, _ = env_easy.step(action)
        assert reward.value >= 0.7

    def test_archiving_critical_email_penalized(self, env_easy):
        """Archiving a critical email should be penalized."""
        obs = env_easy.reset()
        email_id = obs.inbox[0].id  # critical email
        action = Action(
            action_type="archive",
            email_id=email_id,
            urgency="low",
            category="internal",
        )
        _, reward, _, _ = env_easy.step(action)
        assert reward.value < 0.3

    def test_spam_correctly_marked(self, env_medium):
        """Marking a spam email as spam should score high."""
        obs = env_medium.reset()
        # e011 is spam email
        spam_email = next((e for e in obs.inbox if e.id == "e011"), None)
        if spam_email is None:
            pytest.skip("spam email not in task2 set")
        action = Action(
            action_type="mark_spam",
            email_id="e011",
            urgency="low",
            category="spam",
        )
        _, reward, _, _ = env_medium.step(action)
        assert reward.value >= 0.5

    def test_reward_partial_credit_urgency(self, env_easy):
        """Close urgency guess (high for critical) gives partial credit."""
        obs = env_easy.reset()
        email_id = obs.inbox[0].id
        action = Action(
            action_type="escalate",
            email_id=email_id,
            urgency="high",   # close but not exact
            category="engineering",
            department="engineering",
        )
        _, reward, _, _ = env_easy.step(action)
        assert 0.1 < reward.value < 0.9  # partial, not full

    def test_unknown_email_id_returns_zero_reward(self, env_easy):
        env_easy.reset()
        action = Action(
            action_type="archive",
            email_id="nonexistent",
            urgency="low",
        )
        _, reward, _, info = env_easy.step(action)
        assert reward.value == 0.0

    def test_final_score_normalized(self, env_medium):
        obs = env_medium.reset()
        for email in obs.inbox:
            action = Action(
                action_type="escalate",
                email_id=email.id,
                urgency="high",
                category="support",
                department="support",
            )
            _, _, done, _ = env_medium.step(action)
            if done:
                break
        score = env_medium.final_score()
        assert 0.0 <= score <= 1.0


# ── Task difficulty ───────────────────────────

class TestTaskDifficulty:
    def test_task1_has_1_email(self, env_easy):
        obs = env_easy.reset()
        assert len(obs.inbox) == 1

    def test_task2_has_10_emails(self, env_medium):
        obs = env_medium.reset()
        assert len(obs.inbox) == 10

    def test_task3_has_25_emails(self, env_hard):
        obs = env_hard.reset()
        assert len(obs.inbox) == 25

    def test_task3_contains_duplicate(self, env_hard):
        obs = env_hard.reset()
        thread_ids = [e.thread_id for e in obs.inbox]
        # t001 appears twice (e001 and e014)
        assert thread_ids.count("t001") >= 2

    def test_all_task_ids_valid(self):
        for task_id in ["classify_urgency", "triage_and_route", "inbox_zero"]:
            env = EmailTriageEnv(task_id=task_id)
            obs = env.reset()
            assert obs.task_id == task_id

    def test_invalid_task_raises(self):
        with pytest.raises(AssertionError):
            EmailTriageEnv(task_id="invalid_task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
