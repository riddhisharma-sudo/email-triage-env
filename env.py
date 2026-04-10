"""
Email Triage OpenEnv Environment
=================================
A real-world environment simulating corporate inbox management.

Three tasks:
  - Task 1 (easy):   classify_urgency   — classify a single email
  - Task 2 (medium): triage_and_route   — process 10 emails
  - Task 3 (hard):   inbox_zero         — full inbox management (25 emails)
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Pydantic Models (OpenEnv spec compliance)
# ──────────────────────────────────────────────

class EmailObject(BaseModel):
    id: str
    subject: str
    sender: str
    sender_name: str
    body: str
    timestamp: str
    has_attachment: bool
    thread_id: str


class Observation(BaseModel):
    task_id: str
    step_number: int
    inbox: List[EmailObject] = Field(default_factory=list)
    current_email: Optional[EmailObject] = None
    processed_count: int = 0
    pending_count: int = 0
    sla_breaches: int = 0
    escalated_count: int = 0
    done: bool = False
    message: str = ""


class Action(BaseModel):
    action_type: str  # classify | route | reply | archive | escalate | mark_spam | defer | flag
    email_id: str
    urgency: Optional[str] = None      # critical | high | medium | low
    category: Optional[str] = None     # support | sales | internal | spam | legal | finance | hr | engineering
    department: Optional[str] = None
    reply_text: Optional[str] = None
    reason: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class State(BaseModel):
    task_id: str
    step_number: int
    episode_id: str
    processed_emails: List[Dict[str, Any]] = Field(default_factory=list)
    pending_emails: List[Dict[str, Any]] = Field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False


# ──────────────────────────────────────────────
# Grader helpers
# ──────────────────────────────────────────────

URGENCY_LEVELS = {"critical": 4, "high": 3, "medium": 2, "low": 1}
URGENCY_SCORE = {
    (4, 4): 1.0, (3, 3): 1.0, (2, 2): 1.0, (1, 1): 1.0,
    (4, 3): 0.5, (3, 4): 0.6,  # close misses
    (3, 2): 0.4, (2, 3): 0.5,
    (2, 1): 0.3, (1, 2): 0.4,
    (4, 2): 0.0, (4, 1): 0.0,  # critical missed badly
    (3, 1): 0.1,
    (2, 4): 0.2, (1, 3): 0.2, (1, 4): 0.1,
}

DEPT_MAP = {
    "support": {"support", "engineering"},
    "sales": {"sales"},
    "internal": {"hr", "internal"},
    "spam": {None, "spam"},
    "legal": {"legal"},
    "finance": {"finance"},
    "hr": {"hr"},
    "engineering": {"engineering", "support"},
}


def score_urgency(predicted: Optional[str], ground_truth: str) -> float:
    if predicted is None:
        return 0.0
    pred_lvl = URGENCY_LEVELS.get(predicted, 0)
    gt_lvl = URGENCY_LEVELS.get(ground_truth, 0)
    return URGENCY_SCORE.get((gt_lvl, pred_lvl), 0.1)


def score_routing(predicted_dept: Optional[str], gt_category: str) -> float:
    if predicted_dept is None:
        return 0.0
    valid_depts = DEPT_MAP.get(gt_category, set())
    if predicted_dept in valid_depts:
        return 1.0
    # partial credit for reasonable near-misses
    if gt_category == "support" and predicted_dept == "sales":
        return 0.3
    if gt_category == "finance" and predicted_dept in ("legal", "support"):
        return 0.2
    return 0.0


def score_action(predicted_action: str, gt_action: str, gt_urgency: str) -> float:
    if predicted_action == gt_action:
        return 1.0
    # Partial credit for reasonable alternatives
    acceptable_alts = {
        "escalate": {"flag", "route"},
        "route": {"reply", "escalate"},
        "archive": {"mark_spam"},
        "mark_spam": {"archive"},
        "reply": {"route"},
    }
    alts = acceptable_alts.get(gt_action, set())
    if predicted_action in alts:
        return 0.5
    # Penalize archiving/ignoring critical emails
    if gt_urgency == "critical" and predicted_action in ("archive", "defer", "mark_spam"):
        return -0.2
    return 0.0


def score_reply_quality(reply_text: Optional[str], gt_action: str) -> float:
    """Simple heuristic: replies should be substantive (>20 chars), professional."""
    if gt_action != "reply":
        return 1.0  # not expected — no penalty
    if not reply_text:
        return 0.0
    if len(reply_text.strip()) < 20:
        return 0.2
    if len(reply_text.strip()) < 50:
        return 0.6
    return 1.0


def score_duplicate_detection(action: Action, email_data: Dict) -> float:
    """Reward archiving known duplicates, penalize re-escalating them."""
    is_dup = email_data.get("is_duplicate", False)
    if is_dup:
        if action.action_type == "archive":
            return 1.0
        if action.action_type in ("escalate", "flag"):
            return 0.0
        return 0.5
    return 1.0  # not a duplicate, no penalty


# ──────────────────────────────────────────────
# Core environment
# ──────────────────────────────────────────────

class EmailTriageEnv:
    """
    Email Triage OpenEnv Environment.

    Implements OpenEnv spec:
      reset() → Observation
      step(action) → (Observation, Reward, done, info)
      state() → State
    """

    TASK_EMAIL_COUNT = {
        "classify_urgency": 1,
        "triage_and_route": 10,
        "inbox_zero": 25,
    }
    TASK_MAX_STEPS = {
        "classify_urgency": 5,
        "triage_and_route": 20,
        "inbox_zero": 60,
    }

    def __init__(self, task_id: str = "classify_urgency"):
        assert task_id in self.TASK_EMAIL_COUNT, (
            f"Unknown task_id '{task_id}'. Choose from: {list(self.TASK_EMAIL_COUNT)}"
        )
        self.task_id = task_id
        self.episode_id = f"{task_id}_{int(time.time())}"
        self._raw_emails: List[Dict] = []
        self._pending: List[Dict] = []
        self._processed: List[Dict] = []
        self.step_number = 0
        self.total_reward = 0.0
        self.sla_breaches = 0
        self.done = False
        self._action_log: List[Dict] = []

    # ── OpenEnv interface ──────────────────────

    def reset(self) -> Observation:
        from data.emails import get_task1_email, get_task2_emails, get_task3_emails, strip_ground_truth

        self.episode_id = f"{self.task_id}_{int(time.time())}"
        self.step_number = 0
        self.total_reward = 0.0
        self.sla_breaches = 0
        self.done = False
        self._action_log = []
        self._processed = []

        if self.task_id == "classify_urgency":
            self._raw_emails = [get_task1_email()]
        elif self.task_id == "triage_and_route":
            self._raw_emails = get_task2_emails()
        else:
            self._raw_emails = get_task3_emails()

        self._pending = copy.deepcopy(self._raw_emails)

        inbox = [EmailObject(**strip_ground_truth(e)) for e in self._pending]
        current = inbox[0] if inbox else None

        return Observation(
            task_id=self.task_id,
            step_number=0,
            inbox=inbox,
            current_email=current,
            processed_count=0,
            pending_count=len(inbox),
            sla_breaches=0,
            escalated_count=0,
            done=False,
            message=f"Inbox loaded with {len(inbox)} email(s). Process them to complete the task.",
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            return self._make_obs(), Reward(value=0.0, feedback="Episode already done."), True, {}

        self.step_number += 1
        max_steps = self.TASK_MAX_STEPS[self.task_id]

        # Find the target email
        email_data = self._find_pending(action.email_id)
        if email_data is None:
            # Try processed (agent re-acted on already-handled email)
            reward = Reward(value=0.0, feedback=f"Email {action.email_id} not found or already processed.")
            obs = self._make_obs()
            self._maybe_finish(max_steps)
            return obs, reward, self.done, {"error": "email_not_found"}

        # Grade the action
        reward, info = self._grade_action(action, email_data)

        # Move email to processed
        email_data["agent_action"] = action.model_dump()
        email_data["reward_earned"] = reward.value
        self._pending = [e for e in self._pending if e["id"] != action.email_id]
        self._processed.append(email_data)

        self.total_reward += reward.value
        self._action_log.append({"step": self.step_number, "action": action.model_dump(), "reward": reward.value})

        # Check SLA breach
        if email_data.get("sla_hours", 999) < 1 and action.action_type in ("archive", "defer", "mark_spam"):
            self.sla_breaches += 1

        self._maybe_finish(max_steps)
        obs = self._make_obs()
        return obs, reward, self.done, info

    def state(self) -> State:
        return State(
            task_id=self.task_id,
            step_number=self.step_number,
            episode_id=self.episode_id,
            processed_emails=self._processed,
            pending_emails=self._pending,
            total_reward=self.total_reward,
            done=self.done,
        )

    # ── Grading logic ──────────────────────────

    def _grade_action(self, action: Action, email_data: Dict) -> Tuple[Reward, Dict]:
        gt_urgency = email_data["gt_urgency"]
        gt_category = email_data["gt_category"]
        gt_dept = email_data["gt_department"]
        gt_action = email_data["gt_action"]

        # Component scores
        urgency_score = score_urgency(action.urgency, gt_urgency)
        routing_score = score_routing(action.department, gt_category)
        action_score = score_action(action.action_type, gt_action, gt_urgency)
        reply_score = score_reply_quality(action.reply_text, gt_action)
        dup_score = score_duplicate_detection(action, email_data)

        # Task-specific weighting
        if self.task_id == "classify_urgency":
            # Urgency classification is the only thing that matters
            weights = {"urgency": 0.6, "category": 0.4, "action": 0.0, "reply": 0.0, "dup": 0.0}
            category_score = 1.0 if (action.category == gt_category) else 0.3
            routing_score = category_score
        elif self.task_id == "triage_and_route":
            weights = {"urgency": 0.3, "category": 0.3, "action": 0.3, "reply": 0.1, "dup": 0.0}
        else:  # inbox_zero
            weights = {"urgency": 0.2, "category": 0.2, "action": 0.25, "reply": 0.15, "dup": 0.2}

        composite = (
            weights["urgency"] * urgency_score
            + weights["category"] * routing_score
            + weights["action"] * action_score
            + weights["reply"] * reply_score
            + weights["dup"] * dup_score
        )
        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        breakdown = {
            "urgency_score": round(urgency_score, 3),
            "routing_score": round(routing_score, 3),
            "action_score": round(action_score, 3),
            "reply_score": round(reply_score, 3),
            "dup_score": round(dup_score, 3),
        }

        # Build human-readable feedback
        feedback_parts = []
        if urgency_score < 0.5:
            feedback_parts.append(f"Urgency mismatch (predicted={action.urgency}, actual={gt_urgency})")
        if routing_score < 0.5:
            feedback_parts.append(f"Routing error (predicted dept={action.department}, expected={gt_dept})")
        if action_score < 0.5:
            feedback_parts.append(f"Wrong action (took={action.action_type}, expected={gt_action})")
        if not feedback_parts:
            feedback_parts.append("Good job!")

        return (
            Reward(value=round(composite, 4), breakdown=breakdown, feedback="; ".join(feedback_parts)),
            {"gt_urgency": gt_urgency, "gt_action": gt_action, "gt_dept": gt_dept},
        )

    # ── Helpers ───────────────────────────────

    def _find_pending(self, email_id: str) -> Optional[Dict]:
        for e in self._pending:
            if e["id"] == email_id:
                return e
        return None

    def _maybe_finish(self, max_steps: int):
        if not self._pending or self.step_number >= max_steps:
            self.done = True

    def _make_obs(self) -> Observation:
        from data.emails import strip_ground_truth

        inbox = [EmailObject(**strip_ground_truth(e)) for e in self._pending]
        current = inbox[0] if inbox else None
        escalated = sum(1 for e in self._processed if e.get("agent_action", {}).get("action_type") == "escalate")

        return Observation(
            task_id=self.task_id,
            step_number=self.step_number,
            inbox=inbox,
            current_email=current,
            processed_count=len(self._processed),
            pending_count=len(inbox),
            sla_breaches=self.sla_breaches,
            escalated_count=escalated,
            done=self.done,
            message=(
                "All emails processed!" if self.done and not self._pending
                else f"{len(inbox)} email(s) remaining."
            ),
        )

    def final_score(self) -> float:
        """
        Normalized score in [0, 1] across the full episode.
        Used by inference.py for [END] logging.
        """
        total_emails = len(self._raw_emails)
        if total_emails == 0:
            return 0.0
        # Average per-email reward, with SLA breach penalty
        avg = self.total_reward / total_emails
        penalty = 0.05 * self.sla_breaches
        return max(0.0, min(1.0, avg - penalty))
