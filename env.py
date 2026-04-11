"""
Email Triage OpenEnv v2
========================
Genuine sequential decision-making environment.

Key mechanics:
  1. Thread dependencies — emails unlock only after predecessors handled
  2. SLA clock — each step() ages all pending emails; breaches penalise score
  3. Context rules — correct action changes based on what agent did earlier
  4. Adversarial emails — phishing disguised as legitimate
  5. Escalation budget — limited escalations force prioritisation
  6. Reply quality — keyword-based scoring for substantive replies
  7. Cascade scoring — mishandling e001 changes grading of e014
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(v: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, v))


# ── Pydantic models ───────────────────────────────────────────────────────────

class EmailObject(BaseModel):
    id: str
    subject: str
    sender: str
    sender_name: str
    body: str
    timestamp: str
    has_attachment: bool
    thread_id: str
    sla_steps_remaining: int = 999   # visible to agent — countdown
    is_unlocked: bool = True          # False = depends on prior email


class Observation(BaseModel):
    task_id: str
    step_number: int
    inbox: List[EmailObject] = Field(default_factory=list)
    current_email: Optional[EmailObject] = None
    processed_count: int = 0
    pending_count: int = 0
    sla_breaches: int = 0
    escalation_budget: int = 5        # remaining escalations allowed
    done: bool = False
    message: str = ""


class Action(BaseModel):
    action_type: str   # escalate|route|reply|archive|mark_spam|defer|flag
    email_id: str
    urgency: Optional[str] = None
    category: Optional[str] = None
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
    processed_emails: List[Dict] = Field(default_factory=list)
    pending_emails: List[Dict] = Field(default_factory=list)
    action_history: List[Dict] = Field(default_factory=list)
    sla_breaches: int = 0
    escalation_budget: int = 5
    total_reward: float = 0.0
    done: bool = False


# ── Grading helpers ───────────────────────────────────────────────────────────

URGENCY_LEVELS = {"critical": 4, "high": 3, "medium": 2, "low": 1}

URGENCY_SCORE_TABLE = {
    (4, 4): 0.98, (3, 3): 0.98, (2, 2): 0.98, (1, 1): 0.98,
    (4, 3): 0.50, (3, 4): 0.60,
    (3, 2): 0.40, (2, 3): 0.50,
    (2, 1): 0.30, (1, 2): 0.40,
    (4, 2): 0.05, (4, 1): 0.02,
    (3, 1): 0.10,
    (2, 4): 0.20, (1, 3): 0.20, (1, 4): 0.10,
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

ACTION_ALTS = {
    "escalate": {"flag", "route"},
    "route": {"reply", "escalate"},
    "archive": {"mark_spam"},
    "mark_spam": {"archive"},
    "reply": {"route"},
    "defer": {"archive"},
}


def _score_urgency(predicted: Optional[str], gt: str) -> float:
    if not predicted:
        return 0.05
    p = URGENCY_LEVELS.get(predicted, 0)
    g = URGENCY_LEVELS.get(gt, 0)
    return URGENCY_SCORE_TABLE.get((g, p), 0.10)


def _score_routing(predicted_dept: Optional[str], gt_category: str) -> float:
    if not predicted_dept:
        return 0.05
    valid = DEPT_MAP.get(gt_category, set())
    if predicted_dept in valid:
        return 0.98
    if gt_category == "support" and predicted_dept == "sales":
        return 0.30
    if gt_category == "finance" and predicted_dept in ("legal", "support"):
        return 0.20
    return 0.05


def _score_action(predicted: str, gt: str, gt_urgency: str, budget_exceeded: bool) -> float:
    if predicted == gt:
        return 0.98
    alts = ACTION_ALTS.get(gt, set())
    if predicted in alts:
        return 0.50
    # Penalise: using escalate when budget is 0
    if predicted == "escalate" and budget_exceeded:
        return 0.10
    # Penalise ignoring critical
    if gt_urgency == "critical" and predicted in ("archive", "defer", "mark_spam"):
        return 0.02
    return 0.05


def _score_reply(reply_text: Optional[str], keywords: List[str], gt_action: str) -> float:
    if gt_action != "reply":
        return 0.98  # not required, no penalty
    if not reply_text or len(reply_text.strip()) < 20:
        return 0.05
    text_lower = reply_text.lower()
    if not keywords:
        # No specific keywords — just check length and professionalism
        return 0.70 if len(reply_text.strip()) > 50 else 0.40
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    ratio = hits / len(keywords)
    return max(0.10, min(0.98, 0.20 + 0.78 * ratio))


def _score_phishing(predicted_action: str, is_phishing: bool) -> float:
    """Extra score component for phishing detection."""
    if not is_phishing:
        return 0.98  # not applicable
    if predicted_action == "mark_spam":
        return 0.98  # caught it
    if predicted_action in ("reply", "route", "escalate"):
        return 0.02  # acted on it — dangerous
    return 0.30  # archived/deferred — missed but didn't act on it


# ── Main environment ──────────────────────────────────────────────────────────

class EmailTriageEnvV2:
    """
    Email Triage v2 — genuine sequential decision-making.

    New mechanics vs v1:
      - Thread dependencies: emails unlock after predecessors
      - SLA clock: sla_steps_remaining counts down each step
      - Context rules: correct action changes based on agent history
      - Escalation budget: max 5 escalations per episode
      - Phishing detection: adversarial emails with subtle tells
      - Cascade scoring: mishandling thread-head penalises follow-ups
    """

    TASK_CONFIG = {
        "classify_urgency": {
            "email_fn": "get_task1_emails",
            "max_steps": 5,
            "escalation_budget": 3,
            "weights": {"urgency": 0.55, "routing": 0.30, "action": 0.0, "reply": 0.0, "phishing": 0.15},
        },
        "triage_and_route": {
            "email_fn": "get_task2_emails",
            "max_steps": 20,
            "escalation_budget": 4,
            "weights": {"urgency": 0.25, "routing": 0.25, "action": 0.25, "reply": 0.10, "phishing": 0.15},
        },
        "inbox_zero": {
            "email_fn": "get_task3_emails",
            "max_steps": 50,
            "escalation_budget": 5,
            "weights": {"urgency": 0.15, "routing": 0.20, "action": 0.25, "reply": 0.15, "phishing": 0.25},
        },
    }

    def __init__(self, task_id: str = "classify_urgency"):
        assert task_id in self.TASK_CONFIG, f"Unknown task: {task_id}"
        self.task_id = task_id
        self.cfg = self.TASK_CONFIG[task_id]
        self.episode_id = f"{task_id}_{int(time.time())}"
        self._raw_emails: List[Dict] = []
        self._pending: List[Dict] = []       # locked + unlocked emails
        self._unlocked_ids: set = set()      # emails available to act on
        self._processed: List[Dict] = []
        self._action_history: List[Dict] = []  # {email_id, action_type, step}
        self.step_number = 0
        self.total_reward = 0.0
        self.sla_breaches = 0
        self.escalation_budget = self.cfg["escalation_budget"]
        self.done = False

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        import importlib, sys
        sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__file__)))
        from data.emails import get_task1_emails, get_task2_emails, get_task3_emails

        self.episode_id = f"{self.task_id}_{int(time.time())}"
        self.step_number = 0
        self.total_reward = 0.0
        self.sla_breaches = 0
        self.escalation_budget = self.cfg["escalation_budget"]
        self.done = False
        self._action_history = []
        self._processed = []

        fn_map = {
            "get_task1_emails": get_task1_emails,
            "get_task2_emails": get_task2_emails,
            "get_task3_emails": get_task3_emails,
        }
        self._raw_emails = fn_map[self.cfg["email_fn"]]()
        self._pending = copy.deepcopy(self._raw_emails)

        # Unlock emails with no dependencies
        self._unlocked_ids = {
            e["id"] for e in self._pending if not e.get("depends_on")
        }

        return self._make_obs(
            message=f"Inbox loaded: {len(self._pending)} emails. "
                    f"Escalation budget: {self.escalation_budget}. "
                    f"Unlocked: {len(self._unlocked_ids)} emails available."
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            return self._make_obs(), Reward(value=_SCORE_MIN, feedback="Episode done."), True, {}

        self.step_number += 1
        max_steps = self.cfg["max_steps"]

        # Tick SLA clock on all pending emails
        sla_newly_breached = self._tick_sla()

        # Find email
        email_data = self._find_unlocked(action.email_id)
        if email_data is None:
            locked = self._find_pending(action.email_id)
            if locked:
                msg = f"Email {action.email_id} is locked — process its dependencies first."
            else:
                msg = f"Email {action.email_id} not found."
            reward = Reward(value=_SCORE_MIN, feedback=msg)
            self._maybe_finish(max_steps)
            return self._make_obs(message=msg), reward, self.done, {"error": msg}

        # Check escalation budget
        budget_exceeded = (action.action_type == "escalate" and self.escalation_budget <= 0)

        # Compute effective ground truth (may be modified by context rules)
        effective_gt = self._resolve_context(email_data)

        # Grade
        reward, breakdown = self._grade(action, email_data, effective_gt, budget_exceeded)

        # Consume escalation budget
        if action.action_type == "escalate" and not budget_exceeded:
            self.escalation_budget -= 1

        # Record action in history
        self._action_history.append({
            "step": self.step_number,
            "email_id": action.email_id,
            "action_type": action.action_type,
            "reward": reward.value,
        })

        # Move email to processed
        email_data["agent_action"] = action.model_dump()
        email_data["effective_gt"] = effective_gt
        email_data["reward_earned"] = reward.value
        self._pending = [e for e in self._pending if e["id"] != action.email_id]
        self._unlocked_ids.discard(action.email_id)
        self._processed.append(email_data)

        # Unlock downstream emails
        newly_unlocked = self._unlock_dependents(action.email_id)

        self.total_reward += reward.value

        self._maybe_finish(max_steps)
        msg = f"{len(self._pending)} remaining. "
        if newly_unlocked:
            msg += f"Unlocked: {newly_unlocked}. "
        if sla_newly_breached:
            msg += f"SLA BREACHED: {sla_newly_breached}."

        obs = self._make_obs(message=msg)
        return obs, reward, self.done, {
            "effective_gt_action": effective_gt["gt_action"],
            "newly_unlocked": newly_unlocked,
            "sla_breaches_this_step": sla_newly_breached,
            "escalation_budget": self.escalation_budget,
        }

    def state(self) -> State:
        return State(
            task_id=self.task_id,
            step_number=self.step_number,
            episode_id=self.episode_id,
            processed_emails=self._processed,
            pending_emails=self._pending,
            action_history=self._action_history,
            sla_breaches=self.sla_breaches,
            escalation_budget=self.escalation_budget,
            total_reward=self.total_reward,
            done=self.done,
        )

    def final_score(self) -> float:
        total = len(self._raw_emails)
        if total == 0:
            return _SCORE_MIN
        avg = self.total_reward / total
        sla_penalty = 0.04 * self.sla_breaches
        budget_penalty = 0.02 * max(0, -self.escalation_budget)
        return _clamp(avg - sla_penalty - budget_penalty)

    # ── Grading ───────────────────────────────────────────────────────────────

    def _grade(
        self, action: Action, email_data: Dict, effective_gt: Dict, budget_exceeded: bool
    ) -> Tuple[Reward, Dict]:
        w = self.cfg["weights"]
        gt_action = effective_gt["gt_action"]
        gt_urgency = effective_gt["gt_urgency"]
        gt_category = email_data["gt_category"]
        gt_dept = effective_gt.get("gt_department", email_data["gt_department"])
        is_phishing = email_data.get("is_phishing", False)
        reply_keywords = effective_gt.get("reply_required_keywords",
                         email_data.get("reply_required_keywords", []))

        s_urgency  = _score_urgency(action.urgency, gt_urgency)
        s_routing  = _score_routing(action.department, gt_category)
        s_action   = _score_action(action.action_type, gt_action, gt_urgency, budget_exceeded)
        s_reply    = _score_reply(action.reply_text, reply_keywords, gt_action)
        s_phishing = _score_phishing(action.action_type, is_phishing)

        composite = (
            w["urgency"]  * s_urgency +
            w["routing"]  * s_routing +
            w["action"]   * s_action  +
            w["reply"]    * s_reply   +
            w["phishing"] * s_phishing
        )
        composite = _clamp(composite)

        breakdown = {
            "urgency":  round(s_urgency, 3),
            "routing":  round(s_routing, 3),
            "action":   round(s_action, 3),
            "reply":    round(s_reply, 3),
            "phishing": round(s_phishing, 3),
        }

        feedback = []
        if s_urgency < 0.5:
            feedback.append(f"Wrong urgency (predicted={action.urgency}, actual={gt_urgency})")
        if s_routing < 0.5:
            feedback.append(f"Wrong department (predicted={action.department}, expected={gt_dept})")
        if s_action < 0.5:
            feedback.append(f"Wrong action (took={action.action_type}, expected={gt_action})")
        if is_phishing and action.action_type != "mark_spam":
            feedback.append("Missed phishing email!")
        if budget_exceeded:
            feedback.append("Escalation budget exhausted!")
        if not feedback:
            feedback.append("Correct!")

        return Reward(value=round(composite, 4), breakdown=breakdown, feedback="; ".join(feedback)), breakdown

    # ── Context resolution ────────────────────────────────────────────────────

    def _resolve_context(self, email_data: Dict) -> Dict:
        """
        Modify ground truth based on what the agent has already done.
        This is the core of sequential dependency scoring.
        """
        rule = email_data.get("context_rule")
        base = {
            "gt_action": email_data["gt_action"],
            "gt_urgency": email_data["gt_urgency"],
            "gt_department": email_data["gt_department"],
            "reply_required_keywords": email_data.get("reply_required_keywords", []),
        }
        if not rule:
            return base

        prior_email_id = rule.get("if_email")
        prior_action = self._get_prior_action(prior_email_id)

        # Rule: "if prior email had action X, then this email's gt changes"
        if "if_action" in rule:
            if prior_action == rule["if_action"]:
                base["gt_action"] = rule.get("then_action", base["gt_action"])
                base["gt_urgency"] = rule.get("then_urgency", base["gt_urgency"])
                if "then_reply_keywords" in rule:
                    base["reply_required_keywords"] = rule["then_reply_keywords"]

        # Rule: "if prior email did NOT have action X, then this email's gt changes"
        if "if_action_not" in rule:
            if prior_action != rule["if_action_not"]:
                base["gt_action"] = rule.get("then_action", base["gt_action"])
                base["gt_urgency"] = rule.get("then_urgency", base["gt_urgency"])

        return base

    def _get_prior_action(self, email_id: Optional[str]) -> Optional[str]:
        if not email_id:
            return None
        for h in self._action_history:
            if h["email_id"] == email_id:
                return h["action_type"]
        return None

    # ── SLA clock ─────────────────────────────────────────────────────────────

    def _tick_sla(self) -> List[str]:
        """Decrement SLA counters. Return list of newly breached email ids."""
        newly_breached = []
        for email in self._pending:
            if email.get("sla_minutes", 999) < 999:
                email["sla_minutes"] = max(0, email["sla_minutes"] - 1)
                if email["sla_minutes"] == 0:
                    if not email.get("_sla_breached"):
                        email["_sla_breached"] = True
                        self.sla_breaches += 1
                        newly_breached.append(email["id"])
        return newly_breached

    # ── Dependency management ─────────────────────────────────────────────────

    def _unlock_dependents(self, processed_id: str) -> List[str]:
        """Unlock emails that were waiting on processed_id."""
        newly_unlocked = []
        for email in self._pending:
            if email["id"] in self._unlocked_ids:
                continue
            deps = email.get("depends_on", [])
            processed_ids = {e["id"] for e in self._processed}
            if all(d in processed_ids for d in deps):
                self._unlocked_ids.add(email["id"])
                newly_unlocked.append(email["id"])
        return newly_unlocked

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_unlocked(self, email_id: str) -> Optional[Dict]:
        if email_id not in self._unlocked_ids:
            return None
        return self._find_pending(email_id)

    def _find_pending(self, email_id: str) -> Optional[Dict]:
        for e in self._pending:
            if e["id"] == email_id:
                return e
        return None

    def _maybe_finish(self, max_steps: int):
        if not self._pending or self.step_number >= max_steps:
            self.done = True

    def _make_obs(self, message: str = "") -> Observation:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.emails import strip_ground_truth

        inbox = []
        for e in self._pending:
            stripped = strip_ground_truth(e)
            inbox.append(EmailObject(
                **stripped,
                sla_steps_remaining=e.get("sla_minutes", 999),
                is_unlocked=(e["id"] in self._unlocked_ids),
            ))

        # Sort: unlocked first, then by SLA urgency
        inbox.sort(key=lambda x: (not x.is_unlocked, x.sla_steps_remaining))
        current = next((e for e in inbox if e.is_unlocked), None)

        return Observation(
            task_id=self.task_id,
            step_number=self.step_number,
            inbox=inbox,
            current_email=current,
            processed_count=len(self._processed),
            pending_count=len(self._pending),
            sla_breaches=self.sla_breaches,
            escalation_budget=self.escalation_budget,
            done=self.done,
            message=message or f"{len(self._pending)} email(s) remaining.",
        )
