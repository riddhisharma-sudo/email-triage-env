"""
Email dataset v2 — with thread dependencies, time-sensitive SLA,
adversarial/phishing emails, and cross-email state effects.

Key design principles:
- Emails in the same thread AFFECT each other's correct action
- Some emails are only actionable after others are handled
- Phishing emails disguised as legitimate
- Budget constraints (limited escalations)
- Correct action depends on WHAT THE AGENT HAS ALREADY DONE
"""

import copy
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────
# Thread definitions — emails that depend on each other
# ─────────────────────────────────────────────
#
# Each email has:
#   id, subject, sender, body, timestamp, has_attachment, thread_id
#   gt_urgency, gt_category, gt_department, gt_action
#   sla_minutes  — how many env steps before SLA breach
#   depends_on   — list of email ids that must be processed first
#   unlocks      — list of email ids this action reveals/unlocks
#   is_phishing  — disguised as legitimate, correct action is mark_spam
#   phishing_tell— subtle clue in the email
#   context_rule — dict describing how prior actions change correct response
#     e.g. {"if_email": "e002", "if_action": "escalate", "then_action": "route"}
#   reply_required_keywords — words that must appear in reply_text for full score
#   budget_cost  — how many escalation budget points this action costs

EMAILS: List[Dict[str, Any]] = [

    # ══════════════════════════════════════════
    # THREAD 1: Production Outage Cascade
    # e001 → agent escalates → e014 unlocks (follow-up from CTO)
    # If agent archives e001, e014 becomes a complaint and gt_action changes
    # ══════════════════════════════════════════
    {
        "id": "e001",
        "subject": "CRITICAL: Payment service down — transactions failing",
        "sender": "monitoring@company.com",
        "sender_name": "PagerDuty Alert",
        "body": (
            "ALERT: Payment service (pay-svc-prod) is returning 500 errors. "
            "Error rate: 94% over last 5 minutes. Affected: checkout, subscriptions. "
            "Revenue impact: ~$12,000/minute. On-call engineer: not acknowledging. "
            "Escalate immediately to incident commander."
        ),
        "timestamp": "2024-03-15T09:00:00Z",
        "has_attachment": False,
        "thread_id": "t_outage",
        "gt_urgency": "critical",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "escalate",
        "sla_minutes": 2,
        "depends_on": [],
        "unlocks": ["e014"],
        "is_phishing": False,
        "budget_cost": 1,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e014",
        "subject": "Re: CRITICAL: Payment service down — status update?",
        "sender": "cto@company.com",
        "sender_name": "Priya Mehta (CTO)",
        "body": (
            "Team — I've been notified of the payment outage. "
            "What's the current status? Do we have an ETA for resolution? "
            "Board is asking questions. Please reply with a status update within 10 minutes."
        ),
        "timestamp": "2024-03-15T09:08:00Z",
        "has_attachment": False,
        "thread_id": "t_outage",
        "gt_urgency": "critical",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "reply",
        "sla_minutes": 3,
        "depends_on": ["e001"],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": ["status", "investigating", "team"],
        # If e001 was NOT escalated, this becomes an angry complaint → escalate
        "context_rule": {
            "if_email": "e001",
            "if_action_not": "escalate",
            "then_action": "escalate",
            "then_urgency": "critical",
        },
    },

    # ══════════════════════════════════════════
    # THREAD 2: Legal chain — NDA → contract → lawsuit threat
    # Must handle in order; each response affects next
    # ══════════════════════════════════════════
    {
        "id": "e002",
        "subject": "NDA signature required — partnership with Nexus AI",
        "sender": "legal@nexusai.com",
        "sender_name": "Nexus AI Legal",
        "body": (
            "Please find attached our standard NDA for the proposed partnership. "
            "We require signatures from both parties before sharing technical specs. "
            "Please route to your legal team for review and return signed within 5 business days."
        ),
        "timestamp": "2024-03-13T10:00:00Z",
        "has_attachment": True,
        "thread_id": "t_legal",
        "gt_urgency": "medium",
        "gt_category": "legal",
        "gt_department": "legal",
        "gt_action": "route",
        "sla_minutes": 30,
        "depends_on": [],
        "unlocks": ["e009"],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e009",
        "subject": "Re: NDA — urgent follow-up, deadline tomorrow",
        "sender": "ceo@nexusai.com",
        "sender_name": "Marcus Webb (Nexus AI CEO)",
        "body": (
            "We sent our NDA 48 hours ago and have not received any acknowledgment. "
            "Our board meeting is tomorrow where we planned to announce this partnership. "
            "If we don't hear back by 5pm today, we will need to proceed with a competitor. "
            "This is a $4M annual deal."
        ),
        "timestamp": "2024-03-15T08:00:00Z",
        "has_attachment": False,
        "thread_id": "t_legal",
        "gt_urgency": "high",
        "gt_category": "sales",
        "gt_department": "sales",
        "gt_action": "escalate",
        "sla_minutes": 5,
        "depends_on": ["e002"],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 1,
        "reply_required_keywords": [],
        # If e002 was already routed, this just needs a reply acknowledging
        "context_rule": {
            "if_email": "e002",
            "if_action": "route",
            "then_action": "reply",
            "then_urgency": "high",
            "then_reply_keywords": ["apologies", "legal", "today"],
        },
    },

    # ══════════════════════════════════════════
    # THREAD 3: Customer data loss complaint → refund chain
    # ══════════════════════════════════════════
    {
        "id": "e003",
        "subject": "Data missing after your migration — 6 months of records gone",
        "sender": "cto@enterprise-client.com",
        "sender_name": "David Park (EnterpriseCo CTO)",
        "body": (
            "Your team ran a database migration last night and wiped 6 months of our project data. "
            "Account: ENT-00291. We have 200 engineers who cannot work. "
            "I need a call with your CEO and VP Engineering in the next 2 hours. "
            "We are considering legal action."
        ),
        "timestamp": "2024-03-15T07:30:00Z",
        "has_attachment": False,
        "thread_id": "t_customer",
        "gt_urgency": "critical",
        "gt_category": "support",
        "gt_department": "support",
        "gt_action": "escalate",
        "sla_minutes": 3,
        "depends_on": [],
        "unlocks": ["e016"],
        "is_phishing": False,
        "budget_cost": 1,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e016",
        "subject": "Re: Data loss — refund request $48,000",
        "sender": "cto@enterprise-client.com",
        "sender_name": "David Park (EnterpriseCo CTO)",
        "body": (
            "Following up from this morning. Your team called and confirmed the data loss. "
            "We are formally requesting a full refund of our annual contract ($48,000) "
            "plus compensation for 8 hours of engineer downtime (~$40,000). "
            "Please confirm receipt and expected timeline for resolution."
        ),
        "timestamp": "2024-03-15T11:00:00Z",
        "has_attachment": True,
        "thread_id": "t_customer",
        "gt_urgency": "high",
        "gt_category": "finance",
        "gt_department": "finance",
        "gt_action": "route",
        "sla_minutes": 10,
        "depends_on": ["e003"],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": {
            "if_email": "e003",
            "if_action_not": "escalate",
            "then_action": "escalate",
            "then_urgency": "critical",
        },
    },

    # ══════════════════════════════════════════
    # THREAD 4: Phishing chain — sophisticated attack
    # e004 looks like IT support, e011 is follow-up from same fake sender
    # ══════════════════════════════════════════
    {
        "id": "e004",
        "subject": "Action required: Verify your Microsoft 365 account",
        "sender": "it-support@company-helpdesk.net",  # NOT company.com — subtle
        "sender_name": "IT Support Team",
        "body": (
            "Dear user, We have detected unusual sign-in activity on your Microsoft 365 account. "
            "To secure your account, please verify your credentials within 24 hours by clicking: "
            "https://microsoft365-verify.company-helpdesk.net/login "
            "Failure to verify will result in account suspension. "
            "— IT Security Team"
        ),
        "timestamp": "2024-03-15T06:00:00Z",
        "has_attachment": False,
        "thread_id": "t_phishing",
        "gt_urgency": "low",
        "gt_category": "spam",
        "gt_department": None,
        "gt_action": "mark_spam",
        "sla_minutes": 999,
        "depends_on": [],
        "unlocks": ["e011"],
        "is_phishing": True,
        "phishing_tell": "sender domain is company-helpdesk.net not company.com",
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e011",
        "subject": "URGENT: Your account will be suspended in 2 hours",
        "sender": "it-support@company-helpdesk.net",
        "sender_name": "IT Support Team",
        "body": (
            "This is your final warning. Your Microsoft 365 account will be suspended "
            "in 2 hours unless you verify immediately. "
            "Click here: https://microsoft365-verify.company-helpdesk.net/urgent "
            "This is an automated security system message."
        ),
        "timestamp": "2024-03-15T10:00:00Z",
        "has_attachment": False,
        "thread_id": "t_phishing",
        "gt_urgency": "low",
        "gt_category": "spam",
        "gt_department": None,
        "gt_action": "mark_spam",
        "sla_minutes": 999,
        "depends_on": ["e004"],
        "unlocks": [],
        "is_phishing": True,
        "phishing_tell": "same fake domain, escalating pressure tactic",
        "budget_cost": 0,
        "reply_required_keywords": [],
        # If agent already caught e004 as spam, this is easier
        # If agent missed e004, this is still spam but harder to see
        "context_rule": None,
    },

    # ══════════════════════════════════════════
    # THREAD 5: Security vulnerability — time bomb
    # Must be escalated fast or SLA breach causes score penalty
    # ══════════════════════════════════════════
    {
        "id": "e005",
        "subject": "Bug bounty submission — SQL injection in /api/users",
        "sender": "researcher@bugcrowd.com",
        "sender_name": "BugCrowd Researcher",
        "body": (
            "Hello, I've discovered a critical SQL injection vulnerability in your /api/users endpoint. "
            "CVSS: 9.1. An attacker can dump the entire users table including password hashes. "
            "PoC attached. Per your bug bounty policy, I expect acknowledgment within 24 hours "
            "and a fix within 7 days. I will disclose publicly if not acknowledged."
        ),
        "timestamp": "2024-03-15T08:30:00Z",
        "has_attachment": True,
        "thread_id": "t_security",
        "gt_urgency": "critical",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "escalate",
        "sla_minutes": 4,
        "depends_on": [],
        "unlocks": ["e017"],
        "is_phishing": False,
        "budget_cost": 1,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e017",
        "subject": "Re: Bug bounty — 24hr acknowledgment deadline in 1 hour",
        "sender": "researcher@bugcrowd.com",
        "sender_name": "BugCrowd Researcher",
        "body": (
            "I submitted a critical SQL injection report 23 hours ago and have received no response. "
            "Per my disclosure policy, I will post this publicly in 1 hour if not acknowledged. "
            "This affects all user accounts. Please respond immediately."
        ),
        "timestamp": "2024-03-16T07:30:00Z",
        "has_attachment": False,
        "thread_id": "t_security",
        "gt_urgency": "critical",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "reply",
        "sla_minutes": 2,
        "depends_on": ["e005"],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": ["acknowledge", "investigating", "thank"],
        "context_rule": {
            "if_email": "e005",
            "if_action": "escalate",
            "then_action": "reply",
            "then_urgency": "critical",
            "then_reply_keywords": ["acknowledge", "investigating", "thank"],
        },
    },

    # ══════════════════════════════════════════
    # Standalone emails (Task 1 uses only e001)
    # ══════════════════════════════════════════
    {
        "id": "e006",
        "subject": "Invoice overdue 30 days — $24,500 — service suspension notice",
        "sender": "billing@cloudprovider.com",
        "sender_name": "CloudProvider Billing",
        "body": (
            "Invoice #CP-2024-0291 for $24,500 is now 30 days overdue. "
            "Per your service agreement, we will suspend your account in 48 hours "
            "unless payment is received. This would affect your production infrastructure. "
            "Please contact us immediately to arrange payment."
        ),
        "timestamp": "2024-03-14T09:00:00Z",
        "has_attachment": True,
        "thread_id": "t_finance",
        "gt_urgency": "high",
        "gt_category": "finance",
        "gt_department": "finance",
        "gt_action": "escalate",
        "sla_minutes": 15,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 1,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e007",
        "subject": "Re: Q1 OKR review — slides needed",
        "sender": "chief-of-staff@company.com",
        "sender_name": "Chief of Staff",
        "body": (
            "Reminder: Q1 OKR review is tomorrow at 2pm. "
            "Each team lead needs to submit their slides by 9am today to ops@company.com. "
            "Please forward to relevant team leads."
        ),
        "timestamp": "2024-03-14T08:00:00Z",
        "has_attachment": False,
        "thread_id": "t_internal",
        "gt_urgency": "medium",
        "gt_category": "internal",
        "gt_department": "hr",
        "gt_action": "route",
        "sla_minutes": 20,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e008",
        "subject": "Vendor proposal — AI transcription service $800/month",
        "sender": "sales@ai-transcribe.io",
        "sender_name": "AI Transcribe Sales",
        "body": (
            "Hello, we'd love to offer your team our AI meeting transcription service. "
            "Plans from $800/month for unlimited usage. "
            "Happy to set up a 15-minute demo call. Reply to schedule."
        ),
        "timestamp": "2024-03-12T14:00:00Z",
        "has_attachment": True,
        "thread_id": "t_vendor",
        "gt_urgency": "low",
        "gt_category": "sales",
        "gt_department": "sales",
        "gt_action": "defer",
        "sla_minutes": 999,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e010",
        "subject": "GDPR deletion request — 30 day compliance deadline",
        "sender": "privacy@eu-regulator.europa.eu",
        "sender_name": "EU Data Protection Authority",
        "body": (
            "We are writing on behalf of a data subject requesting deletion of all personal data "
            "under GDPR Article 17. Request reference: GDPR-2024-EU-88821. "
            "You have 30 days from receipt to comply. "
            "Failure to comply may result in fines of up to 4% of global annual turnover."
        ),
        "timestamp": "2024-03-15T08:00:00Z",
        "has_attachment": False,
        "thread_id": "t_gdpr",
        "gt_urgency": "high",
        "gt_category": "legal",
        "gt_department": "legal",
        "gt_action": "escalate",
        "sla_minutes": 20,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 1,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e012",
        "subject": "Congratulations — you've been selected for Forbes 30 Under 30!",
        "sender": "nominations@forbes-selections.net",  # not forbes.com
        "sender_name": "Forbes Nominations",
        "body": (
            "Dear Leader, Your company has been selected for the Forbes 30 Under 30 list! "
            "To confirm your spot, please provide your personal details and a $199 processing fee "
            "at: https://forbes-selections.net/confirm "
            "This offer expires in 48 hours."
        ),
        "timestamp": "2024-03-14T11:00:00Z",
        "has_attachment": False,
        "thread_id": "t_spam2",
        "gt_urgency": "low",
        "gt_category": "spam",
        "gt_department": None,
        "gt_action": "mark_spam",
        "sla_minutes": 999,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": True,
        "phishing_tell": "requests payment, fake domain forbes-selections.net not forbes.com",
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e013",
        "subject": "AWS cost anomaly — $67k spike in last 24 hours",
        "sender": "no-reply@aws.amazon.com",
        "sender_name": "AWS Cost Anomaly Detection",
        "body": (
            "Anomaly detected: Your AWS spend increased by $67,432 in the past 24 hours. "
            "Top drivers: EC2 (i3.16xlarge × 40 instances launched 18h ago), S3 egress. "
            "This may indicate a misconfiguration or unauthorized access. "
            "Review your Cost Explorer dashboard immediately."
        ),
        "timestamp": "2024-03-15T07:00:00Z",
        "has_attachment": False,
        "thread_id": "t_aws",
        "gt_urgency": "critical",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "escalate",
        "sla_minutes": 5,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 1,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e015",
        "subject": "New hire start date changed — Alex Kim now starting April 1st",
        "sender": "hr@company.com",
        "sender_name": "HR Team",
        "body": (
            "Please note that Alex Kim's start date has changed from March 18th to April 1st. "
            "Please update any onboarding preparations accordingly. "
            "Laptop provisioning and desk assignment should be rescheduled."
        ),
        "timestamp": "2024-03-14T16:00:00Z",
        "has_attachment": False,
        "thread_id": "t_hr",
        "gt_urgency": "low",
        "gt_category": "hr",
        "gt_department": "hr",
        "gt_action": "route",
        "sla_minutes": 999,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e018",
        "subject": "Partnership proposal — joint webinar series",
        "sender": "partnerships@techconference.io",
        "sender_name": "TechConference Partnerships",
        "body": (
            "We'd love to co-host a 4-part webinar series on AI in enterprise. "
            "Estimated reach: 15,000 attendees. Revenue share: 60/40. "
            "Let us know if you're interested in exploring further."
        ),
        "timestamp": "2024-03-13T12:00:00Z",
        "has_attachment": True,
        "thread_id": "t_partner",
        "gt_urgency": "low",
        "gt_category": "sales",
        "gt_department": "sales",
        "gt_action": "defer",
        "sla_minutes": 999,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e019",
        "subject": "Duplicate: Payment service down",
        "sender": "monitoring2@company.com",
        "sender_name": "Backup Alert System",
        "body": (
            "DUPLICATE ALERT — Same incident as previous alert. "
            "Payment service outage. INC-2024-0891. "
            "This is an automated duplicate from backup monitoring."
        ),
        "timestamp": "2024-03-15T09:02:00Z",
        "has_attachment": False,
        "thread_id": "t_outage",  # same thread as e001
        "gt_urgency": "low",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "archive",
        "sla_minutes": 999,
        "depends_on": ["e001"],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
    {
        "id": "e020",
        "subject": "Support ticket #88291 — user cannot login since yesterday",
        "sender": "support-system@company.com",
        "sender_name": "Support Ticket System",
        "body": (
            "Ticket #88291 has been open for 26 hours. Customer: Jane Doe (jane@acme.com). "
            "Issue: Cannot login — 2FA code not being received. "
            "Customer has sent 3 follow-up messages. SLA: 24 hours (BREACHED). "
            "Please assign to support team immediately."
        ),
        "timestamp": "2024-03-15T10:00:00Z",
        "has_attachment": False,
        "thread_id": "t_ticket",
        "gt_urgency": "high",
        "gt_category": "support",
        "gt_department": "support",
        "gt_action": "route",
        "sla_minutes": 8,
        "depends_on": [],
        "unlocks": [],
        "is_phishing": False,
        "budget_cost": 0,
        "reply_required_keywords": [],
        "context_rule": None,
    },
]


def get_task1_emails():
    """Task 1 (easy): Single clear critical email."""
    return [copy.deepcopy(e) for e in EMAILS if e["id"] == "e001"]


def get_task2_emails():
    """Task 2 (medium): 10 emails — mix of threads, phishing, dependencies."""
    ids = ["e001", "e002", "e003", "e004", "e005", "e006", "e007", "e010", "e012", "e013"]
    return [copy.deepcopy(e) for e in EMAILS if e["id"] in ids]


def get_task3_emails():
    """Task 3 (hard): All 20 emails — full cascades, budget, time pressure."""
    return [copy.deepcopy(e) for e in EMAILS]


def strip_ground_truth(email: Dict) -> Dict:
    """Remove GT fields before sending to agent."""
    hidden = {
        "gt_urgency", "gt_category", "gt_department", "gt_action",
        "sla_minutes", "depends_on", "unlocks", "is_phishing",
        "phishing_tell", "budget_cost", "reply_required_keywords", "context_rule",
    }
    return {k: v for k, v in email.items() if k not in hidden}
