"""
Realistic email dataset for the Email Triage environment.
Emails include metadata needed for grading (ground truth labels).
"""
from typing import List, Dict, Any
import copy

# Ground truth labels used by graders (not exposed to agent in observation)
EMAIL_TEMPLATES: List[Dict[str, Any]] = [
    # --- CRITICAL emails ---
    {
        "id": "e001",
        "subject": "URGENT: Production database down — all services affected",
        "sender": "ops-alerts@company.com",
        "sender_name": "Ops Monitoring",
        "body": (
            "ALERT: Primary production database (prod-db-01) is unreachable as of 14:32 UTC. "
            "All customer-facing services are returning 503. "
            "On-call engineer has not responded in 15 minutes. "
            "Estimated customer impact: ~12,000 active sessions. "
            "Please escalate immediately to CTO and incident commander."
        ),
        "timestamp": "2024-03-15T14:47:00Z",
        "has_attachment": False,
        "thread_id": "t001",
        "gt_urgency": "critical",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "escalate",
        "sla_hours": 0.25,  # 15-minute SLA
    },
    {
        "id": "e002",
        "subject": "Legal notice — cease and desist re: patent infringement",
        "sender": "legal@externaLawFirm.com",
        "sender_name": "Morrison & Foerster LLP",
        "body": (
            "Dear Sir/Madam, We represent TechCorp Industries Inc. and write to notify you "
            "of potential infringement of US Patent #9,876,543. You have 14 days to respond "
            "with a written acknowledgment or we will file for injunctive relief. "
            "Please have your legal counsel contact us immediately."
        ),
        "timestamp": "2024-03-15T09:15:00Z",
        "has_attachment": True,
        "thread_id": "t002",
        "gt_urgency": "critical",
        "gt_category": "legal",
        "gt_department": "legal",
        "gt_action": "escalate",
        "sla_hours": 4,
    },
    # --- HIGH urgency ---
    {
        "id": "e003",
        "subject": "Enterprise client threatening to churn — $2.4M ARR at risk",
        "sender": "j.smith@bigclient.com",
        "sender_name": "James Smith",
        "body": (
            "Hi, I'm the CTO at BigClient Co. We've been experiencing repeated API timeout "
            "issues for the past 3 days and your support team has not resolved the issue. "
            "If this isn't fixed by EOD Friday, we'll be terminating our contract. "
            "We pay $200k/month and expect better service."
        ),
        "timestamp": "2024-03-15T11:30:00Z",
        "has_attachment": False,
        "thread_id": "t003",
        "gt_urgency": "high",
        "gt_category": "support",
        "gt_department": "sales",
        "gt_action": "escalate",
        "sla_hours": 2,
    },
    {
        "id": "e004",
        "subject": "Security vulnerability report — RCE in login endpoint",
        "sender": "bugbounty@hackerone.com",
        "sender_name": "HackerOne Disclosure",
        "body": (
            "A security researcher has responsibly disclosed a critical RCE vulnerability "
            "in your /api/v2/login endpoint. CVSS score: 9.8 (Critical). "
            "The researcher requests acknowledgment within 24 hours per your bug bounty policy. "
            "Full PoC and reproduction steps attached."
        ),
        "timestamp": "2024-03-15T08:00:00Z",
        "has_attachment": True,
        "thread_id": "t004",
        "gt_urgency": "high",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "escalate",
        "sla_hours": 4,
    },
    {
        "id": "e005",
        "subject": "Q4 board presentation — slides needed by 5pm TODAY",
        "sender": "ceo@company.com",
        "sender_name": "Sarah Johnson (CEO)",
        "body": (
            "Team, I need the Q4 financial summary slides for the board meeting at 6pm. "
            "Finance should have them ready. Please send to board-materials@company.com "
            "and cc me by 5pm. This is a hard deadline."
        ),
        "timestamp": "2024-03-15T13:00:00Z",
        "has_attachment": False,
        "thread_id": "t005",
        "gt_urgency": "high",
        "gt_category": "finance",
        "gt_department": "finance",
        "gt_action": "route",
        "sla_hours": 4,
    },
    # --- MEDIUM urgency ---
    {
        "id": "e006",
        "subject": "New feature request — dark mode for mobile app",
        "sender": "customer@example.com",
        "sender_name": "Patricia L.",
        "body": (
            "Hello, I love your product but I'd really like a dark mode option on the mobile app. "
            "My eyes get strained using it at night. A lot of my colleagues feel the same way. "
            "Could this be considered for a future release? Thank you!"
        ),
        "timestamp": "2024-03-14T16:20:00Z",
        "has_attachment": False,
        "thread_id": "t006",
        "gt_urgency": "medium",
        "gt_category": "support",
        "gt_department": "engineering",
        "gt_action": "route",
        "sla_hours": 48,
    },
    {
        "id": "e007",
        "subject": "Invoice #INV-2024-0312 — payment 15 days overdue",
        "sender": "billing@vendor.com",
        "sender_name": "Acme Supplies Billing",
        "body": (
            "This is a reminder that Invoice #INV-2024-0312 for $8,750.00 was due on March 1st "
            "and remains unpaid. Please process payment at your earliest convenience to avoid "
            "a late fee of 1.5% per month. Reply to arrange payment."
        ),
        "timestamp": "2024-03-15T10:00:00Z",
        "has_attachment": True,
        "thread_id": "t007",
        "gt_urgency": "medium",
        "gt_category": "finance",
        "gt_department": "finance",
        "gt_action": "route",
        "sla_hours": 24,
    },
    {
        "id": "e008",
        "subject": "Interview schedule confirmation — Senior Engineer candidate",
        "sender": "recruiting@company.com",
        "sender_name": "HR Recruiting",
        "body": (
            "Hi, We'd like to confirm the interview loop for candidate Alex Chen scheduled for "
            "March 20th. Interviewers: John (10am), Maria (11am), Tech Panel (2-4pm). "
            "Please confirm your availability or suggest changes by March 17th."
        ),
        "timestamp": "2024-03-15T09:45:00Z",
        "has_attachment": False,
        "thread_id": "t008",
        "gt_urgency": "medium",
        "gt_category": "hr",
        "gt_department": "hr",
        "gt_action": "route",
        "sla_hours": 48,
    },
    {
        "id": "e009",
        "subject": "Re: API rate limits — clarification needed",
        "sender": "dev@partnerco.com",
        "sender_name": "Dev Team at PartnerCo",
        "body": (
            "Following up on our earlier thread. We're integrating with your API and hitting "
            "rate limits at around 500 req/min even though our plan allows 1000. "
            "Can your support team look into our account (ID: ACC-88821)? "
            "This is blocking our go-live scheduled for next week."
        ),
        "timestamp": "2024-03-14T14:00:00Z",
        "has_attachment": False,
        "thread_id": "t009",
        "gt_urgency": "medium",
        "gt_category": "support",
        "gt_department": "support",
        "gt_action": "reply",
        "sla_hours": 24,
    },
    {
        "id": "e010",
        "subject": "Quarterly All-Hands — agenda items wanted",
        "sender": "comms@company.com",
        "sender_name": "Internal Comms",
        "body": (
            "Hi everyone, Our Q1 All-Hands is on March 28th. If you have agenda items you'd "
            "like included, please submit them via the form by March 22nd. "
            "Topics: product updates, team wins, OKR review."
        ),
        "timestamp": "2024-03-13T11:00:00Z",
        "has_attachment": False,
        "thread_id": "t010",
        "gt_urgency": "low",
        "gt_category": "internal",
        "gt_department": "hr",
        "gt_action": "archive",
        "sla_hours": 168,
    },
    # --- LOW urgency / spam ---
    {
        "id": "e011",
        "subject": "You've been selected! Claim your $500 Amazon gift card",
        "sender": "noreply@prize-winner-2024.net",
        "sender_name": "Prize Department",
        "body": (
            "Congratulations! You are our lucky winner. Click here to claim your $500 Amazon gift card. "
            "Offer expires in 24 hours. No purchase necessary. Click the link below NOW!"
        ),
        "timestamp": "2024-03-15T07:30:00Z",
        "has_attachment": False,
        "thread_id": "t011",
        "gt_urgency": "low",
        "gt_category": "spam",
        "gt_department": None,
        "gt_action": "mark_spam",
        "sla_hours": 999,
    },
    {
        "id": "e012",
        "subject": "Monthly newsletter — March 2024",
        "sender": "newsletter@techblog.io",
        "sender_name": "TechBlog Monthly",
        "body": (
            "This month's top stories: AI trends in 2024, top frameworks for backend dev, "
            "an interview with Linus Torvalds, and 10 tools every developer should know. "
            "Unsubscribe link at bottom."
        ),
        "timestamp": "2024-03-01T08:00:00Z",
        "has_attachment": False,
        "thread_id": "t012",
        "gt_urgency": "low",
        "gt_category": "spam",
        "gt_department": None,
        "gt_action": "archive",
        "sla_hours": 999,
    },
    {
        "id": "e013",
        "subject": "Re: Re: Re: Team lunch on Friday?",
        "sender": "colleague@company.com",
        "sender_name": "Mike Chen",
        "body": (
            "Sounds good! I'll make a reservation at that Thai place. Noon works for me. "
            "Let me know if anyone has dietary restrictions."
        ),
        "timestamp": "2024-03-14T17:00:00Z",
        "has_attachment": False,
        "thread_id": "t013",
        "gt_urgency": "low",
        "gt_category": "internal",
        "gt_department": None,
        "gt_action": "archive",
        "sla_hours": 999,
    },
    # --- More for task 3 (hard) ---
    {
        "id": "e014",
        "subject": "DUPLICATE: Production database down — all services affected",
        "sender": "on-call@company.com",
        "sender_name": "On-Call System",
        "body": (
            "DUPLICATE ALERT: Same incident as e001. Production DB outage auto-escalation. "
            "This is a duplicate notification. Original ticket: INC-2024-0891."
        ),
        "timestamp": "2024-03-15T14:50:00Z",
        "has_attachment": False,
        "thread_id": "t001",  # same thread as e001
        "gt_urgency": "critical",
        "gt_category": "engineering",
        "gt_department": "engineering",
        "gt_action": "archive",  # should be detected as duplicate
        "sla_hours": 0.25,
        "is_duplicate": True,
        "duplicate_of": "e001",
    },
    {
        "id": "e015",
        "subject": "Contract renewal — 90-day notice required",
        "sender": "contracts@bigvendor.com",
        "sender_name": "BigVendor Contracts",
        "body": (
            "Per section 12.3 of your master service agreement, we're providing 90-day notice "
            "that your current contract expires June 15th. Please confirm renewal intent or "
            "termination by March 22nd to avoid service interruption. Contract value: $180k/yr."
        ),
        "timestamp": "2024-03-14T12:00:00Z",
        "has_attachment": True,
        "thread_id": "t015",
        "gt_urgency": "high",
        "gt_category": "legal",
        "gt_department": "legal",
        "gt_action": "escalate",
        "sla_hours": 48,
    },
    {
        "id": "e016",
        "subject": "Sales pipeline update — March forecast",
        "sender": "sales-ops@company.com",
        "sender_name": "Sales Ops",
        "body": (
            "Hi team, attaching the March sales pipeline update. Total pipeline: $4.2M. "
            "Top deals: Acme Corp ($800k, 80% close prob), TechStartup ($250k, 60%). "
            "Please review before Friday's sales call."
        ),
        "timestamp": "2024-03-14T09:00:00Z",
        "has_attachment": True,
        "thread_id": "t016",
        "gt_urgency": "medium",
        "gt_category": "sales",
        "gt_department": "sales",
        "gt_action": "route",
        "sla_hours": 48,
    },
    {
        "id": "e017",
        "subject": "GDPR data deletion request — 30-day deadline",
        "sender": "privacy@externalregulator.eu",
        "sender_name": "EU Data Protection",
        "body": (
            "We are writing on behalf of a data subject requesting deletion of all personal data "
            "held by your company under GDPR Article 17. You have 30 days from the date of this "
            "notice to comply. Request ID: GDPR-2024-09821. Failure to comply may result in fines."
        ),
        "timestamp": "2024-03-15T08:30:00Z",
        "has_attachment": False,
        "thread_id": "t017",
        "gt_urgency": "high",
        "gt_category": "legal",
        "gt_department": "legal",
        "gt_action": "escalate",
        "sla_hours": 24,
    },
    {
        "id": "e018",
        "subject": "New hire onboarding checklist — Alex starts Monday",
        "sender": "hr@company.com",
        "sender_name": "HR Team",
        "body": (
            "Alex Kim starts on Monday March 18th. Please ensure: laptop provisioned, "
            "Slack/email access set up, desk assigned (Floor 3, seat 22), "
            "buddy assigned. Manager please send welcome email by Friday."
        ),
        "timestamp": "2024-03-14T16:00:00Z",
        "has_attachment": False,
        "thread_id": "t018",
        "gt_urgency": "medium",
        "gt_category": "hr",
        "gt_department": "hr",
        "gt_action": "route",
        "sla_hours": 48,
    },
    {
        "id": "e019",
        "subject": "AWS bill spike — $47k over budget this month",
        "sender": "billing-alerts@aws.amazon.com",
        "sender_name": "AWS Billing",
        "body": (
            "Your AWS monthly spend has exceeded your budget alert threshold. "
            "Current month spend: $127,450 vs budget of $80,000. "
            "Top cost drivers: EC2 ($54k), RDS ($31k), Data Transfer ($22k). "
            "Review your cost explorer dashboard for details."
        ),
        "timestamp": "2024-03-15T06:00:00Z",
        "has_attachment": False,
        "thread_id": "t019",
        "gt_urgency": "high",
        "gt_category": "finance",
        "gt_department": "engineering",
        "gt_action": "escalate",
        "sla_hours": 8,
    },
    {
        "id": "e020",
        "subject": "Phishing attempt — do not click any links",
        "sender": "security@company.com",
        "sender_name": "Security Team",
        "body": (
            "SECURITY ALERT: We have detected a phishing campaign targeting company employees. "
            "You may receive emails purporting to be from 'IT Support' asking you to reset your password. "
            "Do NOT click any links. Forward suspicious emails to security@company.com immediately."
        ),
        "timestamp": "2024-03-15T10:30:00Z",
        "has_attachment": False,
        "thread_id": "t020",
        "gt_urgency": "high",
        "gt_category": "internal",
        "gt_department": "engineering",
        "gt_action": "route",
        "sla_hours": 1,
    },
    {
        "id": "e021",
        "subject": "Office party planning — suggestions welcome",
        "sender": "fun-committee@company.com",
        "sender_name": "Fun Committee",
        "body": (
            "Hey everyone! We're planning a Q2 office party. Themes being considered: "
            "80s retro, beach party, or game night. Vote using the form link by March 20th! "
            "Budget: $50/person."
        ),
        "timestamp": "2024-03-12T14:00:00Z",
        "has_attachment": False,
        "thread_id": "t021",
        "gt_urgency": "low",
        "gt_category": "internal",
        "gt_department": None,
        "gt_action": "archive",
        "sla_hours": 999,
    },
    {
        "id": "e022",
        "subject": "Re: Proposal — marketing partnership opportunity",
        "sender": "partner@marketingco.com",
        "sender_name": "Sarah at MarketingCo",
        "body": (
            "Thanks for the call last week! I've attached our revised proposal for a co-marketing "
            "campaign. Estimated reach: 2M impressions/month. Investment: $15k for 3 months. "
            "Happy to hop on a call to discuss further."
        ),
        "timestamp": "2024-03-13T15:00:00Z",
        "has_attachment": True,
        "thread_id": "t022",
        "gt_urgency": "medium",
        "gt_category": "sales",
        "gt_department": "sales",
        "gt_action": "route",
        "sla_hours": 72,
    },
    {
        "id": "e023",
        "subject": "Customer complaint — data lost during migration",
        "sender": "angry.customer@bizcorp.com",
        "sender_name": "Robert T.",
        "body": (
            "I am absolutely furious. Your team's migration last night deleted 3 months of our "
            "project data. This is unacceptable. I need a call with your CEO TODAY and a written "
            "explanation. We will be consulting our lawyers if this is not resolved immediately. "
            "Account: BIZ-00429."
        ),
        "timestamp": "2024-03-15T09:00:00Z",
        "has_attachment": False,
        "thread_id": "t023",
        "gt_urgency": "critical",
        "gt_category": "support",
        "gt_department": "support",
        "gt_action": "escalate",
        "sla_hours": 1,
    },
    {
        "id": "e024",
        "subject": "LinkedIn: You have 14 new connection requests",
        "sender": "notifications@linkedin.com",
        "sender_name": "LinkedIn",
        "body": (
            "You have 14 new connection requests waiting. Check your network updates: "
            "Alice B. wants to connect, Bob C. wants to connect... and 12 more."
        ),
        "timestamp": "2024-03-15T08:00:00Z",
        "has_attachment": False,
        "thread_id": "t024",
        "gt_urgency": "low",
        "gt_category": "spam",
        "gt_department": None,
        "gt_action": "archive",
        "sla_hours": 999,
    },
    {
        "id": "e025",
        "subject": "PTO request — April 15-19 — needs approval",
        "sender": "employee@company.com",
        "sender_name": "Jordan Park",
        "body": (
            "Hi, I'd like to request PTO from April 15-19 (5 days). "
            "I've checked with the team and coverage is arranged. "
            "Please approve in the HR system or let me know if there are any conflicts."
        ),
        "timestamp": "2024-03-14T11:00:00Z",
        "has_attachment": False,
        "thread_id": "t025",
        "gt_urgency": "low",
        "gt_category": "hr",
        "gt_department": "hr",
        "gt_action": "route",
        "sla_hours": 168,
    },
]


def get_task1_email() -> Dict[str, Any]:
    """Single email for easy task — classify urgency."""
    return copy.deepcopy(EMAIL_TEMPLATES[0])  # critical email


def get_task2_emails() -> List[Dict[str, Any]]:
    """10 emails for medium task — triage and route."""
    return copy.deepcopy(EMAIL_TEMPLATES[:10])


def get_task3_emails() -> List[Dict[str, Any]]:
    """25 emails for hard task — inbox zero."""
    return copy.deepcopy(EMAIL_TEMPLATES)


def strip_ground_truth(email: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ground truth fields before sending to agent."""
    agent_view = {k: v for k, v in email.items()
                  if not k.startswith("gt_") and k not in ("sla_hours", "is_duplicate", "duplicate_of")}
    return agent_view
