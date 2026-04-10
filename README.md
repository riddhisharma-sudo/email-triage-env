---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - agent
  - email
  - nlp
  - real-world
license: mit
---

# 📧 Email Triage OpenEnv

A **real-world OpenEnv environment** simulating corporate inbox management.
An AI agent must classify, prioritize, route, and reply to realistic business emails
with increasing complexity across 3 tasks.

## Why Email Triage?

Email triage is one of the most time-consuming knowledge-work tasks:
- Knowledge workers spend **28% of their workday** on email (McKinsey)
- Misrouted emails cost enterprises significant time and money
- Urgency misjudgment causes SLA breaches and customer churn

This environment provides a **realistic, graded signal** for training agents
to handle real inbox management.

---

## Environment Description

| Property | Value |
|----------|-------|
| **Domain** | Corporate email triage |
| **Tasks** | 3 (easy → hard) |
| **Emails** | 1 / 10 / 25 per task |
| **Actions** | classify, route, reply, archive, escalate, mark_spam, defer, flag |
| **Reward** | Continuous [0.0, 1.0] per email |
| **Episode** | Done when inbox empty or max steps reached |

---

## Observation Space

```json
{
  "task_id": "string",
  "step_number": "int",
  "inbox": [
    {
      "id": "string",
      "subject": "string",
      "sender": "string",
      "sender_name": "string",
      "body": "string",
      "timestamp": "ISO8601",
      "has_attachment": "bool",
      "thread_id": "string"
    }
  ],
  "current_email": "EmailObject | null",
  "processed_count": "int",
  "pending_count": "int",
  "sla_breaches": "int",
  "escalated_count": "int",
  "done": "bool",
  "message": "string"
}
```

## Action Space

```json
{
  "action_type": "classify|route|reply|archive|escalate|mark_spam|defer|flag",
  "email_id": "string",
  "urgency": "critical|high|medium|low",
  "category": "support|sales|internal|spam|legal|finance|hr|engineering",
  "department": "string",
  "reply_text": "string | null",
  "reason": "string | null"
}
```

---

## Tasks

### Task 1 — `classify_urgency` (Easy)
**Goal:** Given a single email, classify its urgency level and category.

- Inbox size: 1 email
- Max steps: 5
- Graded on: urgency accuracy (60%), category accuracy (40%)
- Expected baseline score: ~0.70

### Task 2 — `triage_and_route` (Medium)
**Goal:** Process 10 mixed emails — classify each, route to correct department, and take appropriate action.

- Inbox size: 10 emails
- Max steps: 20
- Graded on: urgency (30%), routing (30%), action choice (30%), reply quality (10%)
- Expected baseline score: ~0.50

### Task 3 — `inbox_zero` (Hard)
**Goal:** Manage a realistic inbox of 25 emails with:
- Duplicate thread detection
- SLA deadline awareness
- Spam filtering
- Legal/finance escalations
- Reply drafting for support queries

- Inbox size: 25 emails
- Max steps: 60
- Graded on: urgency (20%), routing (20%), action (25%), reply quality (15%), duplicate detection (20%)
- SLA breach penalty: -0.05 per breach
- Expected baseline score: ~0.35

---

## Reward Function

Each email action yields a composite reward:

```
reward = w_urgency × urgency_score
       + w_routing × routing_score
       + w_action  × action_score
       + w_reply   × reply_score
       + w_dup     × duplicate_score
```

Weights vary by task. All component scores ∈ [0, 1].

**Partial credit** is awarded for near-misses (e.g. predicting "high" for a "critical" email scores 0.5 rather than 0).

**Penalties:**
- Archiving/deferring a critical email: action_score can go negative (−0.2)
- SLA breach: −0.05 per breach from final score

---

## Setup & Usage

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Local

```bash
pip install -r requirements.txt
touch data/__init__.py
python server.py
```

### API

```bash
# Reset for a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "classify_urgency"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session_id>",
    "action": {
      "action_type": "escalate",
      "email_id": "e001",
      "urgency": "critical",
      "category": "engineering",
      "department": "engineering",
      "reason": "Production outage affecting all customers"
    }
  }'

# Get state
curl http://localhost:7860/state?session_id=<session_id>

# Validate spec compliance
curl http://localhost:7860/validate
```

### Run Baseline

```bash
# Start server in background
python server.py &

# Run inference
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

---

## Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Score | Status |
|------|-------|--------|
| classify_urgency | ~0.72 | ✅ PASS |
| triage_and_route | ~0.51 | ✅ PASS |
| inbox_zero | ~0.38 | ✅ PASS |
| **Overall** | **~0.54** | ✅ |

---

## OpenEnv Spec Compliance

- ✅ Typed Pydantic models: `Observation`, `Action`, `Reward`, `State`
- ✅ `reset()` → clean state + initial observation
- ✅ `step(action)` → observation, reward ∈ [0,1], done, info
- ✅ `state()` → full episode state
- ✅ `openenv.yaml` with metadata
- ✅ `GET /validate` endpoint
- ✅ 3+ tasks with graders (easy → medium → hard)
- ✅ Partial reward signal (not just binary)
- ✅ Dockerfile builds + HF Space deploys

---

## License

MIT
