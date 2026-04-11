"""
inference.py — Email Triage OpenEnv v2 Baseline
=================================================
Agent is aware of:
  - SLA countdowns (prioritises urgent emails)
  - Thread dependencies (processes unlocked emails only)
  - Escalation budget (conserves escalations)
  - Phishing signals (checks sender domain)
  - Context from prior actions

Required env vars:
  HF_TOKEN, API_BASE_URL, MODEL_NAME, ENV_BASE_URL
"""

import json
import os
import time
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "email-triage-env-v2"
SUCCESS_THRESHOLD = 0.40


# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ── Env API ───────────────────────────────────────────────────────────────────

def env_reset(task_id):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(session_id, action):
    r = requests.post(f"{ENV_BASE_URL}/step", json={"session_id": session_id, "action": action}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_score(session_id):
    r = requests.get(f"{ENV_BASE_URL}/score", params={"session_id": session_id}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage agent. You process corporate emails one at a time.

IMPORTANT RULES:
1. Only act on UNLOCKED emails (is_unlocked=true). Locked emails need their dependencies processed first.
2. Prioritise emails with LOW sla_steps_remaining — they breach SLA soon.
3. You have a LIMITED escalation budget. Reserve escalations for truly critical issues.
4. Check sender domains carefully — phishing emails often use fake domains similar to real ones.
5. For reply actions, write a substantive professional reply (50+ words) with relevant keywords.
6. Duplicate emails (same thread, same issue) should be archived.

ACTIONS available: escalate, route, reply, archive, mark_spam, defer, flag

Respond ONLY with valid JSON:
{
  "action_type": "<action>",
  "email_id": "<id>",
  "urgency": "<critical|high|medium|low>",
  "category": "<support|sales|internal|spam|legal|finance|hr|engineering>",
  "department": "<support|sales|hr|legal|finance|engineering|internal>",
  "reply_text": "<substantive reply if action=reply, else null>",
  "reason": "<one sentence explanation>"
}
""").strip()


def build_prompt(obs: dict, history: list) -> str:
    inbox = obs.get("inbox", [])
    current = obs.get("current_email")
    budget = obs.get("escalation_budget", 5)
    sla_breaches = obs.get("sla_breaches", 0)

    # Show unlocked emails sorted by SLA urgency
    unlocked = [e for e in inbox if e.get("is_unlocked")]
    locked_count = len([e for e in inbox if not e.get("is_unlocked")])

    email_list = "\n".join(
        f"  [{e['id']}] SLA:{e.get('sla_steps_remaining','?')} | {e['subject'][:60]} | from: {e['sender']}"
        for e in unlocked[:8]
    )

    current_block = ""
    if current and current.get("is_unlocked"):
        current_block = textwrap.dedent(f"""
        FOCUS EMAIL:
        ID: {current['id']}
        Subject: {current['subject']}
        From: {current['sender_name']} <{current['sender']}>
        SLA steps remaining: {current.get('sla_steps_remaining', '?')}
        Body:
        {current['body']}
        """).strip()

    recent_history = "\n".join(
        f"  Step {h['step']}: [{h['email_id']}] → {h['action']} (reward={h['reward']:.2f})"
        for h in history[-5:]
    ) or "  None yet"

    return textwrap.dedent(f"""
    Step {obs['step_number']} | Budget: {budget} escalations left | SLA breaches: {sla_breaches}
    Pending: {obs['pending_count']} ({locked_count} locked, waiting on dependencies)

    {current_block}

    UNLOCKED INBOX:
    {email_list or '  (none)'}

    RECENT ACTIONS:
    {recent_history}

    Act on the FOCUS EMAIL. Respond with JSON.
    """).strip()


def get_action(client, obs, history):
    prompt = build_prompt(obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        inbox = obs.get("inbox", [])
        unlocked = [e for e in inbox if e.get("is_unlocked")]
        email_id = unlocked[0]["id"] if unlocked else "e001"
        return {"action_type": "archive", "email_id": email_id,
                "urgency": "low", "category": "internal", "department": "hr",
                "reply_text": None, "reason": "fallback"}


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(client, task_id):
    max_steps = {"classify_urgency": 5, "triage_and_route": 20, "inbox_zero": 50}[task_id]
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards, steps_taken, score, success = [], 0, 0.0, False
    session_id, history = None, []

    try:
        resp = env_reset(task_id)
        session_id = resp["session_id"]
        obs = resp["observation"]

        for step in range(1, max_steps + 1):
            if obs.get("done"):
                break
            inbox = obs.get("inbox", [])
            if not any(e.get("is_unlocked") for e in inbox):
                break

            action_dict = get_action(client, obs, history)
            action_str = json.dumps({k: v for k, v in action_dict.items() if v is not None and k != "reply_text"})

            try:
                step_resp = env_step(session_id, action_dict)
                reward = step_resp["reward"]
                done = step_resp["done"]
                obs = step_resp["observation"]
                error = step_resp.get("info", {}).get("error")
            except Exception as e:
                reward, done, error = 0.0, True, str(e)

            rewards.append(reward)
            steps_taken = step
            history.append({"step": step, "email_id": action_dict.get("email_id"), "action": action_dict.get("action_type"), "reward": reward})
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        if session_id:
            try:
                score = env_score(session_id)["final_score"]
            except Exception:
                score = min(max(sum(rewards) / max(len(rewards), 1), 0.001), 0.999)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["classify_urgency", "triage_and_route", "inbox_zero"]
    results = {}

    for task_id in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        results[task_id] = run_task(client, task_id)
        time.sleep(1)

    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for task_id, s in results.items():
        status = "PASS" if s >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  {task_id:30s}: {s:.3f}  [{status}]", flush=True)
    overall = sum(results.values()) / len(results)
    print(f"  {'OVERALL':30s}: {overall:.3f}", flush=True)


if __name__ == "__main__":
    main()
