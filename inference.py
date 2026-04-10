"""
inference.py — Email Triage OpenEnv Baseline
=============================================

Runs an LLM agent (via OpenAI-compatible client) against all 3 tasks
and emits mandatory stdout format: [START], [STEP], [END].

Required env vars:
  HF_TOKEN      — API key
  API_BASE_URL  — inference endpoint (default: HF router)
  MODEL_NAME    — model identifier

Usage:
  python inference.py
"""

import json
import os
import textwrap
import time
import uuid
from typing import List, Optional

import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "email-triage-env"

MAX_STEPS = {
    "classify_urgency": 5,
    "triage_and_route": 20,
    "inbox_zero": 60,
}
SUCCESS_THRESHOLD = 0.4  # score >= this → success


# ── Logging (mandatory format) ────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment API client ────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(session_id: str, action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"session_id": session_id, "action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_score(session_id: str) -> dict:
    r = requests.get(f"{ENV_BASE_URL}/score", params={"session_id": session_id}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── LLM agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage assistant. You will be given emails from a corporate inbox.

For each email you must decide the correct action. Respond ONLY with valid JSON in this exact format:
{
  "action_type": "<classify|route|reply|archive|escalate|mark_spam|defer|flag>",
  "email_id": "<the email id>",
  "urgency": "<critical|high|medium|low>",
  "category": "<support|sales|internal|spam|legal|finance|hr|engineering>",
  "department": "<support|sales|hr|legal|finance|engineering|internal>",
  "reply_text": "<a brief professional reply if action_type is reply, else null>",
  "reason": "<one sentence explanation>"
}

Action selection guide:
- escalate: critical/high urgency issues needing immediate management attention
- route: send to correct department for handling
- reply: directly respond to sender (for support/partnership queries)
- archive: low priority informational emails, completed threads
- mark_spam: unsolicited commercial email, phishing, prizes
- defer: non-urgent emails that can wait >48h
- flag: suspicious or ambiguous — needs human review
- classify: ONLY for task 1 (single email classification)

For DUPLICATE emails (same thread, same issue), use archive.
CRITICAL urgency emails must NEVER be archived or deferred.
""").strip()


def build_user_prompt(obs: dict, step: int) -> str:
    inbox = obs.get("inbox", [])
    current = obs.get("current_email")

    if current:
        email_block = textwrap.dedent(f"""
        CURRENT EMAIL TO PROCESS:
        ID: {current['id']}
        Subject: {current['subject']}
        From: {current['sender_name']} <{current['sender']}>
        Date: {current['timestamp']}
        Has attachment: {current['has_attachment']}
        Body:
        {current['body']}
        """).strip()
    else:
        email_block = "No email currently selected."

    pending_subjects = "\n".join(
        f"  [{e['id']}] {e['subject']} — from {e['sender_name']}"
        for e in inbox[:10]
    )

    return textwrap.dedent(f"""
    Step {step} | Inbox: {obs['pending_count']} remaining | Processed: {obs['processed_count']}

    {email_block}

    REMAINING INBOX PREVIEW:
    {pending_subjects if pending_subjects else "  (empty)"}

    Respond with JSON action for the CURRENT EMAIL (id: {current['id'] if current else 'N/A'}).
    """).strip()


def get_agent_action(client: OpenAI, obs: dict, step: int, history: list) -> dict:
    """Ask the LLM what action to take, parse JSON response."""
    user_prompt = build_user_prompt(obs, step)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Include last 3 history entries for context
    for h in history[-3:]:
        messages.append({"role": "user", "content": h["prompt"]})
        messages.append({"role": "assistant", "content": h["response"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action_dict = json.loads(raw.strip())
        history.append({"prompt": user_prompt, "response": raw})
        return action_dict
    except Exception as e:
        print(f"[DEBUG] LLM parse error: {e}", flush=True)
        # Fallback action
        inbox = obs.get("inbox", [])
        email_id = obs["current_email"]["id"] if obs.get("current_email") else (inbox[0]["id"] if inbox else "e001")
        return {
            "action_type": "archive",
            "email_id": email_id,
            "urgency": "low",
            "category": "internal",
            "department": "hr",
            "reply_text": None,
            "reason": "fallback action due to parse error",
        }


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    session_id = None
    history = []

    try:
        reset_resp = env_reset(task_id)
        session_id = reset_resp["session_id"]
        obs = reset_resp["observation"]
        max_steps = MAX_STEPS[task_id]

        for step in range(1, max_steps + 1):
            if obs.get("done") or not obs.get("inbox"):
                break

            action_dict = get_agent_action(client, obs, step, history)
            action_str = json.dumps({k: v for k, v in action_dict.items() if v is not None})

            try:
                step_resp = env_step(session_id, action_dict)
                reward = step_resp["reward"]
                done = step_resp["done"]
                obs = step_resp["observation"]
                error = step_resp.get("info", {}).get("error")
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Get final score from env
        if session_id:
            try:
                score_resp = env_score(session_id)
                score = score_resp["final_score"]
            except Exception:
                score = sum(rewards) / max(len(rewards), 1)
                score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = ["classify_urgency", "triage_and_route", "inbox_zero"]
    results = {}

    for task_id in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_task(client, task_id)
        results[task_id] = score
        time.sleep(1)  # small pause between tasks

    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for task_id, score in results.items():
        status = "PASS" if score >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  {task_id:30s}: {score:.3f}  [{status}]", flush=True)
    overall = sum(results.values()) / len(results)
    print(f"  {'OVERALL':30s}: {overall:.3f}", flush=True)


if __name__ == "__main__":
    main()
