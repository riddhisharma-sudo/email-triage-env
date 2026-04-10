"""
FastAPI server for the Email Triage OpenEnv environment.
Implements full OpenEnv spec including /metadata, /schema, /mcp endpoints.
"""
import os
import uuid
from typing import Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Add parent to path so env/data are importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Action, EmailTriageEnv, Observation, Reward, State

app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world email triage environment for AI agent training and evaluation.",
    version="1.0.0",
)

_sessions: Dict[str, EmailTriageEnv] = {}


# ── Request/Response schemas ──────────────────

class ResetRequest(BaseModel):
    task_id: str = "classify_urgency"
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: dict
    info: dict = {}


class StepRequest(BaseModel):
    session_id: str
    action: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    reward_breakdown: dict
    reward_feedback: str
    done: bool
    info: dict = {}


# ── Core OpenEnv endpoints ────────────────────

@app.get("/health")
def health():
    # Must return "status": "healthy" for openenv-core validator
    return {"status": "healthy", "env": "email-triage-env", "version": "1.0.0"}


@app.get("/metadata")
def metadata():
    return {
        "name": "email-triage-env",
        "description": (
            "A real-world email triage environment where an AI agent classifies, "
            "routes, replies to, and manages a realistic corporate inbox across 3 tasks "
            "of increasing difficulty."
        ),
        "version": "1.0.0",
        "tasks": ["classify_urgency", "triage_and_route", "inbox_zero"],
        "author": "openenv-submission",
        "tags": ["email", "triage", "nlp", "productivity", "real-world"],
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["classify", "route", "reply", "archive", "escalate", "mark_spam", "defer", "flag"],
                },
                "email_id": {"type": "string"},
                "urgency": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                "category": {"type": "string"},
                "department": {"type": "string"},
                "reply_text": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["action_type", "email_id"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step_number": {"type": "integer"},
                "inbox": {"type": "array"},
                "current_email": {"type": "object"},
                "processed_count": {"type": "integer"},
                "pending_count": {"type": "integer"},
                "sla_breaches": {"type": "integer"},
                "done": {"type": "boolean"},
                "message": {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step_number": {"type": "integer"},
                "episode_id": {"type": "string"},
                "processed_emails": {"type": "array"},
                "pending_emails": {"type": "array"},
                "total_reward": {"type": "number"},
                "done": {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
def mcp(payload: dict = None):
    """MCP JSON-RPC endpoint required by openenv-core validator."""
    return {
        "jsonrpc": "2.0",
        "id": (payload or {}).get("id", 1),
        "result": {
            "name": "email-triage-env",
            "version": "1.0.0",
            "capabilities": ["reset", "step", "state", "schema", "metadata"],
        },
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None):
    valid_tasks = ["classify_urgency", "triage_and_route", "inbox_zero"]
    task_id = (req.task_id if req else None) or "classify_urgency"
    session_id = (req.session_id if req else None) or str(uuid.uuid4())

    if task_id not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task_id. Choose from: {valid_tasks}")

    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    _sessions[session_id] = env

    return ResetResponse(
        session_id=session_id,
        observation=obs.model_dump(),
        info={"task_id": task_id, "max_steps": env.TASK_MAX_STEPS[task_id]},
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found. Call /reset first.")

    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs, reward, done, info = env.step(action)

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.value,
        reward_breakdown=reward.breakdown,
        reward_feedback=reward.feedback,
        done=done,
        info=info,
    )


@app.get("/state")
def state(session_id: str = Query(...)):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state().model_dump()


@app.get("/score")
def score(session_id: str = Query(...)):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {
        "session_id": session_id,
        "final_score": env.final_score(),
        "total_reward": env.total_reward,
        "sla_breaches": env.sla_breaches,
        "processed_count": len(env._processed),
        "pending_count": len(env._pending),
        "done": env.done,
    }


@app.get("/validate")
def validate():
    task_checks = {}
    for task_id in ["classify_urgency", "triage_and_route", "inbox_zero"]:
        try:
            e = EmailTriageEnv(task_id=task_id)
            obs = e.reset()
            assert isinstance(obs, Observation)
            assert len(obs.inbox) > 0
            email = obs.inbox[0]
            action = Action(action_type="classify", email_id=email.id, urgency="high", category="support")
            obs2, reward, done, info = e.step(action)
            assert isinstance(reward, Reward)
            assert 0.0 <= reward.value <= 1.0
            task_checks[task_id] = "pass"
        except Exception as ex:
            task_checks[task_id] = f"fail: {ex}"

    all_pass = all(v == "pass" for v in task_checks.values())
    return {
        "valid": all_pass,
        "spec_version": "1.0.0",
        "tasks": list(task_checks.keys()),
        "checks": {"tasks": task_checks},
    }


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()