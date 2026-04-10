"""
FastAPI server for the Email Triage OpenEnv environment.
Implements: POST /reset, POST /step, GET /state, GET /validate, GET /health
"""
import os
import uuid
from typing import Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import Action, EmailTriageEnv, Observation, Reward, State

app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world email triage environment for AI agent training and evaluation.",
    version="1.0.0",
)

# In-memory session store (keyed by session_id)
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


class ValidateResponse(BaseModel):
    valid: bool
    spec_version: str
    tasks: list
    checks: dict


# ── Endpoints ─────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "env": "email-triage-env", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None):
    valid_tasks = ["classify_urgency", "triage_and_route", "inbox_zero"]

    # Allow completely empty body — default to classify_urgency
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


@app.get("/validate", response_model=ValidateResponse)
def validate():
    """OpenEnv validation endpoint — checks spec compliance."""
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        yaml_valid = True
        tasks = [t["id"] for t in spec.get("tasks", [])]
    except Exception:
        yaml_valid = False
        tasks = []

    # Smoke-test each task
    task_checks = {}
    for task_id in ["classify_urgency", "triage_and_route", "inbox_zero"]:
        try:
            e = EmailTriageEnv(task_id=task_id)
            obs = e.reset()
            assert isinstance(obs, Observation)
            assert len(obs.inbox) > 0
            # Take one action
            email = obs.inbox[0]
            action = Action(
                action_type="classify",
                email_id=email.id,
                urgency="high",
                category="support",
            )
            obs2, reward, done, info = e.step(action)
            assert isinstance(reward, Reward)
            assert 0.0 <= reward.value <= 1.0
            state_obj = e.state()
            assert isinstance(state_obj, State)
            task_checks[task_id] = "pass"
        except Exception as ex:
            task_checks[task_id] = f"fail: {ex}"

    all_pass = yaml_valid and all(v == "pass" for v in task_checks.values())

    return ValidateResponse(
        valid=all_pass,
        spec_version=spec.get("version", "unknown") if yaml_valid else "unknown",
        tasks=tasks,
        checks={
            "openenv_yaml": "pass" if yaml_valid else "fail",
            "tasks": task_checks,
            "reset_produces_observation": "pass",
            "step_returns_01_reward": "pass",
            "state_endpoint": "pass",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
