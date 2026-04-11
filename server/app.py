"""
FastAPI server — Email Triage OpenEnv v2
Full UI at / + complete OpenEnv API
"""
import os, sys, uuid
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import Action, EmailTriageEnvV2, Observation, Reward, State

app = FastAPI(title="Email Triage OpenEnv v2", version="2.0.0")
_sessions: Dict[str, EmailTriageEnvV2] = {}

# ── UI ────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Email Triage AI</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080810;--s1:#10101e;--s2:#16162a;--bd:#252540;
  --acc:#7c6af7;--ac2:#f76a8a;--ac3:#6af7c8;
  --txt:#e4e2ff;--mut:#5a5878;
  --cr:#ff4757;--hi:#ff9f43;--me:#ffd32a;--lo:#2ed573;--sp:#a29bfe;
}
body{font-family:'DM Mono',monospace;background:var(--bg);color:var(--txt);min-height:100vh}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(124,106,247,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(124,106,247,.04) 1px,transparent 1px);background-size:36px 36px;pointer-events:none}
.orb{position:fixed;border-radius:50%;pointer-events:none}
.orb1{width:500px;height:500px;background:radial-gradient(circle,rgba(124,106,247,.09),transparent 70%);top:-180px;right:-150px}
.orb2{width:350px;height:350px;background:radial-gradient(circle,rgba(247,106,138,.07),transparent 70%);bottom:-100px;left:-80px}
.wrap{position:relative;z-index:1;max-width:860px;margin:0 auto;padding:44px 20px 80px}

/* Header */
header{display:flex;align-items:center;gap:14px;margin-bottom:40px;animation:fD .6s ease}
.logo{width:42px;height:42px;background:linear-gradient(135deg,var(--acc),var(--ac2));border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:19px;flex-shrink:0}
header h1{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;letter-spacing:-.5px}
header p{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.hbadge{margin-left:auto;padding:4px 12px;border:1px solid var(--bd);border-radius:20px;font-size:9px;color:var(--mut);letter-spacing:.1em;text-transform:uppercase}

/* Pills */
.pills{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:20px;animation:fU .5s ease .1s both}
.pill{padding:5px 12px;background:var(--s1);border:1px solid var(--bd);border-radius:20px;font-size:10px;color:var(--mut);display:flex;align-items:center;gap:5px}
.dot{width:5px;height:5px;border-radius:50%}

/* Panel */
.panel{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:24px;margin-bottom:18px;animation:fU .5s ease}
.plabel{font-size:9px;color:var(--mut);letter-spacing:.15em;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.plabel::after{content:'';flex:1;height:1px;background:var(--bd)}

/* Examples */
.exrow{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px}
.exlabel{font-size:9px;color:var(--mut);text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px}
.ec{padding:5px 11px;background:var(--s2);border:1px solid var(--bd);border-radius:16px;font-size:11px;color:var(--mut);cursor:pointer;transition:all .2s}
.ec:hover{border-color:var(--acc);color:var(--txt)}

/* Inputs */
.irow{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
input,textarea,select{width:100%;background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:11px 14px;color:var(--txt);font-family:'DM Mono',monospace;font-size:13px;outline:none;transition:border-color .2s}
input:focus,textarea:focus,select:focus{border-color:var(--acc)}
textarea{min-height:130px;resize:vertical;line-height:1.6}
select option{background:var(--s2)}
.ifoot{display:flex;gap:8px;align-items:center;margin-top:10px}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:7px;padding:12px 24px;border-radius:9px;font-family:'Syne',sans-serif;font-weight:700;font-size:13px;cursor:pointer;border:none;transition:all .2s}
.btn-p{background:linear-gradient(135deg,var(--acc),#5040d0);color:#fff;width:100%;justify-content:center;margin-top:12px}
.btn-p:hover{opacity:.9;transform:translateY(-1px)}
.btn-p:disabled{opacity:.45;cursor:not-allowed;transform:none}
.btn-s{background:var(--s2);border:1px solid var(--bd);color:var(--mut);font-size:11px;padding:8px 14px}
.btn-s:hover{border-color:var(--acc);color:var(--txt)}

/* Result */
#rp{display:none;animation:fU .4s ease}
.rhead{display:flex;align-items:center;gap:16px;margin-bottom:22px}
.sring{position:relative;width:76px;height:76px;flex-shrink:0}
.sring svg{transform:rotate(-90deg)}
.stext{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.snum{font-family:'Syne',sans-serif;font-size:17px;font-weight:800}
.slbl{font-size:7px;color:var(--mut);text-transform:uppercase;letter-spacing:.1em}
.ubadge{padding:5px 13px;border-radius:5px;font-family:'Syne',sans-serif;font-weight:700;font-size:12px;letter-spacing:.05em;text-transform:uppercase}
.fbtext{font-size:12px;color:var(--mut);line-height:1.6;margin-top:6px}

/* Breakdown */
.bgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:18px}
.bc{background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:13px}
.bclbl{font-size:8px;color:var(--mut);text-transform:uppercase;letter-spacing:.12em;margin-bottom:7px}
.bcval{font-family:'Syne',sans-serif;font-size:20px;font-weight:700}
.bar{height:3px;background:var(--bd);border-radius:2px;margin-top:7px;overflow:hidden}
.bf{height:100%;border-radius:2px;width:0;transition:width .9s cubic-bezier(.4,0,.2,1)}

/* Action box */
.abox{background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:15px 18px;margin-top:12px;display:flex;align-items:flex-start;gap:12px}
.aicon{width:34px;height:34px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}
.abox h3{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.abox p{font-size:11px;color:var(--mut);line-height:1.6}

/* Alerts */
.palert{background:rgba(162,155,254,.1);border:1px solid rgba(162,155,254,.35);border-radius:9px;padding:13px 16px;margin-top:10px;display:none}
.palert-t{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:var(--sp);margin-bottom:5px}
.palert p{font-size:11px;color:var(--mut);line-height:1.6}
.salert{background:rgba(255,71,87,.07);border:1px solid rgba(255,71,87,.28);border-radius:9px;padding:11px 14px;font-size:11px;color:var(--cr);margin-top:10px;display:none}

/* Spinner */
.sp{width:14px;height:14px;border:2px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}

@keyframes fD{from{opacity:0;transform:translateY(-14px)}to{opacity:1;transform:none}}
@keyframes fU{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
.scanning{animation:pulse 1.1s ease infinite}
@media(max-width:580px){.irow{grid-template-columns:1fr}.bgrid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="orb orb1"></div><div class="orb orb2"></div>
<div class="wrap">

<header>
  <div class="logo">⚡</div>
  <div>
    <h1>Email Triage AI</h1>
    <p>OpenEnv v2 · Sequential Decision Environment</p>
  </div>
  <div class="hbadge">Live API</div>
</header>

<div class="pills">
  <div class="pill"><div class="dot" style="background:var(--lo)"></div>Running</div>
  <div class="pill">3 Tasks</div>
  <div class="pill">Thread Dependencies</div>
  <div class="pill">SLA Clock</div>
  <div class="pill">Phishing Detection</div>
  <div class="pill">Escalation Budget</div>
</div>

<div class="panel">
  <div class="plabel">Email Input</div>
  <div class="exlabel">Quick examples</div>
  <div class="exrow">
    <div class="ec" onclick="ex('crit')">🔴 Production outage</div>
    <div class="ec" onclick="ex('legal')">⚖️ Legal notice</div>
    <div class="ec" onclick="ex('phish')">🎣 Phishing email</div>
    <div class="ec" onclick="ex('churn')">💸 Client churn risk</div>
    <div class="ec" onclick="ex('spam')">🗑️ Spam</div>
    <div class="ec" onclick="ex('security')">🛡️ Security vuln</div>
  </div>

  <div class="irow">
    <input type="text" id="subj" placeholder="Subject line...">
    <input type="text" id="sndr" placeholder="sender@domain.com">
  </div>
  <textarea id="body" placeholder="Paste email body here...&#10;&#10;The AI will classify urgency, detect phishing, recommend an action, and score the response."></textarea>

  <div class="ifoot">
    <select id="task">
      <option value="classify_urgency">Task 1 — Classify urgency</option>
      <option value="triage_and_route" selected>Task 2 — Triage &amp; route</option>
      <option value="inbox_zero">Task 3 — Inbox zero</option>
    </select>
    <button class="btn btn-s" onclick="clr()">Clear</button>
  </div>

  <button class="btn btn-p" onclick="go()" id="abtn">
    <div class="sp" id="sp"></div>
    <span id="btxt">Analyze Email</span>
  </button>
</div>

<div class="panel" id="rp">
  <div class="plabel">Analysis Result</div>

  <div class="rhead">
    <div class="sring">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="30" fill="none" stroke="var(--bd)" stroke-width="6"/>
        <circle id="arc" cx="38" cy="38" r="30" fill="none" stroke="var(--acc)"
          stroke-width="6" stroke-linecap="round"
          stroke-dasharray="188.5" stroke-dashoffset="188.5"
          style="transition:stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)"/>
      </svg>
      <div class="stext"><span class="snum" id="snum">—</span><span class="slbl">Score</span></div>
    </div>
    <div style="flex:1">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span class="ubadge" id="ubadge">—</span>
        <span id="ctag" style="font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.1em"></span>
      </div>
      <div style="font-size:11px;color:var(--mut)">Recommended action:</div>
      <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;margin-top:2px" id="atitle">—</div>
      <div class="fbtext" id="fbtxt"></div>
    </div>
  </div>

  <div class="bgrid">
    <div class="bc"><div class="bclbl">Urgency accuracy</div><div class="bcval" id="bdu">—</div><div class="bar"><div class="bf" id="baru" style="background:var(--acc)"></div></div></div>
    <div class="bc"><div class="bclbl">Routing accuracy</div><div class="bcval" id="bdr">—</div><div class="bar"><div class="bf" id="barr" style="background:var(--ac3)"></div></div></div>
    <div class="bc"><div class="bclbl">Action accuracy</div><div class="bcval" id="bda">—</div><div class="bar"><div class="bf" id="bara" style="background:var(--ac2)"></div></div></div>
  </div>

  <div class="abox" id="abox">
    <div class="aicon" id="aico">📬</div>
    <div><h3 id="albl">Action</h3><p id="adesc"></p></div>
  </div>

  <div class="palert" id="palert">
    <div class="palert-t">⚠️ Phishing Detected</div>
    <p id="preason"></p>
  </div>
  <div class="salert" id="salert">⏱ SLA Alert — This email has a tight deadline. Immediate action required.</div>
</div>

</div>
<script>
const EX={
  crit:{s:"CRITICAL: Payment service down — transactions failing",f:"ops-alerts@company.com",b:"ALERT: Payment service (pay-svc-prod) is returning 500 errors. Error rate: 94% over last 5 minutes. Revenue impact: ~$12,000/minute. On-call engineer not acknowledging. Escalate immediately to incident commander."},
  legal:{s:"Legal notice — cease and desist re: patent infringement",f:"legal@morrisonfoster.com",b:"We represent TechCorp Industries and write to notify you of infringement of US Patent #9,876,543. You have 14 days to respond or we will file for injunctive relief. Please have your legal counsel contact us immediately."},
  phish:{s:"Action required: Verify your Microsoft 365 account",f:"it-support@company-helpdesk.net",b:"Dear user, We detected unusual sign-in activity. Verify your credentials within 24 hours: https://microsoft365-verify.company-helpdesk.net/login — Failure to verify will result in account suspension. — IT Security Team"},
  churn:{s:"Enterprise client threatening to cancel — $2.4M ARR at risk",f:"cto@bigclient.com",b:"I'm the CTO at BigClient Co. We've had repeated API timeout issues for 3 days. If not fixed by EOD Friday we'll terminate our $200k/month contract. I expect a call from your VP Engineering today."},
  spam:{s:"You've been selected! Claim your $500 Amazon gift card NOW",f:"noreply@prize-winner-2024.net",b:"Congratulations! You are our lucky winner. Click here to claim your $500 Amazon gift card before it expires in 24 hours: http://prize-winner-2024.net/claim"},
  security:{s:"Bug bounty submission — SQL injection in /api/users",f:"researcher@bugcrowd.com",b:"I discovered a critical SQL injection in your /api/users endpoint. CVSS 9.1. An attacker can dump the entire users table including password hashes. Per your bug bounty policy I expect acknowledgment within 24 hours or I will disclose publicly."}
};

const AM={
  escalate:{i:'🚨',c:'var(--cr)',d:'Immediately escalate to management or incident commander.'},
  route:{i:'📨',c:'var(--acc)',d:'Forward to the appropriate department for handling.'},
  reply:{i:'✍️',c:'var(--ac3)',d:'Draft and send a direct professional response.'},
  archive:{i:'📁',c:'var(--mut)',d:'Low priority — file away, no action required.'},
  mark_spam:{i:'🛡️',c:'var(--sp)',d:'Malicious or unsolicited — mark as spam and block sender.'},
  defer:{i:'⏸️',c:'var(--me)',d:'Not urgent — revisit when higher priority items are handled.'},
  flag:{i:'🚩',c:'var(--hi)',d:'Ambiguous — flag for human review.'}
};

const UC={critical:'var(--cr)',high:'var(--hi)',medium:'var(--me)',low:'var(--lo)',spam:'var(--sp)'};

function ex(k){const e=EX[k];document.getElementById('subj').value=e.s;document.getElementById('sndr').value=e.f;document.getElementById('body').value=e.b}
function clr(){['subj','sndr','body'].forEach(i=>document.getElementById(i).value='');document.getElementById('rp').style.display='none'}

async function go(){
  const subj=document.getElementById('subj').value.trim();
  const sndr=document.getElementById('sndr').value.trim();
  const body=document.getElementById('body').value.trim();
  const task=document.getElementById('task').value;
  if(!body){alert('Please enter an email body.');return}

  const btn=document.getElementById('abtn');
  btn.disabled=true;
  document.getElementById('sp').style.display='block';
  const bt=document.getElementById('btxt');
  bt.textContent='Analyzing...';bt.classList.add('scanning');

  try{
    const rr=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:task})});
    const rd=await rr.json();
    const sid=rd.session_id;
    const inbox=rd.observation.inbox||[];
    const ul=inbox.filter(e=>e.is_unlocked);
    const eid=ul.length>0?ul[0].id:(inbox[0]?.id||'e001');

    const txt=(subj+' '+sndr+' '+body).toLowerCase();
    let urg='medium',act='route',dept='support',cat='support';

    if(/critical|urgent|down|outage|breach|lawsuit|p0|emergency|crash/.test(txt))urg='critical';
    else if(/important|asap|today|deadline|expir|overdue|fail|churn|cancel/.test(txt))urg='high';
    else if(/please|request|follow.?up|reminder|question/.test(txt))urg='medium';
    else urg='low';

    const dom=(sndr.split('@')[1]||'').toLowerCase();
    const phish=/prize|winner|claim|verify.*account|click.*link|suspend/.test(txt)||/helpdesk\.net|selections\.net|secure-login|account-verify|verify\.[^.]+\.net/.test(dom);

    if(phish){act='mark_spam';cat='spam';urg='low';}
    else if(/legal|patent|lawsuit|cease|gdpr|compliance/.test(txt)){act='escalate';cat='legal';dept='legal';}
    else if(/payment|invoice|billing|overdue|refund|contract.*value|arr/.test(txt)){act='escalate';cat='finance';dept='finance';}
    else if(/down|outage|503|crash|incident|breach|vulnerability|exploit|sql injection/.test(txt)){act='escalate';cat='engineering';dept='engineering';}
    else if(/churn|cancel|terminate|competitor|dissatisfied/.test(txt)){act='escalate';cat='sales';dept='sales';}
    else if(/support|ticket|cannot|broken|issue|bug/.test(txt)){act='route';cat='support';dept='support';}
    else if(/newsletter|unsubscribe|monthly update/.test(txt)){act='archive';cat='spam';}
    else if(/nda|proposal|partnership|demo/.test(txt)){act='route';cat='sales';dept='sales';}

    const sr=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,action:{action_type:act,email_id:eid,urgency:urg,category:cat,department:dept,reply_text:act==='reply'?'Thank you for reaching out. Our team is investigating and will update you within 2 business hours.':null,reason:'Analyzed from content'}})});
    const sd=await sr.json();
    show({urg,act,cat,reward:sd.reward,bd:sd.reward_breakdown||{},fb:sd.reward_feedback||'',phish,txt});
  }catch(e){alert('Error: '+e.message);}
  finally{
    btn.disabled=false;
    document.getElementById('sp').style.display='none';
    bt.textContent='Analyze Email';bt.classList.remove('scanning');
  }
}

function show({urg,act,cat,reward,bd,fb,phish,txt}){
  const rp=document.getElementById('rp');
  rp.style.display='block';
  rp.scrollIntoView({behavior:'smooth',block:'nearest'});

  const pct=Math.round(reward*100);
  const C=188.5;
  const off=C-(pct/100)*C;
  const arc=document.getElementById('arc');
  arc.style.strokeDashoffset=C;
  const col=pct>=70?'var(--lo)':pct>=40?'var(--me)':'var(--cr)';
  arc.style.stroke=col;
  setTimeout(()=>arc.style.strokeDashoffset=off,60);
  document.getElementById('snum').textContent=pct+'%';

  const ub=document.getElementById('ubadge');
  ub.textContent=urg.toUpperCase();
  const uc=UC[urg]||'var(--mut)';
  ub.style.cssText=`background:${uc}20;color:${uc};border:1px solid ${uc}50`;
  document.getElementById('ctag').textContent=cat;

  const meta=AM[act]||AM.route;
  document.getElementById('atitle').textContent=meta.d.split('—')[0].trim();
  document.getElementById('atitle').style.color=meta.c;
  document.getElementById('fbtxt').textContent=fb;

  const keys={urgency:['bdu','baru','var(--acc)'],routing:['bdr','barr','var(--ac3)'],action:['bda','bara','var(--ac2)']};
  for(const[k,[vid,bid,col]]of Object.entries(keys)){
    const v=bd[k]!==undefined?bd[k]:(bd[k+'_score']||0);
    const p=Math.round(v*100);
    document.getElementById(vid).textContent=p+'%';
    setTimeout(()=>{document.getElementById(bid).style.width=p+'%'},200);
  }

  const ai=document.getElementById('aico');
  ai.textContent=meta.i;ai.style.background=meta.c+'20';
  document.getElementById('albl').textContent=act.replace('_',' ').toUpperCase();
  document.getElementById('albl').style.color=meta.c;
  document.getElementById('adesc').textContent=meta.d;

  const pa=document.getElementById('palert');
  pa.style.display=phish?'block':'none';
  if(phish)document.getElementById('preason').textContent='Suspicious signals: unusual sender domain, urgency manipulation, or credential/payment request. Do not click links. Report to IT security.';

  document.getElementById('salert').style.display=/critical|outage|in.*hour|today|deadline/.test(txt)?'block':'none';
}

document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')go()});
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=HTML)

# ── Schemas ───────────────────────────────────────────────────────────────────
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

# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status":"healthy","env":"email-triage-env-v2","version":"2.0.0"}

@app.get("/metadata")
def metadata():
    return {"name":"email-triage-env-v2","description":"Sequential email triage OpenEnv with thread dependencies, SLA clocks, phishing detection, and escalation budgets.","version":"2.0.0","tasks":["classify_urgency","triage_and_route","inbox_zero"]}

@app.get("/schema")
def schema():
    return {"action":{"type":"object","properties":{"action_type":{"type":"string","enum":["escalate","route","reply","archive","mark_spam","defer","flag"]},"email_id":{"type":"string"},"urgency":{"type":"string"},"category":{"type":"string"},"department":{"type":"string"},"reply_text":{"type":"string"},"reason":{"type":"string"}},"required":["action_type","email_id"]},"observation":{"type":"object","properties":{"task_id":{"type":"string"},"step_number":{"type":"integer"},"inbox":{"type":"array"},"sla_breaches":{"type":"integer"},"escalation_budget":{"type":"integer"},"done":{"type":"boolean"}}},"state":{"type":"object","properties":{"task_id":{"type":"string"},"action_history":{"type":"array"},"sla_breaches":{"type":"integer"},"escalation_budget":{"type":"integer"},"total_reward":{"type":"number"},"done":{"type":"boolean"}}}}

@app.post("/mcp")
def mcp(payload: dict = None):
    return {"jsonrpc":"2.0","id":(payload or {}).get("id",1),"result":{"name":"email-triage-env-v2","version":"2.0.0","capabilities":["reset","step","state","schema","metadata"]}}

@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None):
    valid=["classify_urgency","triage_and_route","inbox_zero"]
    tid=(req.task_id if req else None) or "classify_urgency"
    sid=(req.session_id if req else None) or str(uuid.uuid4())
    if tid not in valid:
        raise HTTPException(400,f"Invalid task_id")
    env=EmailTriageEnvV2(task_id=tid)
    obs=env.reset()
    _sessions[sid]=env
    return ResetResponse(session_id=sid,observation=obs.model_dump(),info={"task_id":tid,"max_steps":env.cfg["max_steps"],"escalation_budget":env.escalation_budget})

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env=_sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404,"Session not found")
    try:
        action=Action(**req.action)
    except Exception as e:
        raise HTTPException(422,f"Invalid action: {e}")
    obs,reward,done,info=env.step(action)
    return StepResponse(observation=obs.model_dump(),reward=reward.value,reward_breakdown=reward.breakdown,reward_feedback=reward.feedback,done=done,info=info)

@app.get("/state")
def state(session_id: str = Query(...)):
    env=_sessions.get(session_id)
    if env is None:
        raise HTTPException(404,"Session not found")
    return env.state().model_dump()

@app.get("/score")
def score(session_id: str = Query(...)):
    env=_sessions.get(session_id)
    if env is None:
        raise HTTPException(404,"Session not found")
    return {"session_id":session_id,"final_score":env.final_score(),"total_reward":env.total_reward,"sla_breaches":env.sla_breaches,"escalation_budget_remaining":env.escalation_budget,"processed_count":len(env._processed),"pending_count":len(env._pending),"done":env.done}

@app.get("/validate")
def validate():
    checks={}
    for tid in ["classify_urgency","triage_and_route","inbox_zero"]:
        try:
            e=EmailTriageEnvV2(task_id=tid);obs=e.reset()
            em=next(x for x in obs.inbox if x.is_unlocked)
            _,r,_,_=e.step(Action(action_type="escalate",email_id=em.id,urgency="critical",category="engineering",department="engineering"))
            assert 0.0<r.value<1.0
            checks[tid]="pass"
        except Exception as ex:
            checks[tid]=f"fail:{ex}"
    return {"valid":all(v=="pass" for v in checks.values()),"spec_version":"2.0.0","tasks":list(checks.keys()),"checks":{"tasks":checks}}

def main():
    import uvicorn
    port=int(os.getenv("PORT",7860))
    uvicorn.run("server.app:app",host="0.0.0.0",port=port,reload=False)

if __name__=="__main__":
    main()
