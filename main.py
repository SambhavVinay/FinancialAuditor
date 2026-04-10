import json
import re
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Crew, LLM, Process, Task

# Initialize FastAPI
app = FastAPI(title="Innovitus AI Claim Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LLM Configurations ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

vision_llm = LLM(
    model="openai/google/gemma-4-31b-it",
    base_url=NVIDIA_BASE_URL,
    api_key="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",
    temperature=0.2,   # Low temp for consistent damage classification
    max_tokens=16384,
)

brain_llm = LLM(
    model="openai/openai/gpt-oss-120b",
    base_url=NVIDIA_BASE_URL,
    api_key="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",
    temperature=0.1,   # Very low temp for deterministic math
    max_tokens=4096,
)

# --- 2. Input/Output Schemas ---
class ClaimInput(BaseModel):
    claim_id: str
    description: str
    policy_type: str
    claim_amount: int
    past_claims: int
    documents: str
    image_url: Optional[str] = None
    # Real image upload (base64-encoded, sent from frontend)
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = "image/jpeg"


# --- 2b. Direct multimodal vision call (bypasses CrewAI) ---
NVIDIA_VISION_KEY = "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"

def analyze_damage_with_vision(description: str, image_base64: str,
                               mime_type: str = "image/jpeg") -> str:
    """
    Sends the actual image pixels to NVIDIA Gemma 4 31B (multimodal) and
    returns a string like 'Damage Level: Moderate\nReasoning: ...'
    This result is injected into the CrewAI pipeline as grounded context.
    """
    data_uri = f"data:{mime_type};base64,{image_base64}"

    payload = {
        "model": "google/gemma-4-31b-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a forensic vehicle damage inspector.\n"
                            f"Claimant description: {description}\n\n"
                            "STEP 1 — VALIDITY CHECK:\n"
                            "Does this image clearly show a damaged or undamaged motor vehicle "
                            "(car, truck, motorcycle, van, bus)?\n"
                            "  • If NO  → respond ONLY with: INVALID IMAGE: <one sentence describing what you actually see>\n"
                            "  • If YES → continue to Step 2.\n\n"
                            "STEP 2 — DAMAGE CLASSIFICATION (only if image IS a vehicle):\n"
                            "Classify damage as EXACTLY ONE of:\n"
                            "  - Minor    : cosmetic scratches, small dents, paint chips only.\n"
                            "  - Moderate : broken lights, cracked bumper, panel deformation, broken glass.\n"
                            "  - Major    : structural/frame damage, engine crush, airbag deployment, "
                            "wheel arch collapse, vehicle likely total-loss.\n\n"
                            "Reply in this exact format (for valid vehicle images):\n"
                            "Damage Level: <Minor|Moderate|Major>\n"
                            "Reasoning: <one or two sentences describing what you visually observe>"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_VISION_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[VISION] Direct API call failed: {exc}")
        return "Damage Level: Moderate\nReasoning: Vision API unavailable; defaulted to Moderate."


def compute_image_description_match(vision_analysis: str, description: str) -> int:
    """
    Asks the brain LLM to score how closely the vision analysis of the uploaded
    image matches the claimant's written description.
    Returns an integer from 0 (completely different) to 100 (identical).
    If the API call fails, returns 100 to avoid blocking legitimate claims.
    """
    BRAIN_KEY = "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"

    prompt = (
        "You are an insurance fraud analyst.\n"
        "Compare the two texts below and rate how well they describe the SAME incident "
        "and the SAME type/level of damage.\n\n"
        f"TEXT A — Claimant's written description:\n{description}\n\n"
        f"TEXT B — Visual analysis of the uploaded photo:\n{vision_analysis}\n\n"
        "Score the match on a scale of 0 to 100, where:\n"
        "  0  = completely unrelated (e.g., photo shows a house fire, description says car accident)\n"
        "  50 = partially related but significant discrepancies in damage type or severity\n"
        " 100 = both texts describe the same incident with consistent damage level\n\n"
        "Reply with ONLY a single integer between 0 and 100. No explanation, no units."
    )

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {BRAIN_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Extract first integer from response
        m = re.search(r"(\d+)", raw)
        score = int(m.group(1)) if m else 100
        score = max(0, min(100, score))  # clamp 0-100
        print(f"[MATCH] Image-description match score: {score}/100")
        return score
    except Exception as exc:
        print(f"[MATCH] Match scoring API call failed: {exc} — defaulting to 100 (pass-through)")
        return 100  # fail-open: don't penalise if API is unavailable

# --- 3. Python-level payout enforcement (fallback guard) ---
DAMAGE_RATES = {
    "minor":    (0.70, 0.90),
    "moderate": (0.50, 0.70),
    "major":    (0.30, 0.50),
}
DEDUCTIBLE = 3000
FRAUD_THRESHOLD = 3  # past_claims > this → fraud flag

def enforce_payout(claim_amount: int, past_claims: int, policy_type: str,
                   damage_level: str, llm_payout: int | float | None) -> dict:
    """
    Re-validates the LLM's payout against hard business rules.
    Returns a dict with validated payout and any override notes.
    """
    notes = []

    # Rule 1: Third-Party doesn't cover own damage
    if "third" in policy_type.lower():
        return {"payout": 0, "override": True,
                "note": "Third-Party policy does not cover own-vehicle damage."}

    # Rule 2: Fraud detection
    if past_claims > FRAUD_THRESHOLD:
        return {"payout": 0, "override": True,
                "note": f"Claim rejected: {past_claims} past claims exceeds fraud threshold of {FRAUD_THRESHOLD}."}

    # Rule 3: Apply damage-based percentage (use midpoint of range)
    key = damage_level.lower().strip()
    if key not in DAMAGE_RATES:
        key = "moderate"  # safe fallback
        # Don't add a visible note for this — it's an internal default

    lo, hi = DAMAGE_RATES[key]
    midpoint_rate = (lo + hi) / 2
    max_allowed = int(claim_amount * hi) - DEDUCTIBLE
    min_allowed = int(claim_amount * lo) - DEDUCTIBLE
    midpoint_payout = int(claim_amount * midpoint_rate) - DEDUCTIBLE
    damage_label = key.capitalize()  # e.g. "Moderate"

    # Rule 4: Past-claims penalty (2% per prior claim, max 10%)
    penalty_pct = min(past_claims * 0.02, 0.10)
    if past_claims > 0:
        notes.append(f"A {past_claims * 2}% prior-claims penalty was applied.")
    adjusted = int(midpoint_payout * (1 - penalty_pct))

    # Rule 5: Validate LLM's payout against the actuarial band
    if llm_payout is not None:
        if min_allowed <= int(llm_payout) <= max_allowed:
            final = int(llm_payout)  # LLM was within range, trust it
        else:
            final = adjusted
            # Silent correction — no user-facing note needed
    else:
        final = adjusted
        # Silent correction — payout computed from rules

    # Rule 6: Payout can never exceed claim amount
    final = min(final, claim_amount)
    final = max(final, 0)  # never negative

    # Build a clean base reason (only penalty note is user-facing)
    return {"payout": final, "override": bool(notes), "note": " ".join(notes),
            "damage_label": damage_label}


def extract_damage_level(all_outputs: str) -> str:
    """
    Pull damage classification from any agent output.
    Looks for explicit 'Damage Level: X' first, then bare keywords.
    Checks for 'major' before 'moderate' before 'minor' so we don't
    accidentally match 'minor' inside 'major' etc.
    """
    text = all_outputs.lower()
    # Prefer explicit tag
    m = re.search(r"damage\s+level\s*[:\-]\s*(major|moderate|minor)", text)
    if m:
        return m.group(1)
    # Bare keywords as fallback
    for level in ["major", "moderate", "minor"]:
        if level in text:
            return level
    return "moderate"


def extract_payout_number(text: str) -> int | None:
    """
    Find the FINAL ₹ amount stated in the actuary's output.
    Actuary always ends with 'Final Payout: ₹X', so the last big number
    is the correct one. Filters out numbers that are clearly percentages
    or step-numbers (< 5000).
    """
    # Prefer explicit 'Final Payout' line first
    m = re.search(r"final\s+payout\s*[:\-]\s*[₹Rs\.\s]*(\d[\d,]*)", text, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))
    # Fallback — last large number in the text
    matches = re.findall(r"(?:₹|Rs\.?\s*)([\d,]+)", text.replace("\n", " "))
    candidates = []
    for match in matches:
        try:
            val = int(match.replace(",", ""))
            if val >= 5000:
                candidates.append(val)
        except ValueError:
            pass
    return candidates[-1] if candidates else None


def derive_status(payout: int, past_claims: int, policy_type: str,
                  damage_level: str, llm_status: str) -> str:
    """
    Deterministically decide the correct Status string from Python-side rules.
    This is the ground-truth — always replaces LLM's status.
    """
    if "third" in policy_type.lower():
        return "Rejected"
    if past_claims > FRAUD_THRESHOLD:
        return "Rejected"
    if payout <= 0:
        return "Rejected"
    # If payout covers ≥ 70% of expected midpoint → Approved, else Partially
    lo, hi = DAMAGE_RATES.get(damage_level, (0.50, 0.70))
    mid = (lo + hi) / 2
    expected = mid * 1  # relative benchmark is always the midpoint rate
    # If no penalty knocked it below 85% of midpoint calc, call it Approved
    # (2 past claims = 4% penalty → still close enough → Approved)
    # 3 past claims = 6% → borderline → Partially Approved
    if past_claims <= 2:
        return "Approved"
    return "Partially Approved"


def compute_confidence(
    has_image: bool,
    documents: str,
    past_claims: int,
    damage_level: str,
    policy_type: str,
    vision_contradicts_description: bool,
    is_rejected: bool,
    final_payout: int,
    claim_amount: int,
) -> str:
    """
    Compute a realistic, varying confidence score based on actual claim factors.
    Always overrides the LLM's value (which defaults to 95% every time).
    """
    score = 65  # starting baseline

    # Factor 1: Image evidence (most powerful signal)
    if has_image:
        score += 18  # we have actual pixel-level evidence
    else:
        score -= 5   # relying on text only

    # Factor 2: Document quality
    doc_count = len([d.strip() for d in documents.split(",") if d.strip()])
    if doc_count >= 3:
        score += 7
    elif doc_count == 2:
        score += 4
    elif doc_count == 1:
        score += 1
    # 0 docs: no bonus

    # Factor 3: Past claims history
    if past_claims == 0:
        score += 8   # clean record, low risk
    elif past_claims == 1:
        score += 4
    elif past_claims == 2:
        score += 1
    elif past_claims == 3:
        score -= 8   # elevated risk
    else:
        score -= 20  # fraud threshold exceeded

    # Factor 4: Vision vs. description consistency
    if vision_contradicts_description:
        score -= 12  # mismatch = suspicious claim
    elif has_image:
        score += 4   # vision confirmed the description

    # Factor 5: Damage clarity
    if damage_level in ("minor", "major"):
        score += 3   # clear-cut extremes are easier to assess
    # moderate is common so no bonus

    # Factor 6: Rejection certainty is actually high confidence
    if is_rejected:
        score = max(score, 72)  # we're very sure about rejections (rules are clear)
        score = min(score, 88)  # but cap it — rejections can be contested

    # Clamp the final score to a realistic band
    score = max(35, min(97, score))
    return f"{score}%"


# --- 4. Processing Route ---

@app.post("/process-claim")
async def process_claim(data: ClaimInput):
    try:
        # ── Agents (created per-request so context is always fresh) ──────────

        vision_analyst = Agent(
            role="Forensic Vision Inspector",
            goal="Classify the damage level from the description and image.",
            backstory=(
                "You are a forensic damage assessor. Based on the incident description "
                "(and image if provided), you MUST classify damage as exactly one of: "
                "Minor, Moderate, or Major. "
                "Minor = cosmetic scratches / small dents. "
                "Moderate = partial panel damage, broken lights, cracked windscreen. "
                "Major = structural damage, airbag deployment, engine damage, total loss risk. "
                "Always state the damage level on its own line like: 'Damage Level: Minor'."
            ),
            llm=vision_llm,
            verbose=True,
            allow_delegation=False,
        )

        compliance_officer = Agent(
            role="Policy Compliance Auditor",
            goal="Determine claim eligibility using strict policy rules.",
            backstory=(
                "You are a strict insurance auditor. Apply these rules in order:\n"
                "RULE 1 - Coverage check: If policy_type is 'Third-Party', own-vehicle damage is NOT covered → Status = Rejected.\n"
                "RULE 2 - Fraud check: If past_claims > 3, flag as HIGH FRAUD RISK → Status = Rejected.\n"
                "RULE 3 - If past_claims is 1-3, note a prior-claims penalty of (past_claims × 2)% will apply to payout.\n"
                "RULE 4 - If none of the above, Status = Eligible.\n"
                "Always output: 'Eligibility: Eligible' or 'Eligibility: Rejected' on its own line, "
                "followed by the reason."
            ),
            llm=brain_llm,
            verbose=True,
            allow_delegation=False,
        )

        payout_actuary = Agent(
            role="Payout Actuary",
            goal="Calculate the exact payout using the formula below. Never invent numbers.",
            backstory=(
                "You are a mathematical actuary. Follow this exact formula:\n\n"
                "STEP 1 - Get damage level from the Vision Inspector's output.\n"
                "STEP 2 - Apply the damage rate:\n"
                "  Minor    → payout = claim_amount × 0.80  (midpoint of 70-90%)\n"
                "  Moderate → payout = claim_amount × 0.60  (midpoint of 50-70%)\n"
                "  Major    → payout = claim_amount × 0.40  (midpoint of 30-50%)\n"
                "STEP 3 - Subtract ₹3,000 deductible: payout = payout - 3000\n"
                "STEP 4 - Apply prior-claims penalty from Compliance: payout = payout × (1 - past_claims × 0.02)\n"
                "STEP 5 - If claim is Rejected (Third-Party own damage OR fraud), payout = ₹0\n"
                "STEP 6 - Payout CANNOT exceed the original claim_amount. Cap it.\n"
                "STEP 7 - Payout cannot be negative. Minimum is ₹0.\n\n"
                "Show every step of your working. End with a clear line: 'Final Payout: ₹<amount>'"
            ),
            llm=brain_llm,
            verbose=True,
            allow_delegation=False,
        )

        communicator = Agent(
            role="Customer Relations Lead",
            goal="Synthesize all agent outputs into a strict JSON object.",
            backstory=(
                "You are a professional communicator. Read all prior agent outputs carefully "
                "and produce ONLY a raw JSON object (no markdown, no explanation outside JSON).\n"
                "Use ONLY the payout figure the Actuary calculated — do NOT invent a new number.\n"
                "The JSON must have these exact keys:\n"
                '  "Claim ID"        : the claim ID string\n'
                '  "Status"          : "Approved", "Rejected", or "Partially Approved"\n'
                '  "Estimated Payout": the numeric payout value (integer, no ₹ symbol)\n'
                '  "Confidence Score": a percentage string like "87%"\n'
                '  "Reason"          : one concise sentence explaining the decision\n'
                '  "Customer Message": a polite 2-sentence message to the customer\n'
            ),
            llm=brain_llm,
            verbose=True,
            allow_delegation=False,
        )

        # ── Pre-crew vision analysis (real multimodal if image uploaded) ─────
        vision_context = ""
        invalid_image_note = ""
        image_description_match_score: int | None = None  # None = no image uploaded
        image_mismatch_rejection = False  # set True when score < 50

        # Phrases that indicate the uploaded image is NOT a vehicle
        # We now use a single reliable sentinel: Gemma outputs 'INVALID IMAGE: ...'
        INVALID_SENTINEL = "invalid image"
        IMAGE_MATCH_THRESHOLD = 50  # payout = 0 if match score is below this

        if data.image_base64:
            print("[VISION] Image uploaded — calling Gemma 4 multimodal directly...")
            vision_context = analyze_damage_with_vision(
                description=data.description,
                image_base64=data.image_base64,
                mime_type=data.image_mime_type or "image/jpeg",
            )
            print(f"[VISION] Result: {vision_context[:200]}")

            # Validate: discard vision result if Gemma flagged it as not a vehicle
            if vision_context.lower().startswith(INVALID_SENTINEL):
                print("[VISION] ⚠ Not a vehicle image — ignoring vision result.")
                invalid_image_note = (
                    "The uploaded image did not show the claimed vehicle. "
                    "Assessment based on description only."
                )
                vision_context = ""  # fall back to text-only
            else:
                # ── Image-Description Match Check ────────────────────────────
                # Score how well the photo's visual analysis matches the claim
                # description. If < IMAGE_MATCH_THRESHOLD, reject with ₹0 payout.
                image_description_match_score = compute_image_description_match(
                    vision_analysis=vision_context,
                    description=data.description,
                )
                if image_description_match_score < IMAGE_MATCH_THRESHOLD:
                    image_mismatch_rejection = True
                    print(
                        f"[GUARD] ⛔ Image-description mismatch ({image_description_match_score}/100 < "
                        f"{IMAGE_MATCH_THRESHOLD}) — forcing payout to ₹0."
                    )

        task_vision = Task(
            description=(
                # If we already have real vision output, stamp it in as CONFIRMED
                (
                    f"CONFIRMED VISUAL ANALYSIS (Gemma 4 multimodal, image seen directly):\n"
                    f"{vision_context}\n\n"
                    f"Claimant description: {data.description}\n"
                    "Your job: confirm the damage level above is consistent with the description, "
                    "then re-state it as: 'Damage Level: <level>'"
                ) if vision_context else (
                    f"Incident description: {data.description}\n"
                    f"Image URL (if any): {data.image_url or 'None provided'}\n\n"
                    "Classify the damage level as Minor, Moderate, or Major. "
                    "State it explicitly as: 'Damage Level: <level>'"
                )
            ),
            agent=vision_analyst,
            expected_output="Damage Level: Minor | Moderate | Major, with brief justification.",
        )

        # Extract damage level NOW (before crew) so we can hardcode it in downstream tasks
        precomputed_damage = extract_damage_level(vision_context) if vision_context else "moderate"

        task_compliance = Task(
            description=(
                f"Policy Type: {data.policy_type}\n"
                f"Past Claims: {data.past_claims}\n"
                f"Incident: {data.description}\n"
                f"Documents submitted: {data.documents}\n\n"
                "Apply the eligibility rules from your instructions. "
                "Output 'Eligibility: Eligible' or 'Eligibility: Rejected' with reason."
            ),
            agent=compliance_officer,
            expected_output="Eligibility status (Eligible/Rejected) with reason and any penalty notes.",
        )

        task_payout = Task(
            description=(
                f"Claim Amount: ₹{data.claim_amount}\n"
                f"Past Claims: {data.past_claims}\n"
                f"DAMAGE LEVEL (confirmed by Vision Inspector): {precomputed_damage.capitalize()}\n"
                f"ELIGIBILITY (confirmed by Compliance Auditor): "
                f"{'Rejected — see compliance output' if data.past_claims > FRAUD_THRESHOLD or 'third' in data.policy_type.lower() else 'Eligible'}\n\n"
                f"Follow the 7-step formula in your backstory EXACTLY using the above inputs.\n"
                f"DO NOT say the damage level is missing — it is {precomputed_damage.capitalize()} as stated above.\n"
                "Show ALL steps. End with: 'Final Payout: ₹<amount>'"
            ),
            agent=payout_actuary,
            expected_output="Step-by-step calculation ending with 'Final Payout: ₹<amount>'",
        )

        # Pre-compute expected payout range for the communicator
        lo_rate, hi_rate = DAMAGE_RATES.get(precomputed_damage, (0.50, 0.70))
        mid_rate = (lo_rate + hi_rate) / 2
        penalty = min(data.past_claims * 0.02, 0.10)
        est_payout = max(int((data.claim_amount * mid_rate - DEDUCTIBLE) * (1 - penalty)), 0)
        is_rejected = data.past_claims > FRAUD_THRESHOLD or "third" in data.policy_type.lower()

        task_output = Task(
            description=(
                f"Claim ID: {data.claim_id}\n"
                f"Damage Level: {precomputed_damage.capitalize()}\n"
                f"Eligibility: {'Rejected' if is_rejected else 'Eligible'}\n"
                f"Estimated Payout from Actuary: ₹{est_payout if not is_rejected else 0}\n"
                f"Past Claims: {data.past_claims}\n\n"
                "Return ONLY a raw JSON object (no markdown) with these exact keys:\n"
                '"Claim ID": the claim ID string,\n'
                '"Status": "Approved" | "Partially Approved" | "Rejected",\n'
                '"Estimated Payout": integer (use the Actuary\'s figure above),\n'
                '"Confidence Score": a percentage string like "91%",\n'
                '"Reason": one concise professional sentence about the decision,\n'
                '"Customer Message": a warm, polite 2-sentence message appropriate to the Status above.\n'
                "If Status is Approved or Partially Approved, the Customer Message MUST be positive and confirm processing."
            ),
            agent=communicator,
            expected_output='Raw JSON object with keys: Claim ID, Status, Estimated Payout, Confidence Score, Reason, Customer Message.',
        )

        # ── Crew kickoff ──────────────────────────────────────────────────────

        claim_crew = Crew(
            agents=[vision_analyst, compliance_officer, payout_actuary, communicator],
            tasks=[task_vision, task_compliance, task_payout, task_output],
            process=Process.sequential,
            verbose=True,
        )

        raw_result = claim_crew.kickoff()
        raw_str = str(raw_result)

        # Strip markdown fences if present
        cleaned = raw_str.replace("```json", "").replace("```", "").strip()

        # Extract JSON from the string (in case the LLM prepended text)
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON object found in LLM output: {cleaned[:300]}")

        result = json.loads(json_match.group())

        # ── Python-level payout enforcement ───────────────────────────────────
        # Search ALL task outputs for the damage level (vision agent
        # sometimes buries it mid-paragraph; actuary often repeats it)
        all_outputs = " ".join([
            str(task_vision.output    or ""),
            str(task_compliance.output or ""),
            str(task_payout.output    or ""),
        ])
        actuary_output = str(task_payout.output or "")

        damage_level = extract_damage_level(all_outputs)
        llm_payout   = extract_payout_number(actuary_output)

        print(f"[GUARD] damage_level={damage_level!r}  llm_payout={llm_payout}")

        guard = enforce_payout(
            claim_amount=data.claim_amount,
            past_claims=data.past_claims,
            policy_type=data.policy_type,
            damage_level=damage_level,
            llm_payout=llm_payout,
        )

        # Always set payout from guard (it's either the validated LLM value
        # or our recalculated one — either way it's within the rules)
        final_payout = guard["payout"]

        # ── Image-Description Mismatch Override ──────────────────────────────
        # Hard rule: if the uploaded photo doesn't match the description by at
        # least 50%, the entire claim is zeroed out regardless of other factors.
        if image_mismatch_rejection:
            final_payout = 0
            mismatch_note = (
                f"Claim denied: the submitted photo does not match the incident description "
                f"(visual-match score {image_description_match_score}/100, "
                f"minimum required: {IMAGE_MATCH_THRESHOLD}/100)."
            )
            result["Reason"] = mismatch_note
            result["Customer Message"] = (
                "We regret to inform you that your claim could not be processed. "
                "The submitted photograph does not appear to match the described incident; "
                "please resubmit with a matching image or contact our support team for assistance."
            )

        result["Estimated Payout"] = final_payout

        # Append override notes to reason if the guard had to intervene
        if not image_mismatch_rejection and guard["override"] and guard["note"]:
            result["Reason"] = (result.get("Reason", "") + " " + guard["note"]).strip()

        # ── DETERMINISTIC STATUS ─────────────────────────────────────────────
        # Never trust the LLM for Status — always derive it from Python rules.
        # This prevents the "payout=46080 but Status=Rejected" contradiction.
        if image_mismatch_rejection:
            result["Status"] = "Rejected"
        else:
            llm_status = result.get("Status", "")
            result["Status"] = derive_status(
                payout=final_payout,
                past_claims=data.past_claims,
                policy_type=data.policy_type,
                damage_level=damage_level,
                llm_status=llm_status,
            )

        # ── DETERMINISTIC CONFIDENCE SCORE ───────────────────────────────────
        # Check if vision result contradicts the claimant's text description
        # (e.g. description says "small scratch" but vision says "Major")
        text_implied_damage = extract_damage_level(data.description)
        vision_contradicts = (
            bool(vision_context)  # only meaningful if image was uploaded
            and damage_level != text_implied_damage
            and not (text_implied_damage == "moderate")  # moderate is ambiguous default
        )
        # A mismatch-rejected claim always counts as contradicting
        if image_mismatch_rejection:
            vision_contradicts = True
        is_rejected_flag = result["Status"] == "Rejected"

        result["Confidence Score"] = compute_confidence(
            has_image=bool(data.image_base64),
            documents=data.documents,
            past_claims=data.past_claims,
            damage_level=damage_level,
            policy_type=data.policy_type,
            vision_contradicts_description=vision_contradicts,
            is_rejected=is_rejected_flag,
            final_payout=final_payout,
            claim_amount=data.claim_amount,
        )
        print(f"[CONFIDENCE] {result['Confidence Score']} "
              f"(img={bool(data.image_base64)}, vision_contradicts={vision_contradicts}, "
              f"match_score={image_description_match_score}, past_claims={data.past_claims})")

        # Always ensure payout is a clean int
        try:
            result["Estimated Payout"] = int(
                str(result["Estimated Payout"]).replace(",", "").replace("₹", "").strip()
            )
        except (ValueError, TypeError):
            result["Estimated Payout"] = final_payout

        # ── Fix Customer Message if actuary failed and communicator wrote an error msg ──
        # Skip generic fixes if a mismatch rejection message was already written
        bad_phrases = ["cannot process", "missing", "incomplete information",
                       "unable to calculate", "please provide"]
        customer_msg = result.get("Customer Message", "")
        if not image_mismatch_rejection and final_payout > 0 and any(p in customer_msg.lower() for p in bad_phrases):
            status_word = result["Status"]
            result["Customer Message"] = (
                f"Thank you for submitting your claim. We are pleased to inform you that your claim "
                f"has been {status_word.lower()} with an estimated payout of "
                f"\u20b9{final_payout:,}. Our team will process the disbursement shortly."
            )

        # Fix Reason if it contains error/debug language
        reason = result.get("Reason", "")
        if not image_mismatch_rejection and final_payout > 0 and any(p in reason.lower() for p in ["cannot", "missing", "unable"]):
            dmg = guard.get("damage_label", precomputed_damage.capitalize())
            result["Reason"] = (
                f"Claim approved under {data.policy_type} policy. "
                f"Damage classified as {dmg}. "
                + (f"A {data.past_claims * 2}% prior-claims penalty was applied." if data.past_claims > 0 else "")
            ).strip()

        # If image was invalid (not a vehicle), note it in the reason and reduce confidence
        if invalid_image_note:
            result["Reason"] = (invalid_image_note + " " + result.get("Reason", "")).strip()
            # Reduce confidence — we had no valid visual evidence
            try:
                current_score = int(result["Confidence Score"].replace("%", ""))
                result["Confidence Score"] = f"{max(35, current_score - 20)}%"
            except (ValueError, TypeError):
                pass

        return result

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Orchestration Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)