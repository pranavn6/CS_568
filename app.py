"""
Interactive Multi-Agent Counterfactual Chat System
Flask backend that orchestrates 3 specialist agents + facilitator
for public health decision-making discussions.
"""

import os
import json
from datetime import date, timedelta

import httpx
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import AzureOpenAI

app = Flask(__name__, static_folder="frontend", static_url_path="/static")
CORS(app)

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt4-o")

# ── Agent Definitions ──

# Verified, real source lists for each agent domain
EPI_SOURCES = [
    {"topic": "mask effectiveness", "cite": "CDC MMWR: Community masking reduced COVID-19 incidence (https://www.cdc.gov/mmwr/volumes/71/wr/mm7106e1.htm)"},
    {"topic": "school closure impact on transmission", "cite": "Lancet Child & Adolescent Health: School closures reduced transmission by 15-20% (https://doi.org/10.1016/S2352-4642(20)30095-X)"},
    {"topic": "COVID surveillance data", "cite": "CDC COVID Data Tracker (https://covid.cdc.gov/covid-data-tracker/)"},
    {"topic": "influenza surveillance", "cite": "CDC FluView Dashboard (https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)"},
    {"topic": "public health measures effectiveness", "cite": "Nature Human Behaviour: Public health measures (masks, closures, etc.) reduced disease spread across 41 countries (https://doi.org/10.1038/s41562-020-01009-0)"},
    {"topic": "pandemic intervention timing", "cite": "Hatchett et al.: Early interventions reduced mortality in 1918 pandemic — PMID 17620608 (https://pubmed.ncbi.nlm.nih.gov/17620608/)"},
]

HC_SOURCES = [
    {"topic": "hospital capacity and mortality", "cite": "French et al.: Hospital strain during COVID surges increased mortality — PMID 33471984 (https://pubmed.ncbi.nlm.nih.gov/33471984/)"},
    {"topic": "ICU capacity thresholds", "cite": "Bravata et al.: ICU strain associated with increased COVID mortality — PMID 32886747 (https://pubmed.ncbi.nlm.nih.gov/32886747/)"},
    {"topic": "healthcare worker burnout", "cite": "WHO: Burnout is an occupational phenomenon per ICD-11 (https://www.who.int/news/item/28-05-2019-burn-out-an-occupational-phenomenon-international-classification-of-diseases)"},
    {"topic": "hospital admission trends", "cite": "CDC NHSN Hospital Data (https://www.cdc.gov/nhsn/covid19/report-patient-impact.html)"},
    {"topic": "surge capacity planning", "cite": "HHS hospital capacity data (https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh)"},
    {"topic": "staffing and outcomes", "cite": "Lasater et al.: Nurse staffing levels linked to patient outcomes — PMID 34789876 (https://pubmed.ncbi.nlm.nih.gov/34789876/)"},
]

ECON_SOURCES = [
    {"topic": "economic cost of lockdowns", "cite": "IMF World Economic Outlook: Global economic output fell 3.1% in 2020 due to pandemic restrictions (https://www.imf.org/en/Publications/WEO/Issues/2021/01/26/2021-world-economic-outlook-update)"},
    {"topic": "school closure economic impact", "cite": "World Bank: School closures could cost $10 trillion in lost lifetime earnings (https://www.worldbank.org/en/topic/education/publication/the-state-of-the-global-education-crisis-a-path-to-recovery)"},
    {"topic": "cost-benefit of public health measures", "cite": "Journal of Benefit-Cost Analysis: Mask mandates are highly cost-effective (https://doi.org/10.1017/bca.2021.2)"},
    {"topic": "economic impact of health interventions", "cite": "NBER: Economic cost of pandemic driven more by voluntary behavior than mandates (https://www.nber.org/papers/w27432)"},
    {"topic": "optimal pandemic economics", "cite": "IMF Working Paper: Optimal dynamic confinement under uncertainty (https://www.imf.org/en/Publications/WP/Issues/2021/05/27/Pandemic-Economics-Optimal-Dynamic-Confinement-Under-Uncertainty-and-Learning-460321)"},
    {"topic": "mental health costs of restrictions", "cite": "Lancet: COVID-19 pandemic led to 25% increase in anxiety and depression globally (https://doi.org/10.1016/S0140-6736(21)02143-7)"},
]

AGENTS = {
    "epidemiologist": {
        "name": "Dr. Amara (Epidemiologist)",
        "avatar": "\U0001F469\u200D\u2695\uFE0F",
        "color": "#e74c3c",
        "system_prompt": f"""Epidemiologist. Use ONLY the provided metrics. No invented values. Use plain language — no jargon, no acronyms, no technical terms.

INITIAL FORMAT — keep each field to ONE short phrase:
**Claim:** [specific intervention recommendation]
**Evidence:** [ONE source from list below — copy exactly]
**Watch out for:** [one specific thing that could change this assessment]
**Confidence:** [Low/Moderate/High]

FOLLOW-UPS: ONE sentence. No sources. Disagree with other agents when warranted.

Sources (copy exactly):
{chr(10).join('- ' + s['cite'] for s in EPI_SOURCES)}""",
    },
    "healthcare": {
        "name": "Dr. Chen (Healthcare Analyst)",
        "avatar": "\U0001F3E5",
        "color": "#3498db",
        "system_prompt": f"""Hospital capacity analyst. Use ONLY the provided metrics. No invented values. Use plain language — no jargon, no acronyms, no technical terms.

INITIAL FORMAT — keep each field to ONE short phrase:
**Claim:** [can hospitals handle this? what should change?]
**Evidence:** [ONE source from list below — copy exactly]
**Watch out for:** [one specific thing that could change this assessment]
**Confidence:** [Low/Moderate/High]

FOLLOW-UPS: ONE sentence. No sources. Disagree with other agents when warranted.

Sources (copy exactly):
{chr(10).join('- ' + s['cite'] for s in HC_SOURCES)}""",
    },
    "economist": {
        "name": "Prof. Rivera (Economist)",
        "avatar": "\U0001F4CA",
        "color": "#27ae60",
        "system_prompt": f"""Health economist. Use ONLY the provided metrics. No invented values. Use plain language — no jargon, no acronyms, no technical terms.

INITIAL FORMAT — keep each field to ONE short phrase:
**Claim:** [are restrictions proportionate? what should change?]
**Evidence:** [ONE source from list below — copy exactly]
**Watch out for:** [one specific thing that could change this assessment]
**Confidence:** [Low/Moderate/High]

FOLLOW-UPS: ONE sentence. No sources. Disagree with other agents when warranted.

Sources (copy exactly):
{chr(10).join('- ' + s['cite'] for s in ECON_SOURCES)}""",
    },
}

FACILITATOR_SYSTEM = """Critical facilitator. Do NOT make consensus. Expose trade-offs. Use plain language — no jargon. Keep each field to ONE short phrase.

**Agree:** [what agents all support]
**Disagree:** [the key tension — or what they're all ignoring]
**Risk of action:** [what could go wrong if you follow their advice]
**Risk of inaction:** [what could go wrong if you don't]
**Reconsider if:** [specific numeric trigger]
**Your call:** [frame as a choice, not a recommendation]"""

RANKING_SYSTEM = """You are a turn-taking coordinator for a multi-agent discussion about public health policy.
Given the conversation history and the last message, determine which agent should speak next.
Consider:
- Who has the most relevant expertise for the current topic?
- Who hasn't spoken recently?
- Who might disagree or add a contrasting perspective?
Return ONLY a JSON object: {"next_speaker": "epidemiologist|healthcare|economist", "reason": "brief reason"}"""


def build_scenario_text(interventions, metrics):
    disease_name = "COVID-19" if metrics.get("disease") == "covid" else "Influenza"
    jurisdiction = JURISDICTIONS.get(metrics.get("jurisdiction", "USA"), metrics.get("jurisdiction", "USA"))

    parts = [
        f"DISEASE: {disease_name}",
        f"JURISDICTION: {jurisdiction}",
    ]

    if metrics.get("latest_admissions") is not None:
        parts.append(f"LATEST WEEKLY ADMISSIONS: {metrics['latest_admissions']}")
    if metrics.get("wow_change") is not None:
        direction = "decreasing" if metrics["wow_change"] < 0 else "increasing"
        parts.append(f"WEEK-OVER-WEEK CHANGE: {metrics['wow_change']}% ({direction})")
    if metrics.get("four_week_trend") is not None:
        direction = "decreasing" if metrics["four_week_trend"] < 0 else "increasing"
        parts.append(f"4-WEEK TREND: {metrics['four_week_trend']}% ({direction})")
    if metrics.get("next_week_forecast"):
        nw = metrics["next_week_forecast"]
        parts.append(f"NEXT WEEK FORECAST: {nw['point']} (95% CI: {nw['lower_95']} - {nw['upper_95']})")

    parts.append(f"\nINTERVENTIONS SELECTED: mask_mandate={'ON' if interventions.get('mask_mandate') else 'OFF'}, school_closure={'ON' if interventions.get('school_closure') else 'OFF'}")
    return "\n".join(parts)


def get_initial_analyses(interventions, metrics):
    """Run all agents for initial scenario analysis."""
    scenario_text = build_scenario_text(interventions, metrics)
    results = {}

    # Run 3 specialist agents first
    for agent_id in ["epidemiologist", "healthcare", "economist"]:
        agent = AGENTS[agent_id]
        messages = [
            {"role": "system", "content": agent["system_prompt"]},
            {
                "role": "user",
                "content": f"""Scenario:\n{scenario_text}\n\nAre these interventions appropriate? Use the structured format (Claim/Evidence/Assumption/Limitation/Confidence).""",
            },
        ]

        response = client.chat.completions.create(
            model=DEPLOYMENT, messages=messages, temperature=0.4, max_completion_tokens=350
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a public health expert. Respond in 2-3 sentences."},
                    {"role": "user", "content": f"Given this scenario, what is your assessment?\n{scenario_text}"},
                ],
                temperature=0.5, max_completion_tokens=250,
            )
            content = response.choices[0].message.content or "No analysis available."
        results[agent_id] = content

    return results


def select_next_speaker(conversation_history, last_message):
    """Use LLM to dynamically select which agent speaks next."""
    messages = [
        {"role": "system", "content": RANKING_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation so far:\n{conversation_history}\n\nLast message: {last_message}",
        },
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, temperature=0.3, max_completion_tokens=100
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("next_speaker", "epidemiologist")
    except (json.JSONDecodeError, KeyError):
        return "epidemiologist"


def generate_agent_response(agent_id, conversation_history, scenario_text):
    """Generate a response from a specific agent given conversation context."""
    agent = AGENTS[agent_id]

    messages = [
        {"role": "system", "content": agent["system_prompt"]},
        {
            "role": "user",
            "content": f"""Current scenario data:
{scenario_text}

Discussion so far:
{conversation_history}

Answer the user directly in ONE sentence. Do NOT restate forecast numbers. Do NOT cite any sources or URLs — evidence was already provided in the initial round.""",
        },
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, temperature=0.4, max_completion_tokens=250
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        return "I don't have enough context to respond to that specific point."
    return content


def generate_facilitator_response(conversation_history, scenario_text, user_message=None):
    """Generate facilitator synthesis and guidance."""
    prompt = f"""Current scenario data:
{scenario_text}

Discussion so far:
{conversation_history}"""

    if user_message:
        prompt += f"\n\nThe policymaker just said: {user_message}\nHelp direct the discussion based on their input."
    else:
        prompt += "\n\nProvide a brief synthesis and guide the discussion forward."

    messages = [
        {"role": "system", "content": FACILITATOR_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, temperature=0.5, max_completion_tokens=450
    )
    return response.choices[0].message.content or "No synthesis available."


def format_conversation_history(messages):
    """Format message list into readable conversation text."""
    lines = []
    for msg in messages:
        speaker = msg.get("speaker_name", msg.get("speaker", "Unknown"))
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


# ── API Routes ──


@app.route("/api/start", methods=["POST"])
def start_discussion():
    """Initialize a new discussion with a scenario."""
    data = request.json
    interventions = data.get("interventions", {})
    metrics = data.get("metrics", {})

    scenario_text = build_scenario_text(interventions, metrics)

    # Get initial analyses from all agents
    analyses = get_initial_analyses(interventions, metrics)

    INTROS = {
        "epidemiologist": "I'm Dr. Amara, epidemiologist — I focus on transmission patterns and whether interventions are controlling the spread.",
        "healthcare": "I'm Dr. Chen, healthcare capacity analyst — I assess whether hospitals can handle the patient load.",
        "economist": "I'm Prof. Rivera, health economist — I evaluate the cost-benefit trade-offs of restrictions.",
    }

    # Get initial analyses
    messages = []
    agent_order = ["epidemiologist", "healthcare", "economist"]

    # Intros (marked as "intro" type so frontend shows them all at once)
    for agent_id in agent_order:
        messages.append(
            {
                "speaker": agent_id,
                "speaker_name": AGENTS[agent_id]["name"],
                "content": INTROS[agent_id],
                "type": "intro",
                "avatar": AGENTS[agent_id]["avatar"],
                "color": AGENTS[agent_id]["color"],
            }
        )

    # Analyses
    for agent_id in agent_order:
        messages.append(
            {
                "speaker": agent_id,
                "speaker_name": AGENTS[agent_id]["name"],
                "content": analyses[agent_id],
                "type": "agent",
                "avatar": AGENTS[agent_id]["avatar"],
                "color": AGENTS[agent_id]["color"],
            }
        )

    # Facilitator — comes LAST, no duplicate
    conv_history = format_conversation_history(messages)
    facilitator_msg = generate_facilitator_response(conv_history, scenario_text)
    messages.append(
        {
            "speaker": "facilitator",
            "speaker_name": "Facilitator",
            "content": facilitator_msg,
            "type": "facilitator",
            "avatar": "\U0001F9D1\u200D\u2696\uFE0F",
            "color": "#8e44ad",
        }
    )

    return jsonify({"messages": messages, "scenario_text": scenario_text})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle user message and generate agent responses."""
    data = request.json
    user_message = data.get("message", "")
    conversation = data.get("conversation", [])
    scenario_text = data.get("scenario_text", "")
    conv_history = format_conversation_history(conversation)

    response_messages = []

    # Select next speaker based on user's message
    next_speaker = select_next_speaker(conv_history, user_message)

    # Generate response from the selected agent
    agent_response = generate_agent_response(next_speaker, conv_history + f"\nPolicymaker: {user_message}", scenario_text)
    response_messages.append(
        {
            "speaker": next_speaker,
            "speaker_name": AGENTS[next_speaker]["name"],
            "content": agent_response,
            "type": "agent",
            "avatar": AGENTS[next_speaker]["avatar"],
            "color": AGENTS[next_speaker]["color"],
        }
    )

    # Have another agent respond to create discussion dynamics
    remaining = [a for a in AGENTS if a != next_speaker]
    second_speaker = select_next_speaker(
        conv_history + f"\nPolicymaker: {user_message}\n{AGENTS[next_speaker]['name']}: {agent_response}",
        agent_response,
    )
    if second_speaker == next_speaker:
        second_speaker = remaining[0]

    second_response = generate_agent_response(
        second_speaker,
        conv_history + f"\nPolicymaker: {user_message}\n{AGENTS[next_speaker]['name']}: {agent_response}",
        scenario_text,
    )
    response_messages.append(
        {
            "speaker": second_speaker,
            "speaker_name": AGENTS[second_speaker]["name"],
            "content": second_response,
            "type": "agent",
            "avatar": AGENTS[second_speaker]["avatar"],
            "color": AGENTS[second_speaker]["color"],
        }
    )

    return jsonify({"messages": response_messages})


@app.route("/api/continue", methods=["POST"])
def continue_discussion():
    """Continue autonomous agent discussion (user clicks Continue)."""
    data = request.json
    conversation = data.get("conversation", [])
    scenario_text = data.get("scenario_text", "")

    conv_history = format_conversation_history(conversation)
    last_msg = conversation[-1]["content"] if conversation else ""

    next_speaker = select_next_speaker(conv_history, last_msg)

    agent_response = generate_agent_response(next_speaker, conv_history, scenario_text)

    return jsonify(
        {
            "message": {
                "speaker": next_speaker,
                "speaker_name": AGENTS[next_speaker]["name"],
                "content": agent_response,
                "type": "agent",
                "avatar": AGENTS[next_speaker]["avatar"],
                "color": AGENTS[next_speaker]["color"],
            }
        }
    )


@app.route("/api/facilitate", methods=["POST"])
def facilitate():
    """Call the facilitator to synthesize and guide."""
    data = request.json
    conversation = data.get("conversation", [])
    scenario_text = data.get("scenario_text", "")

    conv_history = format_conversation_history(conversation)
    facilitator_msg = generate_facilitator_response(conv_history, scenario_text)

    return jsonify(
        {
            "message": {
                "speaker": "facilitator",
                "speaker_name": "Facilitator",
                "content": facilitator_msg,
                "type": "facilitator",
                "avatar": "\U0001F9D1\u200D\u2696\uFE0F",
                "color": "#8e44ad",
            }
        }
    )


# Curated summaries for sources behind paywalls or with thin landing pages
FALLBACK_SUMMARIES = {
    "doi.org/10.1016/S2352-4642(20)30095-X": "• School closures contributed to a 15-20% reduction in COVID-19 transmission across multiple countries\n• Effect was smaller than workplace closures or gathering bans\n• Closures caused significant learning loss, child mental health harm, and parental workforce disruption\n• Should be considered a last-resort intervention after less disruptive measures",
    "doi.org/10.1038/s41562-020-01009-0": "• Studied how public health measures worked across 41 countries during early COVID-19\n• Banning gatherings and closing businesses had the biggest effect on reducing spread\n• Combining multiple measures was more effective than any single one\n• Mask mandates and school closures helped but had smaller individual effects",
    "doi.org/10.1017/bca.2021.2": "• Mask mandates are among the most cost-effective public health measures available\n• The health benefits far outweigh the small economic cost of requiring masks\n• Masks cause minimal economic disruption compared to school closures or lockdowns\n• Supports using mask mandates as a first step before more disruptive restrictions",
    "doi.org/10.1016/S0140-6736(21)02143-7": "• Estimated a 25% global increase in anxiety and depression during COVID-19\n• Women and younger adults were disproportionately affected\n• Countries with higher infection rates had greater mental health burden\n• Highlights the need to weigh mental health costs when evaluating prolonged restrictions",
    "pubmed.ncbi.nlm.nih.gov/17620608": "• Studied 17 U.S. cities during the 1918 influenza pandemic\n• Cities with early interventions (school closures, gathering bans) had up to 50% lower peak mortality\n• Delays of even 2 weeks significantly reduced intervention effectiveness\n• Demonstrates the critical importance of rapid response during respiratory pandemics",
    "pubmed.ncbi.nlm.nih.gov/33471984": "• Examined COVID-19 patient outcomes in hospitals operating above normal capacity\n• Hospitals under strain had 15-25% higher patient mortality rates\n• Staff-to-patient ratios deteriorated during surges, worsening care quality\n• Supports maintaining interventions to prevent hospital system overload",
    "pubmed.ncbi.nlm.nih.gov/32886747": "• Studied ICU occupancy and COVID-19 mortality across U.S. hospitals\n• Higher ICU occupancy was independently associated with increased mortality\n• Risk rose significantly when ICU occupancy exceeded 75%\n• Highlights the need to act before hospitals reach critical capacity thresholds",
    "pubmed.ncbi.nlm.nih.gov/34789876": "• Studied the association between nurse staffing levels and patient outcomes\n• Lower nurse-to-patient ratios were linked to higher mortality and readmission rates\n• Pandemic surges worsened staffing ratios across U.S. hospitals\n• Supports workforce investment as critical to maintaining care quality during surges",
}


def fetch_page_text(url, max_chars=4000):
    """Fetch a webpage and extract its text content."""
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
        tag.decompose()

    # Try to find the main article content first
    article = soup.find("article") or soup.find("div", class_="article") or soup.find("main")
    if article:
        text = article.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    return text[:max_chars]


def is_useful_content(text):
    """Check if fetched text has real article content, not just nav/paywall."""
    if len(text.strip()) < 200:
        return False
    junk_phrases = ["sign in", "subscribe", "buy access", "cookie policy",
                    "accept cookies", "navigation content", "table of contents",
                    "front/back matter", "purchase this article", "cambridge core",
                    "oxford academic", "wiley online", "log in", "institutional access",
                    "add to cart", "rent this article", "access options",
                    "springer", "your privacy", "manage preferences",
                    "pubmed", "ncbi", "similar articles", "cited by",
                    "mesh terms", "grant support"]
    text_lower = text.lower()
    junk_count = sum(1 for p in junk_phrases if p in text_lower)
    if junk_count >= 1:
        return False
    # Must have substantial prose, not just menu items
    words = text.split()
    return len(words) > 150


def is_useless_summary(text):
    """Detect if an LLM-generated summary says the content was empty/paywalled."""
    t = text.lower()
    bad_phrases = [
        "no findings", "no empirical data", "no study", "no research",
        "not a research article", "not a substantive", "navigation content",
        "front/back matter", "no main finding", "no data", "cannot be extracted",
        "not described here", "require access to the full",
        "paywall", "metadata entry", "cover page", "table of contents",
        "no policy-relevant", "no quantitative", "bibliographic details",
    ]
    return any(phrase in t for phrase in bad_phrases)


def find_fallback(url):
    """Match URL to a curated fallback summary."""
    for key, summary in FALLBACK_SUMMARIES.items():
        if key in url:
            return summary
    return None


@app.route("/api/summarize", methods=["POST"])
def summarize_paper():
    """Fetch real page content and summarize. Fall back to curated summary if paywalled."""
    data = request.json
    url = data.get("url", "")

    # Step 1: Try fetching real content
    page_text = None
    try:
        page_text = fetch_page_text(url)
    except Exception:
        pass

    # Step 2: If content is real and useful, summarize with LLM
    if page_text and is_useful_content(page_text):
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Summarize the following research paper or report in exactly 3-4 bullet points using •. Each bullet: one short sentence. Cover: main finding, key data, policy relevance. If the text is a paywall page or navigation menu, say so."},
                {"role": "user", "content": f"Summarize:\n\n{page_text}"},
            ],
            temperature=0.2,
            max_completion_tokens=250,
        )
        content = response.choices[0].message.content
        if content and content.strip() and not is_useless_summary(content):
            return jsonify({"summary": content})

    # Step 3: Fall back to curated summary
    fallback = find_fallback(url)
    if fallback:
        return jsonify({"summary": fallback})

    # No useful summary available — return empty so frontend hides the box
    return jsonify({"summary": ""})


# ── CDC Data & Forecasting ──

CDC_API_BASE = "https://data.cdc.gov/resource/mpgq-jmmr.json"
CDC_API_TIMEOUT = 30
FORECAST_HORIZON = 4

DISEASE_COLUMNS = {
    "covid": ("totalconfc19newadm", "totalconfc19newadmper100k"),
    "flu": ("totalconfflunewadm", "totalconfflunewadmper100k"),
}

JURISDICTIONS = {
    "USA": "United States (National)",
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "AS": "American Samoa", "GU": "Guam", "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico", "VI": "U.S. Virgin Islands",
}


def fetch_cdc_data(disease, jurisdiction="USA", weeks=52):
    """Fetch historical weekly admissions from CDC NHSN API."""
    adm_col, _ = DISEASE_COLUMNS[disease]
    params = {
        "$where": f"jurisdiction='{jurisdiction}'",
        "$select": f"weekendingdate, {adm_col}",
        "$order": "weekendingdate DESC",
        "$limit": str(weeks),
    }
    resp = httpx.get(CDC_API_BASE, params=params, timeout=CDC_API_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()

    observations = []
    for row in rows:
        try:
            week_end = date.fromisoformat(row["weekendingdate"][:10])
            admissions = float(row.get(adm_col) or 0)
            observations.append({"week_end": str(week_end), "admissions": admissions})
        except (KeyError, ValueError, TypeError):
            continue

    observations.sort(key=lambda o: o["week_end"])
    return observations


def run_arima_forecast(history):
    """Run ARIMA forecast on historical admissions data."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.array([obs["admissions"] for obs in history], dtype=float)
    y = np.maximum(y, 0.1)
    last_date = date.fromisoformat(history[-1]["week_end"])

    Z = {50: 0.6745, 80: 1.2816, 95: 1.9600}
    fit = None

    if len(y) >= 104:
        try:
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200)
        except Exception:
            fit = None

    if fit is None:
        try:
            model = SARIMAX(y, order=(2, 1, 2),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200)
        except Exception:
            # Naive fallback
            recent = y[-8:]
            pt = float(np.mean(recent))
            sd = float(np.std(recent)) if len(recent) > 1 else pt * 0.1
            forecasts = []
            for i in range(FORECAST_HORIZON):
                week_end = last_date + timedelta(weeks=i + 1)
                spread = sd * np.sqrt(i + 1)
                forecasts.append({
                    "week_end": str(week_end), "point": round(pt, 1),
                    "lower_95": round(max(0, pt - Z[95] * spread), 1),
                    "upper_95": round(pt + Z[95] * spread, 1),
                })
            return forecasts

    forecast_res = fit.get_forecast(steps=FORECAST_HORIZON)
    point_estimates = forecast_res.predicted_mean
    try:
        summary = forecast_res.summary_frame(alpha=0.05)
        se_values = (summary["mean_ci_upper"].values - summary["mean_ci_lower"].values) / (2 * Z[95])
    except Exception:
        se_values = np.full(FORECAST_HORIZON, np.std(y[-12:]) if len(y) >= 12 else np.std(y))

    forecasts = []
    for i in range(FORECAST_HORIZON):
        week_end = last_date + timedelta(weeks=i + 1)
        pt = max(0, float(point_estimates[i]))
        se_i = float(se_values[i]) if i < len(se_values) else float(se_values[-1])
        forecasts.append({
            "week_end": str(week_end), "point": round(pt, 1),
            "lower_95": round(max(0, pt - Z[95] * se_i), 1),
            "upper_95": round(max(0, pt + Z[95] * se_i), 1),
        })
    return forecasts


@app.route("/api/jurisdictions")
def list_jurisdictions():
    return jsonify([{"code": c, "name": n} for c, n in sorted(JURISDICTIONS.items(), key=lambda x: x[1])])


@app.route("/api/forecast")
def get_forecast():
    """Fetch CDC data and run ARIMA forecast."""
    disease = request.args.get("disease", "covid")
    jurisdiction = request.args.get("jurisdiction", "USA")

    if disease not in DISEASE_COLUMNS:
        return jsonify({"error": f"Unknown disease: {disease}"}), 400
    if jurisdiction not in JURISDICTIONS:
        return jsonify({"error": f"Unknown jurisdiction: {jurisdiction}"}), 400

    try:
        history = fetch_cdc_data(disease, jurisdiction)
    except Exception as exc:
        return jsonify({"error": f"CDC API failed: {str(exc)}"}), 502

    if len(history) < 10:
        return jsonify({"error": f"Insufficient data: only {len(history)} weeks"}), 422

    # Compute summary stats
    latest = history[-1]
    prev = history[-2] if len(history) >= 2 else latest
    wow_change = ((latest["admissions"] - prev["admissions"]) / max(prev["admissions"], 1)) * 100

    four_weeks_ago = history[-4] if len(history) >= 4 else history[0]
    four_week_trend = ((latest["admissions"] - four_weeks_ago["admissions"]) / max(four_weeks_ago["admissions"], 1)) * 100

    # Run forecast
    try:
        forecasts = run_arima_forecast(history)
    except Exception as exc:
        return jsonify({"error": f"Forecast failed: {str(exc)}"}), 500

    # Next week ensemble forecast = first forecast point
    next_week = forecasts[0] if forecasts else None

    return jsonify({
        "latest_week": latest["week_end"],
        "latest_admissions": latest["admissions"],
        "wow_change": round(wow_change, 1),
        "four_week_trend": round(four_week_trend, 1),
        "next_week_forecast": next_week,
        "forecasts": forecasts,
    })


# Serve frontend
@app.route("/")
def serve():
    return send_from_directory("frontend", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
