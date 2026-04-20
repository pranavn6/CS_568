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

AGENTS = {
    "epidemiologist": {
        "name": "Dr. Amara (Epidemiologist)",
        "avatar": "E",
        "color": "#e74c3c",
        "system_prompt": """You are Dr. Amara, an epidemiologist advising a policymaker on interventions (mask mandate, school closure, vaccination rate).

INITIAL ROUND: Assess whether the chosen interventions are appropriate given the forecast numbers. Reference the data.
FOLLOW-UP ROUNDS: Answer the user's question directly. Do NOT repeat forecast numbers or evidence already cited in the discussion. Focus on new reasoning and new evidence.

Cite one credible source with its URL. Do NOT cite a source already mentioned by another agent in this conversation. Example format: "A CDC study found masks reduced transmission by 53% (https://www.cdc.gov/mmwr/volumes/70/wr/mm7010e3.htm)."

1-2 sentences only. No lists, no headers.""",
    },
    "healthcare": {
        "name": "Dr. Chen (Healthcare Analyst)",
        "avatar": "H",
        "color": "#3498db",
        "system_prompt": """You are Dr. Chen, a healthcare capacity analyst advising a policymaker on interventions (mask mandate, school closure, vaccination rate).

INITIAL ROUND: Assess whether the chosen interventions will keep hospital admissions manageable given the forecast. Reference the data.
FOLLOW-UP ROUNDS: Answer the user's question directly. Do NOT repeat forecast numbers or evidence already cited in the discussion. Focus on new reasoning and new evidence.

Cite one credible source with its URL. Do NOT cite a source already mentioned by another agent in this conversation. Example format: "NEJM data shows ICU mortality rises sharply above 85% capacity (https://www.nejm.org/doi/full/10.1056/NEJMsa2029806)."

1-2 sentences only. No lists, no headers.""",
    },
    "economist": {
        "name": "Prof. Rivera (Economist)",
        "avatar": "$",
        "color": "#27ae60",
        "system_prompt": """You are Prof. Rivera, a health economist advising a policymaker on interventions (mask mandate, school closure, vaccination rate).

INITIAL ROUND: Assess whether the restrictions are proportionate to the threat shown by the forecast data. Reference the data.
FOLLOW-UP ROUNDS: Answer the user's question directly. Do NOT repeat forecast numbers or evidence already cited in the discussion. Focus on new reasoning and new evidence.

Cite one credible source with its URL. Do NOT cite a source already mentioned by another agent in this conversation. Example format: "IMF estimated lockdowns cost 3-4% of GDP annually (https://www.imf.org/en/Publications/WEO/Issues/2020/09/30/world-economic-outlook-october-2020)."

1-2 sentences only. No lists, no headers.""",
    },
}

FACILITATOR_SYSTEM = """You are a brief facilitator. Summarize where agents agree or disagree on the chosen interventions, then ask the policymaker one short question. Do NOT repeat forecast numbers. 2 sentences only."""

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

    parts = [f"Disease: {disease_name}", f"Jurisdiction: {jurisdiction}"]

    if metrics.get("latest_admissions") is not None:
        parts.append(f"Latest weekly hospital admissions: {metrics['latest_admissions']}")
    if metrics.get("wow_change") is not None:
        parts.append(f"Week-over-week change: {metrics['wow_change']}%")
    if metrics.get("four_week_trend") is not None:
        parts.append(f"4-week trend: {metrics['four_week_trend']}%")
    if metrics.get("next_week_forecast"):
        nw = metrics["next_week_forecast"]
        parts.append(f"Next week forecast: {nw['point']} (95% CI: {nw['lower_95']} - {nw['upper_95']})")

    parts.append(f"\nInterventions: {json.dumps(interventions)}")
    return "\n".join(parts)


def get_initial_analyses(interventions, metrics):
    """Run all 3 agents for initial scenario analysis."""
    scenario_text = build_scenario_text(interventions, metrics)
    results = {}

    for agent_id, agent in AGENTS.items():
        messages = [
            {"role": "system", "content": agent["system_prompt"]},
            {
                "role": "user",
                "content": f"""Scenario:\n{scenario_text}\n\nAre the chosen interventions appropriate given this forecast data? What should change? 1-2 sentences, reference the numbers.""",
            },
        ]

        response = client.chat.completions.create(
            model=DEPLOYMENT, messages=messages, temperature=0.3, max_completion_tokens=250
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            # Retry with simpler prompt
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a public health expert. Respond in 1-2 sentences."},
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

Answer the user's latest question or comment directly. Do NOT restate the forecast numbers. Provide new reasoning or evidence with a source URL. 1-2 sentences only.""",
        },
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, temperature=0.3, max_completion_tokens=250
    )
    return response.choices[0].message.content or "No response available."


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
        model=DEPLOYMENT, messages=messages, temperature=0.3, max_completion_tokens=250
    )
    return response.choices[0].message.content or "No summary available."


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

    # Get initial analyses from all 3 agents
    analyses = get_initial_analyses(interventions, metrics)

    # Build initial messages
    messages = []
    agent_order = ["epidemiologist", "healthcare", "economist"]
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

    # Facilitator synthesizes
    conv_history = format_conversation_history(messages)
    facilitator_msg = generate_facilitator_response(conv_history, scenario_text)
    messages.append(
        {
            "speaker": "facilitator",
            "speaker_name": "Facilitator",
            "content": facilitator_msg,
            "type": "facilitator",
            "avatar": "F",
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
                "avatar": "F",
                "color": "#8e44ad",
            }
        }
    )


@app.route("/api/summarize", methods=["POST"])
def summarize_paper():
    """Summarize a research paper given its URL."""
    data = request.json
    url = data.get("url", "")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a research summarizer. Given a URL to a published study or report, use your knowledge of that publication to provide a brief summary. If you recognize the paper, summarize it. If not, infer from the URL (domain, path, keywords) what it likely covers and summarize accordingly. Always produce output — never say you cannot access the URL."},
            {"role": "user", "content": f"Summarize this paper/report in 3-4 bullet points (use • for bullets). Cover: main finding, key data, and policy relevance.\n\nURL: {url}"},
        ],
        temperature=0.3,
        max_completion_tokens=300,
    )

    return jsonify({"summary": response.choices[0].message.content})


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
