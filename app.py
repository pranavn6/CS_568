"""
Interactive Multi-Agent Counterfactual Chat System
Flask backend that orchestrates 3 specialist agents + facilitator
for public health decision-making discussions.
"""

import os
import json
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
        "system_prompt": """You are Dr. Amara, a senior epidemiologist. Expertise: infection spread dynamics, R0, case/death trends, intervention effectiveness.

CRITICAL RULE: Every response MUST be exactly 1-2 sentences. Never exceed 2 sentences. Be direct and data-driven. Reference specific numbers from the scenario. Defer hospital/economic topics to colleagues.""",
    },
    "healthcare": {
        "name": "Dr. Chen (Healthcare Analyst)",
        "avatar": "H",
        "color": "#3498db",
        "system_prompt": """You are Dr. Chen, a healthcare capacity analyst. Expertise: hospital load, ICU capacity, staff burnout, surge planning.

CRITICAL RULE: Every response MUST be exactly 1-2 sentences. Never exceed 2 sentences. Be direct and reference specific hospital load numbers from the scenario. Defer economic/epidemiological modeling to colleagues.""",
    },
    "economist": {
        "name": "Prof. Rivera (Economist)",
        "avatar": "$",
        "color": "#27ae60",
        "system_prompt": """You are Prof. Rivera, a health economist. Expertise: cost-benefit analysis of restrictions, economic disruption, societal costs.

CRITICAL RULE: Every response MUST be exactly 1-2 sentences. Never exceed 2 sentences. Be direct and reference the economic cost metric from the scenario. Defer clinical/hospital topics to colleagues.""",
    },
}

FACILITATOR_SYSTEM = """You are the Discussion Facilitator. Synthesize what agents said, identify tensions, and ask the policymaker one guiding question.

CRITICAL RULE: Every response MUST be exactly 2-3 sentences. Never exceed 3 sentences."""

RANKING_SYSTEM = """You are a turn-taking coordinator for a multi-agent discussion about public health policy.
Given the conversation history and the last message, determine which agent should speak next.
Consider:
- Who has the most relevant expertise for the current topic?
- Who hasn't spoken recently?
- Who might disagree or add a contrasting perspective?
Return ONLY a JSON object: {"next_speaker": "epidemiologist|healthcare|economist", "reason": "brief reason"}"""


def build_scenario_text(interventions, metrics):
    scenario = {"interventions": interventions, "metrics": metrics}
    return json.dumps(scenario, indent=2)


def get_initial_analyses(interventions, metrics):
    """Run all 3 agents for initial scenario analysis."""
    scenario_text = build_scenario_text(interventions, metrics)
    results = {}

    for agent_id, agent in AGENTS.items():
        messages = [
            {"role": "system", "content": agent["system_prompt"]},
            {
                "role": "user",
                "content": f"""Scenario:\n{scenario_text}\n\nGive your initial take in 1-2 sentences only. Reference specific numbers from the data.""",
            },
        ]

        response = client.chat.completions.create(
            model=DEPLOYMENT, messages=messages, temperature=0.3, max_completion_tokens=150
        )
        results[agent_id] = response.choices[0].message.content

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

Continue the discussion from your perspective. Build on what others have said, add new insights, or respectfully challenge points you disagree with. Reference evidence where possible. Keep it very concise (1-2 sentences).""",
        },
    ]

    response = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, temperature=0.3, max_completion_tokens=150
    )
    return response.choices[0].message.content


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
        model=DEPLOYMENT, messages=messages, temperature=0.4, max_completion_tokens=250
    )
    return response.choices[0].message.content


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


# Serve frontend
@app.route("/")
def serve():
    return send_from_directory("frontend", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
