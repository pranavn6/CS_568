import os
from openai import AzureOpenAI
from utils import load_json, load_prompt, save_json, format_scenario
from config import PROMPT_DIR, OUTPUT_DIR

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]


def run_agent(agent_name, prompt_file, scenario):
    prompt = load_prompt(os.path.join(PROMPT_DIR, prompt_file))
    scenario_text = format_scenario(scenario)

    full_prompt = prompt + "\n\nScenario:\n" + scenario_text

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a precise analytical assistant."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def run_consensus(results):
    prompt = load_prompt(os.path.join(PROMPT_DIR, "consensus.txt"))

    full_prompt = (
        prompt
        + "\n\nEpidemiologist:\n" + results["epidemiologist"]
        + "\n\nHealthcare:\n" + results["healthcare"]
        + "\n\nEconomist:\n" + results["economist"]
    )

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a synthesis and reasoning assistant."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def run_all_agents(scenario):
    results = {
        "epidemiologist": run_agent("epidemiologist", "epidemiologist.txt", scenario),
        "healthcare": run_agent("healthcare", "healthcare.txt", scenario),
        "economist": run_agent("economist", "economist.txt", scenario),
    }

    results["consensus"] = run_consensus(results)
    return results


if __name__ == "__main__":
    scenario = load_json("data/sample_scenario_1.json")
    output = run_all_agents(scenario)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(os.path.join(OUTPUT_DIR, "sample_scenario_1_output.json"), output)
    print(output)