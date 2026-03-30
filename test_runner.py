import os
from utils import load_json, save_json
from agent_runner import run_all_agents
from config import OUTPUT_DIR

scenario_files = [
    "data/sample_scenario_1.json",
    "data/sample_scenario_2.json",
    "data/sample_scenario_3.json"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in scenario_files:
    scenario = load_json(path)
    results = run_all_agents(scenario)

    file_name = os.path.basename(path).replace(".json", "_output.json")
    save_path = os.path.join(OUTPUT_DIR, file_name)

    save_json(save_path, results)

    print(f"\n=== Results for {path} ===")
    for key, value in results.items():
        print(f"\n{key.upper()}:\n{value}")

    print(f"\nSaved to: {save_path}")