import json
import os

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_scenario(scenario):
    return json.dumps(scenario, indent=2)