import os
import yaml
import json
import subprocess
import time
from collections import deque

def find_result_dirs(root_dir):
    """
    Yield directories that contain:
      - trial_logs.jsonl
      - .hydra/config.yaml
      - .hydra/overrides.yaml
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".hydra" not in dirnames:
            continue
        hydra_dir = os.path.join(dirpath, ".hydra")
        hydra_files = os.listdir(hydra_dir)
        if "config.yaml" in hydra_files and "overrides.yaml" in hydra_files:
            yield dirpath

def read_n_trials_from_config(config_path):
    """
    Extract n_trials from config.yaml.
    """
    with open(config_path, "r") as f:
        for line in f:
            if "n_trials:" in line:
                return int(line.strip().split(":")[1])
    raise ValueError("n_trials not found")

def read_n_trials_from_last_line(trial_logs_path):
    """
    Read last line of trial_logs.jsonl and get n_trials.
    """
    with open(trial_logs_path, "r") as f:
        try:
            last_line = deque(f, maxlen=1)[0]
        except IndexError:
            raise ValueError("File is empty.")
    record = json.loads(last_line)
    if "n_trials" not in record:
        raise ValueError(f"'n_trials' not found in last line of {trial_logs_path}")
    return record["n_trials"]

def read_overrides(overrides_path):
    """
    Read overrides.yaml and return as list of arguments.
    """
    with open(overrides_path, "r") as f:
        overrides = yaml.safe_load(f)
    if not isinstance(overrides, list):
        raise ValueError(f"{overrides_path} does not contain a list of overrides.")
    return overrides

def main(root_dir):
    all_ok = True
    for dirpath in find_result_dirs(root_dir):
        error = False
        config_path = os.path.join(dirpath, ".hydra/config.yaml")
        trial_logs_path = os.path.join(dirpath, "trial_logs.jsonl")

        try:
            n_trials_expected = read_n_trials_from_config(config_path)
            n_trials_actual = read_n_trials_from_last_line(trial_logs_path)
            if n_trials_actual != n_trials_expected:
                print(f"[INCOMPLETE] {dirpath}: last n_trials={n_trials_actual} (expected {n_trials_expected})")
                error = True
        except Exception as e:
            error = True
            print(f"[INCOMPLETE] {dirpath}: trail_logs missing")
        
        if error:
            all_ok = False
            overrides_path = os.path.join(dirpath, ".hydra/overrides.yaml")
            overrides = read_overrides(overrides_path)
            override_str = " ".join(overrides)
            
            command = f"sbatch start_create_cmds.sh {override_str}"
            process = subprocess.run(command, shell=True)
            if process.returncode != 0:
                print(f"Command failed: {command}")

    if all_ok:
        print("\n✅ All runs are complete.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
