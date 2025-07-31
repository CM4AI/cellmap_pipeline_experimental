import json

configs_file_path = "./configs/generate_hierarchy_configs.json"

with open (configs_file_path, 'r') as f:
    configs = json.load(f)

for config in configs:
    for k,v in config.items():
        if k.endswith("_run_id") and (not v or len(v.strip()) < 1):
            raise Exception(f"'{k}' needs to be provided")

print(configs)