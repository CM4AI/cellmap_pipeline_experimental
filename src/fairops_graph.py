import mlflow
from mlflow.tracking import MlflowClient
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter

NEO4J_URI = "neo4j+s://f2de980c.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "kWsYUax7JsaNQXOVpm-kzwMfjZZilQ5aLWxSmGBe32g"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5005"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

def fetch_all_experiment_ids():
    client = MlflowClient()
    experiments = client.search_experiments(filter_string="")
    return [exp.experiment_id for exp in experiments]

def fetch_runs_for_experiments(experiment_ids):
    client = MlflowClient()
    all_runs = []
    for exp_id in experiment_ids:
        exp = client.get_experiment(exp_id)
        exp_name = exp.name

        runs = client.search_runs([exp_id], filter_string="", max_results=1000)
        for run in runs:
            if "mlflow.parentRunId" in run.data.tags:
                run_id = run.info.run_id
                params = {k: str(v) for k, v in run.data.params.items()}
                all_runs.append({
                    "run_id": run_id,
                    "params": params,
                    "experiment_name": exp_name
                })

                # 🔍 Debug print
                print(f"\nRun ID: {run_id}")
                if params:
                    print("  Params:")
                    for k, v in params.items():
                        print(f"    {k}: {v}")
                else:
                    print("No parameters found.")

    print(f"\n Total child runs fetched: {len(all_runs)}")
    return all_runs

def create_nodes_and_edges (driver, runs):
    with driver.session() as session:
        for run in runs:
            session.run(
                """
                MERGE (r:Run {run_id: $run_id})
                SET r.experiment_name = $experiment_name,
                    r += $params
                """,
                run_id=run["run_id"],
                params=run["params"],
                experiment_name=run["experiment_name"]
            )

        run_ids = set(run["run_id"] for run in runs)

        for run in runs:
            src_id = run["run_id"]
            for key, val in run["params"].items():
                if key.endswith("_run_id") and val in run_ids:
                    session.run(
                        """
                        MATCH (src:Run {run_id: $src_id}), (dst:Run {run_id: $dst_id})
                        MERGE (src)-[:LINKS_TO]->(dst)
                        """,
                        src_id=src_id,
                        dst_id=val
                    )
                    print(f"Created edge: {src_id} ➜ {val} ({key})")



if __name__ == "__main__":

    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))

    experiment_ids = fetch_all_experiment_ids()
    print(f"Found experiment IDs: {experiment_ids}")

    runs = fetch_runs_for_experiments(experiment_ids)
    print(f"Fetched {len(runs)} runs from all experiments")

    create_nodes_and_edges(driver, runs)
    print("Created all nodes and relationships in Neo4j")

    driver.close()
    print("\n Network graph generation is completed")
