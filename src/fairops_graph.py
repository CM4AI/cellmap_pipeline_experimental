import mlflow
from mlflow.tracking import MlflowClient
from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://f2de980c.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "kWsYUax7JsaNQXOVpm-kzwMfjZZilQ5aLWxSmGBe32g"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5005"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

def sanitize_params(params):
    safe_params = {}
    for k, v in params.items():
        if isinstance(v, (int, float, bool)):
            safe_params[k] = v
        elif isinstance(v, str):
            if v.isdigit():
                safe_params[k] = int(v)
            else:
                try:
                    safe_params[k] = float(v)
                except ValueError:
                    safe_params[k] = v.strip()
        else:
            safe_params[k] = str(v)
    return safe_params

def fetch_all_experiments():
    experiments = client.search_experiments(filter_string="")
    print(f"Fetched {len(experiments)} experiments.")
    return [(exp.experiment_id, exp.name) for exp in experiments]

def fetch_parent_runs(experiment_id):
    runs = client.search_runs([experiment_id], filter_string="", max_results=1000)
    parent_runs = []
    for run in runs:
        if "mlflow.parentRunId" not in run.data.tags:
            run_id = run.info.run_id.strip()
            run_name = run.data.tags.get("mlflow.runName", "(no name)").strip()
            parent_runs.append({"run_id": run_id, "run_name": run_name})
    print(f"Experiment {experiment_id}: fetched {len(parent_runs)} parent runs.")
    return parent_runs

def fetch_child_runs(parent_run_id):
    child_runs = []
    experiments = client.search_experiments(filter_string="")
    for exp in experiments:
        runs = client.search_runs([exp.experiment_id], filter_string="", max_results=1000)
        for run in runs:
            tags = run.data.tags
            if tags.get("mlflow.parentRunId", None) == parent_run_id:
                run_id = run.info.run_id.strip()
                run_name = tags.get("mlflow.runName", "(no name)").strip()
                params = sanitize_params(run.data.params)
                child_runs.append({
                    "run_id": run_id,
                    "run_name": run_name,
                    "parent_run_id": parent_run_id,
                    "params": params,
                    "experiment_id": exp.experiment_id,
                    "experiment_name": exp.name
                })
    print(f"Parent run {parent_run_id}: fetched {len(child_runs)} child runs.")
    return child_runs

def create_nodes_and_relationships(driver, experiments, parent_runs_map, child_runs_map):
    with driver.session() as session:

        for exp_id, exp_name in experiments:
            session.run(
                """
                MERGE (e:Experiment {experiment_id: $exp_id})
                SET e.experiment_name = $exp_name
                """,
                exp_id=exp_id,
                exp_name=exp_name
            )
        print(f"Created {len(experiments)} experiment nodes.")

        total_parent_runs = 0
        for exp_id, parents in parent_runs_map.items():
            for p in parents:
                session.run(
                    """
                    MERGE (p:ParentRun {run_id: $run_id})
                    SET p.run_name = $run_name,
                        p.name = $run_name
                    """,
                    run_id=p["run_id"],
                    run_name=p["run_name"]
                )
                session.run(
                    """
                    MATCH (e:Experiment {experiment_id: $exp_id}), (p:ParentRun {run_id: $run_id})
                    MERGE (e)-[:RELATED]-(p)
                    """,
                    exp_id=exp_id,
                    run_id=p["run_id"]
                )
                total_parent_runs += 1
        print(f"Created and linked {total_parent_runs} parent run nodes to experiments.")

        total_child_runs = 0
        for parent_id, children in child_runs_map.items():
            for c in children:
                safe_params = sanitize_params(c["params"])
                session.run(
                    """
                    MERGE (c:ChildRun {run_id: $run_id})
                    SET c.run_name = $run_name,
                        c.name = $run_name,
                        c.parent_run_id = $parent_run_id,
                        c.experiment_id = $experiment_id,
                        c.experiment_name = $experiment_name,
                        c += $params
                    """,
                    run_id=c["run_id"],
                    run_name=c["run_name"],
                    parent_run_id=c["parent_run_id"],
                    experiment_id=c["experiment_id"],
                    experiment_name=c["experiment_name"],
                    params=safe_params
                )
                session.run(
                    """
                    MATCH (parent:ParentRun {run_id: $parent_run_id}), (child:ChildRun {run_id: $child_run_id})
                    MERGE (parent)-[:RELATED]-(child)
                    """,
                    parent_run_id=c["parent_run_id"],
                    child_run_id=c["run_id"]
                )
                total_child_runs += 1
        print(f"Created and linked {total_child_runs} child run nodes.")

        all_child_run_ids = set()
        for children in child_runs_map.values():
            for c in children:
                all_child_run_ids.add(c["run_id"])

        for children in child_runs_map.values():
            for c in children:
                src_id = c["run_id"]
                params = c["params"]
                for k, v in params.items():
                    if isinstance(v, str):
                        v_stripped = v.strip()
                    else:
                        v_stripped = v

                    if k.endswith("_run_id") and v_stripped in all_child_run_ids:
                        session.run(
                            """
                            MATCH (src:ChildRun {run_id: $src_id}), (dst:ChildRun {run_id: $dst_id})
                            MERGE (dst)-[:LINKS_TO]->(src)
                            """,
                            src_id=src_id,
                            dst_id=v_stripped
                        )
                        print(f"Created directional LINKS_TO link: {v_stripped} -> {src_id} ({k})")



def main():
    experiments = fetch_all_experiments()

    parent_runs_map = {}
    child_runs_map = {}

    for exp_id, exp_name in experiments:
        parents = fetch_parent_runs(exp_id)
        parent_runs_map[exp_id] = parents

        for p in parents:
            children = fetch_child_runs(p["run_id"])
            child_runs_map[p["run_id"]] = children

    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))

    create_nodes_and_relationships(driver, experiments, parent_runs_map, child_runs_map)

    driver.close()
    print("\nNetwork graph generation is completed")

if __name__ == "__main__":
    main()
