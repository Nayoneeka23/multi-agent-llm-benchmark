import sys
import json
import csv
import os
import time
import subprocess
from datetime import datetime, timezone


sys.path.insert(0, '/home/jovyan/multi_agent_benchmark')
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from agents.coordinator import send_task
from benchmarks.prompts import SIMPLE_PROMPTS, COMPLEX_PROMPTS

NETWORK_CONDITIONS = [
    {"name": "baseline", "netem": None},
    {"name": "100ms_delay", "netem": "delay 100ms"},
    {"name": "jitter_50ms_gaussian", "netem": "delay 100ms 50ms distribution normal"},
    {"name": "packet_loss_5percent", "netem": "loss 5%"},
    {"name": "iot_edge_extreme", "netem": "delay 500ms loss 2% rate 256kbit"}
]
AGENT_PORTS = (8001, 8002, 8003)
RESET_CMD = "sudo tc qdisc del dev lo root 2>/dev/null; true"
EXPECTED_WORKERS = ("worker_1", "worker_2", "worker_3")
WORKER_METRICS = ("end_to_end_ms", "ollama_wall_time_ms", "model_compute_time_ms", "prompt_eval_time_ms", "generation_time_ms", "load_time_ms", "ollama_total_duration_ms", "prompt_token_count", "generated_token_count")

def run_tc(command):
    return subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

def reset_network():
    subprocess.run("sudo tc qdisc del dev lo root", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def apply_network(condition):
    reset_network()

    if condition["netem"]:
        try:
            run_tc("sudo tc qdisc add dev lo root handle 1: prio bands 3")
            run_tc(f"sudo tc qdisc add dev lo parent 1:3 handle 30: netem {condition['netem']}")

            for port in AGENT_PORTS:
                run_tc(f"sudo tc filter add dev lo protocol ip parent 1: prio 1 u32 match ip protocol 6 0xff match ip dport {port} 0xffff flowid 1:3")
                run_tc(f"sudo tc filter add dev lo protocol ip parent 1: prio 1 u32 match ip protocol 6 0xff match ip sport {port} 0xffff flowid 1:3")

            output = run_tc("sudo tc qdisc show dev lo").stdout
            if "netem" not in output:
                raise RuntimeError("tc netem was not actually activated.")

        except Exception as e:
            reset_network()
            raise RuntimeError(f"Failed to apply {condition['name']}: {e}")

        time.sleep(1)

    print(f"\n>>> Network condition verified: {condition['name']}")


def run_benchmark():
    results = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prompts = (
        [{"type": "simple", "index": i, "text": p} for i, p in enumerate(SIMPLE_PROMPTS)] +
        [{"type": "complex", "index": i, "text": p} for i, p in enumerate(COMPLEX_PROMPTS)]
    )
    total_runs = len(NETWORK_CONDITIONS) * len(prompts)
    count = 0

    for condition in NETWORK_CONDITIONS:
        apply_network(condition)
        try:
            for prompt in prompts:
                count += 1
                base_row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "prompt_run_id": f"{run_id}_{count:03d}",
                    "pattern": "coordinator_worker",
                    "network_condition": condition["name"],
                    "prompt_type": prompt["type"],
                    "prompt_index": prompt["index"],
                    "prompt_length_chars": len(prompt["text"])
                }
                print(f"\n[{count}/{total_runs}] {condition['name']} | {prompt['type']} #{prompt['index']}")

                start = time.perf_counter()
                try:
                    worker_results = send_task(
                        task=prompt["text"],
                        network_condition=condition["name"],
                        image_path=None
                    )
                    benchmark_total_time_ms = round((time.perf_counter() - start) * 1000, 2)
                    returned = {item.get("worker"): item for item in worker_results}

                    for worker_id in EXPECTED_WORKERS:
                        item = returned.get(worker_id)
                        if item:
                            row = {
                                **base_row,
                                "worker_id": worker_id,
                                "model": item.get("model", "mistral"),
                                "subtask": item.get("subtask"),
                                "response_length_chars": len(item.get("response", "")),
                                "benchmark_total_time_ms": benchmark_total_time_ms,
                                **{key: item.get(key) for key in WORKER_METRICS},
                                "status": "success",
                                "error": None
                            }
                        else:
                            row = {
                                **base_row,
                                "worker_id": worker_id,
                                "model": "mistral",
                                "subtask": None,
                                "response_length_chars": 0,
                                "benchmark_total_time_ms": benchmark_total_time_ms,
                                **{key: None for key in WORKER_METRICS},
                                "status": "failed",
                                "error": "No result returned by coordinator; see agent_messages.jsonl for the worker error."
                            }
                        results.append(row)

                    print(f"  -> workers={len(worker_results)}/3 | benchmark={benchmark_total_time_ms}ms")

                except Exception as e:
                    benchmark_total_time_ms = round((time.perf_counter() - start) * 1000, 2)
                    results.append({
                        **base_row,
                        "worker_id": None,
                        "model": None,
                        "subtask": None,
                        "response_length_chars": 0,
                        "benchmark_total_time_ms": benchmark_total_time_ms,
                        **{key: None for key in WORKER_METRICS},
                        "status": "failed",
                        "error": str(e)
                    })
                    print(f"  -> FAILED: {e}")
        finally:
            reset_network()

    out_dir = "/home/jovyan/multi_agent_benchmark/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)
    json_path = f"{out_dir}/benchmark_coordinator_worker_results_{run_id}.json"
    csv_path = f"{out_dir}/benchmark_coordinator_worker_results_{run_id}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nBenchmark complete: {len(results)} records")
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")


if __name__ == "__main__":
    run_benchmark()
