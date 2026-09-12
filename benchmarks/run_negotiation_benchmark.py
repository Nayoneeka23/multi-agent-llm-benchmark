import sys
import json
import csv
import os
import time
import subprocess

from datetime import datetime, timezone

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark')
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from agents.negotiation_agent import run_negotiation
from benchmarks.prompts import SIMPLE_PROMPTS, COMPLEX_PROMPTS

MAX_ROUNDS = 4

NETWORK_CONDITIONS = [
    {"name": "baseline", "netem": None},
    {"name": "100ms_delay", "netem": "delay 100ms"},
    {"name": "jitter_50ms_gaussian", "netem": "delay 100ms 50ms distribution normal"},
    {"name": "packet_loss_5percent", "netem": "loss 5%"},
    {"name": "iot_edge_extreme", "netem": "delay 500ms loss 2% rate 256kbit"}
]
AGENT_PORTS = (8001, 8002, 8003)
RESET_CMD = "sudo tc qdisc del dev lo root 2>/dev/null; true"

RESPONDER_METRICS = (
    "end_to_end_ms",
    "ollama_wall_time_ms",
    "model_compute_time_ms",
    "prompt_eval_time_ms",
    "generation_time_ms",
    "load_time_ms",
    "ollama_total_duration_ms",
)


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


def average(records, key):
    values = [item.get(key) for item in records if isinstance(item.get(key), (int, float))]
    return round(sum(values) / len(values), 2) if values else None


def total(records, key):
    values = [item.get(key) for item in records if isinstance(item.get(key), (int, float))]
    return sum(values) if values else None


def summarize_responder_metrics(responses):
    summary = {"responder_request_count": len(responses)}

    for key in RESPONDER_METRICS:
        summary[f"responder_avg_{key}"] = average(responses, key)

    summary["responder_prompt_token_count_total"] = total(responses, "prompt_token_count")
    summary["responder_generated_token_count_total"] = total(responses, "generated_token_count")
    return summary


def run_negotiation_benchmark():
    results = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    all_prompts = (
        [{"type": "simple", "index": i, "text": p} for i, p in enumerate(SIMPLE_PROMPTS)] +
        [{"type": "complex", "index": i, "text": p} for i, p in enumerate(COMPLEX_PROMPTS)]
    )

    total_runs = len(NETWORK_CONDITIONS) * len(all_prompts)
    count = 0
    for condition in NETWORK_CONDITIONS:
        apply_network(condition)
        try:
            for prompt in all_prompts:
                count += 1
                print(f"\n[{count}/{total_runs}] {condition['name']} | {prompt['type']} #{prompt['index']}")
    
                base_row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "pattern": "negotiation",
                    "network_condition": condition["name"],
                    "prompt_type": prompt["type"],
                    "prompt_index": prompt["index"],
                    "prompt_length": len(prompt["text"]),
                    "max_rounds": MAX_ROUNDS,
                }
                try:
                    start = time.perf_counter()
                    result = run_negotiation(topic=prompt["text"], network_condition=condition["name"], max_rounds=MAX_ROUNDS)
                    benchmark_total_time_ms = round((time.perf_counter() - start) * 1000, 2)
                    responses = result.get("responses", [])
    
                    row = {
                        **base_row,
                        "rounds": result.get("rounds"),
                        "agreed": result.get("agreed"),
                        "agreed_by": result.get("agreed_by", "none"),
                        "final_proposal": result.get("final", ""),
                        "benchmark_total_time_ms": benchmark_total_time_ms,
                        "status": result.get("status", "success"),
                        "error": result.get("error"),
                        **summarize_responder_metrics(responses)
                    }
                except Exception as e:
                    row = {
                        **base_row,
                        "rounds": None,
                        "agreed": None,
                        "agreed_by": "none",
                        "final_proposal": "",
                        "benchmark_total_time_ms": None,
                        "status": "failed",
                        "error": str(e),
                        **summarize_responder_metrics([])
                    }
    
                results.append(row)
        finally:
                reset_network()

    out_dir = "/home/jovyan/multi_agent_benchmark/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)

    json_path = f"{out_dir}/benchmark_negotiation_results_{run_id}.json"
    csv_path = f"{out_dir}/benchmark_negotiation_results_{run_id}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nBenchmark complete: {len(results)} runs")
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")


if __name__ == "__main__":
    run_negotiation_benchmark()