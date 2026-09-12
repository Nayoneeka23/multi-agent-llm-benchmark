import sys
import json
import csv
import os
import time
import subprocess
from datetime import datetime, timezone

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark')
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from agents.peer_agent import run_p2p_exchange
from benchmarks.prompts import SIMPLE_PROMPTS, COMPLEX_PROMPTS

MAX_ROUNDS = 3

NETWORK_CONDITIONS = [
    {"name": "baseline", "netem": None},
    {"name": "100ms_delay", "netem": "delay 100ms"},
    {"name": "jitter_50ms_gaussian", "netem": "delay 100ms 50ms distribution normal"},
    {"name": "packet_loss_5percent", "netem": "loss 5%"},
    {"name": "iot_edge_extreme", "netem": "delay 500ms loss 2% rate 256kbit"}
]
AGENT_PORTS = (8001, 8002, 8003)
RESET_CMD = "sudo tc qdisc del dev lo root 2>/dev/null; true"

PEER_PAIRS = [
    ("peer_agent_1", "peer_agent_2"),
    ("peer_agent_2", "peer_agent_3"),
    ("peer_agent_1", "peer_agent_3")
]

METRIC_KEYS = (
    "end_to_end_ms", "ollama_wall_time_ms", "model_compute_time_ms",
    "prompt_eval_time_ms", "generation_time_ms", "load_time_ms",
    "ollama_total_duration_ms", "prompt_token_count", "generated_token_count"
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


def run_p2p_benchmark():
    results = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    prompts = (
        [{"type": "simple", "index": i, "text": p} for i, p in enumerate(SIMPLE_PROMPTS)] +
        [{"type": "complex", "index": i, "text": p} for i, p in enumerate(COMPLEX_PROMPTS)]
    )

    total_runs = len(NETWORK_CONDITIONS) * len(prompts) * len(PEER_PAIRS)
    count = 0

    for condition in NETWORK_CONDITIONS:
        apply_network(condition)

        try:
            for prompt in prompts:
                for my_id, peer_id in PEER_PAIRS:
                    count += 1
                    pair_run_id = f"{run_id}_{count:03d}"

                    base_row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_id,
                        "pair_run_id": pair_run_id,
                        "pattern": "peer_to_peer",
                        "network_condition": condition["name"],
                        "prompt_type": prompt["type"],
                        "prompt_index": prompt["index"],
                        "prompt_length_chars": len(prompt["text"]),
                        "initiator_id": my_id,
                        "peer_id": peer_id,
                        "max_rounds": MAX_ROUNDS
                    }

                    print(
                        f"\n[{count}/{total_runs}] {condition['name']} | "
                        f"{prompt['type']} #{prompt['index']} | {my_id} -> {peer_id}"
                    )

                    start = time.perf_counter()

                    try:
                        result = run_p2p_exchange(
                            topic=prompt["text"],
                            my_id=my_id,
                            peer_id=peer_id,
                            network_condition=condition["name"],
                            max_rounds=MAX_ROUNDS,
                            image_path=None
                        )

                        benchmark_total_time_ms = round(
                            (time.perf_counter() - start) * 1000, 2
                        )

                        history = result.get("history", [])
                        exchanges = result.get("exchanges", [])
                        completed_rounds = sum(
                            item.get("status") == "success"
                            for item in exchanges
                        )

                        if not exchanges:
                            results.append({
                                **base_row,
                                "round_number": None,
                                "model": None,
                                "response_length_chars": 0,
                                "completed_rounds": 0,
                                "history_message_count": len(history),
                                "benchmark_total_time_ms": benchmark_total_time_ms,
                                **{key: None for key in METRIC_KEYS},
                                "status": "failed",
                                "error": result.get("error") or "No exchange records returned."
                            })
                        else:
                            for exchange in exchanges:
                                results.append({
                                    **base_row,
                                    "round_number": exchange.get("round_number"),
                                    "model": exchange.get("model", "mistral"),
                                    "response_length_chars": exchange.get("response_length_chars", 0),
                                    "completed_rounds": completed_rounds,
                                    "history_message_count": len(history),
                                    "benchmark_total_time_ms": benchmark_total_time_ms,
                                    **{key: exchange.get(key) for key in METRIC_KEYS},
                                    "status": exchange.get("status", "success"),
                                    "error": exchange.get("error")
                                })

                        print(
                            f"  -> rounds={completed_rounds}/{MAX_ROUNDS} | "
                            f"history={len(history)} | benchmark={benchmark_total_time_ms}ms"
                        )

                    except Exception as e:
                        benchmark_total_time_ms = round(
                            (time.perf_counter() - start) * 1000, 2
                        )

                        results.append({
                            **base_row,
                            "round_number": None,
                            "model": None,
                            "response_length_chars": 0,
                            "completed_rounds": 0,
                            "history_message_count": 0,
                            "benchmark_total_time_ms": benchmark_total_time_ms,
                            **{key: None for key in METRIC_KEYS},
                            "status": "failed",
                            "error": str(e)
                        })

                        print(f"  -> FAILED: {e}")

        finally:
            reset_network()

    out_dir = "/home/jovyan/multi_agent_benchmark/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)

    json_path = f"{out_dir}/benchmark_peer_to_peer_results_{run_id}.json"
    csv_path = f"{out_dir}/benchmark_peer_to_peer_results_{run_id}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nP2P benchmark complete: {len(results)} records")
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")


if __name__ == "__main__":
    run_p2p_benchmark()