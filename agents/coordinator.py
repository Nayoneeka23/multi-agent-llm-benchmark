import requests
import time
import sys

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from logger import log_message
from llm import call_llm
from message import Message


# ─── 3 WORKERS CONFIGURED ────────────────────────────────────────────────────
WORKERS = [
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003"
]


def send_task(
    task: str,
    network_condition: str = "baseline",
    image_path: str = None
):
    """Coordinator sends task to all workers and collects results."""

    results = []

    # -------------------------------------------------------------------------
    # STEP 1: Coordinator LLM breaks the big task into 3 subtasks
    # -------------------------------------------------------------------------
    prompt = (
        "Break this task into 3 clear subtasks "
        f"(one line each, numbered):\nTask: {task}"
    )

    llm_response, metrics = call_llm(
        prompt,
        model="mistral",
        stream=True,
        return_metrics=True
    )

    # Log coordinator's own LLM call
    log_message(
        sender="coordinator",
        receiver="llm",
        pattern="coordinator_worker",
        content=prompt,
        network_condition=network_condition,
        round_number=0,

        ollama_wall_time_ms=metrics["ollama_wall_time_ms"],
        model_compute_time_ms=metrics["model_compute_time_ms"],
        prompt_eval_time_ms=metrics["prompt_eval_time_ms"],
        generation_time_ms=metrics["generation_time_ms"],
        load_time_ms=metrics["load_time_ms"],
        ollama_total_duration_ms=metrics["ollama_total_duration_ms"],

        prompt_token_count=metrics["prompt_token_count"],
        generated_token_count=metrics["generated_token_count"],

        model="mistral",
        status="success"
    )

    # Get first 3 non-empty lines as subtasks
    subtasks = [
        line.strip()
        for line in llm_response.split("\n")
        if line.strip()
    ][:3]


    # -------------------------------------------------------------------------
    # STEP 2: Send each subtask to one worker
    # -------------------------------------------------------------------------
    for i, (worker_url, subtask) in enumerate(
        zip(WORKERS, subtasks)
    ):

        msg = Message(
            sender="coordinator",
            receiver=f"worker_{i+1}",
            content=subtask.strip(),
            round_number=1,
            pattern="coordinator_worker",
            network_condition=network_condition,
            image_path=image_path
        )

        # Convert Pydantic Message → normal dictionary
        payload = msg.model_dump()

        # Application-level end-to-end timer:
# includes HTTP transfer + network impairment + worker/LLM processing
        start = time.perf_counter()

        try:

            # Send subtask to worker
            r = requests.post(
                f"{worker_url}/process",
                json=payload,
                timeout=120
            )

            # Raise error if worker returned HTTP 4xx / 5xx
            r.raise_for_status()


            # Complete coordinator → worker → coordinator elapsed time
            end_to_end_ms = round(
                (time.perf_counter() - start) * 1000,
                2
            )


            # Convert returned JSON into Python dictionary
            response_data = r.json()
            log_message(
                sender="coordinator",
                receiver=f"worker_{i+1}",
                pattern="coordinator_worker",
                content=subtask.strip(),
                network_condition=network_condition,
                round_number=1,
                status="success"
            )


            # Save worker response + detailed metrics
            results.append({
                "worker": f"worker_{i+1}",

                "subtask": subtask.strip(),

                "response":
                    response_data.get("result", ""),

                "end_to_end_ms":
                    end_to_end_ms,

                "ollama_wall_time_ms":
                    response_data.get("ollama_wall_time_ms"),

                "model_compute_time_ms":
                    response_data.get("model_compute_time_ms"),

                "prompt_eval_time_ms":
                    response_data.get("prompt_eval_time_ms"),

                "generation_time_ms":
                    response_data.get("generation_time_ms"),

                "load_time_ms":
                    response_data.get("load_time_ms"),

                "ollama_total_duration_ms":
                    response_data.get(
                        "ollama_total_duration_ms"
                    ),

                "prompt_token_count":
                    response_data.get("prompt_token_count"),

                "generated_token_count":
                    response_data.get(
                        "generated_token_count"
                    ),

                "model":
                    response_data.get("model", "unknown")
            })


            print(
                f"[Coordinator] Worker {i+1} "
                f"responded in {end_to_end_ms}ms"
            )


        except Exception as e:

            # Log failed communication
            log_message(
                sender="coordinator",
                receiver=f"worker_{i+1}",
                pattern="coordinator_worker",
                content=subtask.strip(),
                network_condition=network_condition,
                round_number=1,
                status="failed",
                error=str(e)
            )

            print(
                f"[Coordinator] Worker {i+1} "
                f"failed: {e}"
            )


    return results


# -----------------------------------------------------------------------------
# RUN COORDINATOR DIRECTLY
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    task = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Summarize the benefits of edge computing"
    )

    condition = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "baseline"
    )

    image_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else None
    )


    print(
        f"\n[Coordinator] Starting task: {task}"
    )

    print(
        f"[Coordinator] Network condition: "
        f"{condition}\n"
    )


    results = send_task(
        task,
        condition,
        image_path
    )


    print(
        "\n[Coordinator] All results:"
    )

    for r in results:

        print(
            f"  {r['worker']} "
            f"({r['model']}): "
            f"{r['response'][:100]}..."
        )