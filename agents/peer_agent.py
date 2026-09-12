import requests
import time
import sys

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from logger import log_message
from llm import call_llm
from message import P2PMessage


PEER_MODEL_MAP = {
    "peer_agent_1": "mistral",
    "peer_agent_2": "mistral",
    "peer_agent_3": "mistral",
}

PEER_PORTS = {
    "peer_agent_1": 8001,
    "peer_agent_2": 8002,
    "peer_agent_3": 8003,
}

METRIC_KEYS = (
    "ollama_wall_time_ms",
    "model_compute_time_ms",
    "prompt_eval_time_ms",
    "generation_time_ms",
    "load_time_ms",
    "ollama_total_duration_ms",
    "prompt_token_count",
    "generated_token_count",
)


def get_peer_url(peer_id):
    return f"http://127.0.0.1:{PEER_PORTS[peer_id]}"


def log_llm(my_id, prompt, condition, round_num, model, metrics):
    """Log one local peer -> LLM call."""
    log_message(
        sender=my_id,
        receiver="llm",
        pattern="peer_to_peer",
        content=prompt,
        network_condition=condition,
        round_number=round_num,
        ollama_wall_time_ms=metrics["ollama_wall_time_ms"],
        model_compute_time_ms=metrics["model_compute_time_ms"],
        prompt_eval_time_ms=metrics["prompt_eval_time_ms"],
        generation_time_ms=metrics["generation_time_ms"],
        load_time_ms=metrics["load_time_ms"],
        ollama_total_duration_ms=metrics["ollama_total_duration_ms"],
        prompt_token_count=metrics["prompt_token_count"],
        generated_token_count=metrics["generated_token_count"],
        model=model,
        status="success"
    )


def run_p2p_exchange(
    topic,
    my_id,
    peer_id,
    network_condition,
    max_rounds=3,
    image_path=None
):
    model = PEER_MODEL_MAP[my_id]
    history = []
    exchanges = []
    peer_url = get_peer_url(peer_id)

    # Round 0: initiator creates opening response
    prompt = (
        f"You are edge computing agent {my_id}. "
        f"Give a brief opening statement about: {topic}"
    )

    my_response, metrics = call_llm(
        prompt,
        model=model,
        stream=True,
        image_path=image_path if model == "llava" else None,
        return_metrics=True
    )

    log_llm(my_id, prompt, network_condition, 0, model, metrics)

    history.append({
        "role": my_id,
        "content": my_response
    })

    print(f"[{my_id}] Round 0: {my_response[:80]}...")

    # P2P HTTP rounds
    for round_num in range(1, max_rounds + 1):
        msg = P2PMessage(
            sender=my_id,
            receiver=peer_id,
            content=my_response,
            history=history,
            round_number=round_num,
            network_condition=network_condition,
            image_path=image_path
        )
        # Application-level end-to-end timer:
# includes HTTP transfer + network impairment + worker/LLM processing

        start = time.perf_counter()

        try:
            r = requests.post(
                f"{peer_url}/p2p_respond",
                json=msg.model_dump(),
                timeout=120
            )
            r.raise_for_status()

            end_to_end_ms = round((time.perf_counter() - start) * 1000, 2)
            data = r.json()
            peer_response = data.get("response", "")

            # Log outgoing message only after successful communication
            log_message(
                sender=my_id,
                receiver=peer_id,
                pattern=msg.pattern,
                content=my_response,
                network_condition=network_condition,
                round_number=round_num,
                status="success"
            )

            # Save measurements needed by run_p2p_benchmark.py
            exchanges.append({
                "sender": my_id,
                "receiver": peer_id,
                "round_number": round_num,
                "model": data.get("model", PEER_MODEL_MAP.get(peer_id, "mistral")),
                "response_length_chars": len(peer_response),
                "end_to_end_ms": end_to_end_ms,
                **{key: data.get(key) for key in METRIC_KEYS},
                "status": "success",
                "error": None
            })

        except Exception as e:
            log_message(
                sender=my_id,
                receiver=peer_id,
                pattern=msg.pattern,
                content=my_response,
                network_condition=network_condition,
                round_number=round_num,
                status="failed",
                error=str(e)
            )

            exchanges.append({
                "sender": my_id,
                "receiver": peer_id,
                "round_number": round_num,
                "model": PEER_MODEL_MAP.get(peer_id, "mistral"),
                "response_length_chars": 0,
                "end_to_end_ms": None,
                **{key: None for key in METRIC_KEYS},
                "status": "failed",
                "error": str(e)
            })

            print(f"[{my_id}] Round {round_num} failed: {e}")

            return {
                "history": history,
                "exchanges": exchanges,
                "status": "failed",
                "error": str(e)
            }

        # worker.py already logs peer -> initiator response
        history.append({
            "role": peer_id,
            "content": peer_response
        })

        print(
            f"[{my_id}] Round {round_num} "
            f"peer responded in {end_to_end_ms}ms"
        )

        # No need to generate another initiator response after final round
        if round_num == max_rounds:
            break

        history_text = "\n".join(
            f"{h['role']}: {h['content']}"
            for h in history[-4:]
        )

        prompt = (
            f"You are {my_id}. Continue with a new point:\n"
            f"{history_text}\n{my_id}:"
        )

        my_response, metrics = call_llm(
            prompt,
            model=model,
            stream=True,
            image_path=image_path if model == "llava" else None,
            return_metrics=True
        )

        log_llm(
            my_id,
            prompt,
            network_condition,
            round_num + 1,
            model,
            metrics
        )

        history.append({
            "role": my_id,
            "content": my_response
        })

    return {
        "history": history,
        "exchanges": exchanges,
        "status": "success",
        "error": None
    }


if __name__ == "__main__":
    topic = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "edge computing vs cloud computing tradeoffs"
    )

    condition = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "baseline"
    )

    result = run_p2p_exchange(
        topic=topic,
        my_id="peer_agent_1",
        peer_id="peer_agent_2",
        network_condition=condition,
        max_rounds=3
    )

    successful_rounds = sum(
        item["status"] == "success"
        for item in result["exchanges"]
    )

    print(
        f"\n[P2P] Completed {successful_rounds} "
        f"HTTP rounds, {len(result['history'])} history messages"
    )