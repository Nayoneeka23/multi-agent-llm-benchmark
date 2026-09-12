import requests
import time
import sys

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from logger import log_message
from llm import call_llm
from message import NegotiationMessage


AGENT_MODEL_MAP = {
    "agent_1": "mistral",
    "agent_2": "mistral",
    "agent_3": "mistral",
}

AGENT_PORTS = {
    "agent_1": 8001,
    "agent_2": 8002,
    "agent_3": 8003,
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


def get_agent_url(agent_id):
    return f"http://127.0.0.1:{AGENT_PORTS[agent_id]}"


def log_llm(agent_id, prompt, condition, round_num, model, metrics):
    log_message(
        sender=agent_id,
        receiver="llm",
        pattern="negotiation",
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


def agent_propose(
    agent_id,
    prompt,
    network_condition,
    round_number,
    image_path=None
):
    model = AGENT_MODEL_MAP[agent_id]

    response, metrics = call_llm(
        prompt,
        model=model,
        stream=True,
        image_path=image_path if model == "llava" else None,
        return_metrics=True
    )

    log_llm(
        agent_id,
        prompt,
        network_condition,
        round_number,
        model,
        metrics
    )

    print(
        f"[{agent_id} | {model}] "
        f"Round {round_number}: {response[:80]}..."
    )

    return response


def run_negotiation(
    topic,
    network_condition="baseline",
    max_rounds=4,
    image_path=None
):
    agents = list(AGENT_MODEL_MAP)
    initiator = agents[0]
    responders = agents[1:]

    all_responses = []
    had_request_failure = False
    failure_errors = []

    print(f"\n[Negotiation] Topic: {topic}")
    print(f"[Negotiation] Network condition: {network_condition}")
    print(f"[Negotiation] Agents: {agents}")

    prompt = (
        f"You are negotiation agent {initiator}. "
        f"Make a specific initial proposal about: {topic}. "
        f"Be concise (2-3 sentences)."
    )

    current_proposal = agent_propose(
        initiator,
        prompt,
        network_condition,
        round_number=0,
        image_path=image_path
    )

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        round_responses = []

        for responder in responders:
            msg = NegotiationMessage(
                sender=initiator,
                receiver=responder,
                content=current_proposal,
                round_number=round_num,
                network_condition=network_condition,
                image_path=image_path
            )

            # Application-level end-to-end timer:
# includes HTTP transfer + network impairment + worker/LLM processing

            start = time.perf_counter()

            try:
                r = requests.post(
                    f"{get_agent_url(responder)}/negotiate",
                    json=msg.model_dump(),
                    timeout=120
                )

                r.raise_for_status()

                end_to_end_ms = round(
                    (time.perf_counter() - start) * 1000,
                    2
                )

                data = r.json()

                log_message(
                    sender=initiator,
                    receiver=responder,
                    pattern=msg.pattern,
                    content=current_proposal,
                    network_condition=network_condition,
                    round_number=round_num,
                    status="success"
                )

                response_record = {
                    "agent": responder,
                    "model": data.get(
                        "model",
                        AGENT_MODEL_MAP[responder]
                    ),
                    "counter": data.get(
                        "counter_proposal",
                        ""
                    ),
                    "agreed": data.get(
                        "agreed",
                        False
                    ),
                    "end_to_end_ms": end_to_end_ms,
                    **{
                        key: data.get(key)
                        for key in METRIC_KEYS
                    },
                    "status": "success",
                    "error": None
                }

                round_responses.append(response_record)
                all_responses.append(response_record)

                print(
                    f"  [{responder} | "
                    f"{response_record['model']}] "
                    f"end-to-end {end_to_end_ms}ms: "
                    f"{response_record['counter'][:80]}..."
                )

            except Exception as e:
                had_request_failure = True

                failure_errors.append(
                    f"{responder} round {round_num}: {e}"
                )

                log_message(
                    sender=initiator,
                    receiver=responder,
                    pattern=msg.pattern,
                    content=current_proposal,
                    network_condition=network_condition,
                    round_number=round_num,
                    status="failed",
                    error=str(e)
                )

                failure_record = {
                    "agent": responder,
                    "model": AGENT_MODEL_MAP[responder],
                    "counter": "",
                    "agreed": False,
                    "end_to_end_ms": None,
                    **{
                        key: None
                        for key in METRIC_KEYS
                    },
                    "status": "failed",
                    "error": str(e)
                }

                round_responses.append(failure_record)
                all_responses.append(failure_record)

                print(
                    f"  [{responder}] "
                    f"Round {round_num} failed: {e}"
                )

        full_agreement = (
            len(round_responses) == len(responders)
            and all(
                item["agreed"]
                for item in round_responses
            )
        )

        if full_agreement:
            print(
                f"\n[Negotiation] "
                f"Full agreement reached at round {round_num}!"
            )

            return {
                "rounds": round_num,
                "agreed": True,
                "agreed_by": ",".join(responders),
                "final": current_proposal,
                "responses": all_responses,
                "status": (
                    "failed"
                    if had_request_failure
                    else "success"
                ),
                "error": (
                    "; ".join(failure_errors)
                    if failure_errors
                    else None
                )
            }

        if round_num == max_rounds:
            break

        feedback = " | ".join(
            item["counter"][:100]
            for item in round_responses
            if item["counter"]
        )

        prompt = (
            f"You are {initiator} negotiating about {topic}. "
            f"The other agents responded: {feedback}. "
            f"Refine your proposal considering their feedback "
            f"(2-3 sentences)."
        )

        current_proposal = agent_propose(
            initiator,
            prompt,
            network_condition,
            round_number=round_num,
            image_path=image_path
        )

    print(
        "\n[Negotiation] "
        "Max rounds reached. No full agreement."
    )

    return {
        "rounds": max_rounds,
        "agreed": False,
        "agreed_by": "none",
        "final": current_proposal,
        "responses": all_responses,
        "status": (
            "failed"
            if had_request_failure
            else "success"
        ),
        "error": (
            "; ".join(failure_errors)
            if failure_errors
            else None
        )
    }


if __name__ == "__main__":
    topic = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "optimal bandwidth allocation for edge nodes"
    )

    condition = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "baseline"
    )

    image = (
        sys.argv[3]
        if len(sys.argv) > 3
        else None
    )

    result = run_negotiation(
        topic=topic,
        network_condition=condition,
        max_rounds=4,
        image_path=image
    )

    print(
        f"\n[Negotiation] Final Result: {result}"
    )