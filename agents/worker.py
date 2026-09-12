import sys
import uvicorn
from fastapi import FastAPI

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from logger import log_message
from llm import call_llm
from message import Message, P2PMessage, NegotiationMessage


app = FastAPI()

WORKER_ID = "worker_1"
MODEL = "mistral"

MODEL_MAP = {
    "worker_1": "mistral",
    "worker_2": "mistral",
    "worker_3": "mistral",
}


@app.post("/process")
def process(msg: Message):
    print(f"[{WORKER_ID}] Received: {msg.content[:60]}")

    prompt = (
        f"You are an edge computing agent using {MODEL}. "
        f"Complete this subtask concisely:\n{msg.content}"
    )

    # Get response + detailed timing metrics from llm.py
    response, metrics = call_llm(
    prompt,
    model=MODEL,
    stream =True,
    image_path=msg.image_path if MODEL == "llava" else None,
    return_metrics=True
    )

    log_message(
        sender=WORKER_ID,
        receiver=msg.sender,
        pattern=msg.pattern,
        content=response,
        network_condition=msg.network_condition,
        round_number=msg.round_number,

        ollama_wall_time_ms=metrics["ollama_wall_time_ms"],
        model_compute_time_ms=metrics["model_compute_time_ms"],
        prompt_eval_time_ms=metrics["prompt_eval_time_ms"],
        generation_time_ms=metrics["generation_time_ms"],
        load_time_ms=metrics["load_time_ms"],
        ollama_total_duration_ms=metrics["ollama_total_duration_ms"],

        prompt_token_count=metrics["prompt_token_count"],
        generated_token_count=metrics["generated_token_count"],

        model=MODEL,
        status="success"
        
    )

    print(
        f"[{WORKER_ID}] "
        f"model compute = {metrics['model_compute_time_ms']}ms, "
        f"Ollama wall = {metrics['ollama_wall_time_ms']}ms"
    )

    return {
        "worker": WORKER_ID,
        "result": response,

        # Kept temporarily so older benchmark code does not break
        "llm_time_ms": metrics["ollama_wall_time_ms"],

        "ollama_wall_time_ms": metrics["ollama_wall_time_ms"],
        "model_compute_time_ms": metrics["model_compute_time_ms"],
        "prompt_eval_time_ms": metrics["prompt_eval_time_ms"],
        "generation_time_ms": metrics["generation_time_ms"],
        "load_time_ms": metrics["load_time_ms"],
        "ollama_total_duration_ms": metrics["ollama_total_duration_ms"],

        "prompt_token_count": metrics["prompt_token_count"],
        "generated_token_count": metrics["generated_token_count"],

        "model": MODEL
    }


@app.post("/p2p_respond")
def p2p_respond(msg: P2PMessage):
    history_text = "\n".join(
        [f"{h['role']}: {h['content']}" for h in msg.history[-4:]]
    )

    prompt = (
        f"You are edge agent {msg.receiver}. "
        f"Respond with a new insight:\n"
        f"{history_text}\n"
        f"{msg.receiver}:"
    )

    # Get response + detailed timing metrics
    response, metrics = call_llm(
    prompt,
    model=MODEL,
    stream =True,
    image_path=msg.image_path if MODEL == "llava" else None,
    return_metrics=True
    )

    log_message(
        sender=msg.receiver,
        receiver=msg.sender,
        pattern=msg.pattern,
        content=response,
        network_condition=msg.network_condition,
        round_number=msg.round_number,

        ollama_wall_time_ms=metrics["ollama_wall_time_ms"],
        model_compute_time_ms=metrics["model_compute_time_ms"],
        prompt_eval_time_ms=metrics["prompt_eval_time_ms"],
        generation_time_ms=metrics["generation_time_ms"],
        load_time_ms=metrics["load_time_ms"],
        ollama_total_duration_ms=metrics["ollama_total_duration_ms"],

        prompt_token_count=metrics["prompt_token_count"],
        generated_token_count=metrics["generated_token_count"],

        model=MODEL,
        status="success"
    )

    print(
        f"[P2P {WORKER_ID}] Round {msg.round_number}: "
        f"model compute = {metrics['model_compute_time_ms']}ms, "
        f"Ollama wall = {metrics['ollama_wall_time_ms']}ms"
    )

    return {
        "response": response,

        # Kept for compatibility with existing scripts
        "llm_time_ms": metrics["ollama_wall_time_ms"],

        "ollama_wall_time_ms": metrics["ollama_wall_time_ms"],
        "model_compute_time_ms": metrics["model_compute_time_ms"],
        "prompt_eval_time_ms": metrics["prompt_eval_time_ms"],
        "generation_time_ms": metrics["generation_time_ms"],
        "load_time_ms": metrics["load_time_ms"],
        "ollama_total_duration_ms": metrics["ollama_total_duration_ms"],

        "prompt_token_count": metrics["prompt_token_count"],
        "generated_token_count": metrics["generated_token_count"],

        "model": MODEL
    }


@app.post("/negotiate")
def negotiate(msg: NegotiationMessage):

    prompt = f"""You are negotiation agent {msg.receiver} using {MODEL}. Round {msg.round_number}.
The other agent proposed: {msg.content}
If acceptable, start with AGREED: and summarize the deal.
Otherwise start with COUNTER: and give a counter-proposal (2 sentences)."""

    # Get response + detailed timing metrics
    response, metrics = call_llm(
    prompt,
    model=MODEL,
    stream =True,
    image_path=msg.image_path if MODEL == "llava" else None,
    return_metrics=True
    )

    agreed = response.strip().startswith("AGREED")

    log_message(
        sender=msg.receiver,
        receiver=msg.sender,
        pattern=msg.pattern,
        content=response,
        network_condition=msg.network_condition,
        round_number=msg.round_number,

        ollama_wall_time_ms=metrics["ollama_wall_time_ms"],
        model_compute_time_ms=metrics["model_compute_time_ms"],
        prompt_eval_time_ms=metrics["prompt_eval_time_ms"],
        generation_time_ms=metrics["generation_time_ms"],
        load_time_ms=metrics["load_time_ms"],
        ollama_total_duration_ms=metrics["ollama_total_duration_ms"],

        prompt_token_count=metrics["prompt_token_count"],
        generated_token_count=metrics["generated_token_count"],

        model=MODEL,
        status="success"
    )

    print(
        f"[Negotiation {WORKER_ID}] Round {msg.round_number}: "
        f"agreed={agreed}, "
        f"model compute = {metrics['model_compute_time_ms']}ms, "
        f"Ollama wall = {metrics['ollama_wall_time_ms']}ms"
    )

    return {
        "counter_proposal": response,
        "agreed": agreed,

        # Kept for compatibility with existing benchmark scripts
        "llm_time_ms": metrics["ollama_wall_time_ms"],

        "ollama_wall_time_ms": metrics["ollama_wall_time_ms"],
        "model_compute_time_ms": metrics["model_compute_time_ms"],
        "prompt_eval_time_ms": metrics["prompt_eval_time_ms"],
        "generation_time_ms": metrics["generation_time_ms"],
        "load_time_ms": metrics["load_time_ms"],
        "ollama_total_duration_ms": metrics["ollama_total_duration_ms"],

        "prompt_token_count": metrics["prompt_token_count"],
        "generated_token_count": metrics["generated_token_count"],

        "model": MODEL
    }


if __name__ == "__main__":
    WORKER_ID = sys.argv[1] if len(sys.argv) > 1 else "worker_1"
    PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
    MODEL = MODEL_MAP.get(WORKER_ID, "mistral")

    print(
        f"[{WORKER_ID}] Starting on port {PORT} "
        f"using model: {MODEL}"
    )

    uvicorn.run(app, host="127.0.0.1", port=PORT)