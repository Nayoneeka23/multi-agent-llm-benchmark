import json
import os
from datetime import datetime, timezone

LOG_FILE = os.path.expanduser(
    "~/multi_agent_benchmark/benchmarks/results/agent_messages.jsonl"
)

os.makedirs(
    os.path.dirname(LOG_FILE),
    exist_ok=True
)


def log_message(
    sender,
    receiver,
    pattern,
    content,
    network_condition,
    round_number=None,

    # Old field kept for compatibility
    llm_response_time_ms=None,

    # New proper timing fields
    ollama_wall_time_ms=None,
    model_compute_time_ms=None,
    prompt_eval_time_ms=None,
    generation_time_ms=None,
    load_time_ms=None,
    ollama_total_duration_ms=None,

    # Useful LLM information
    prompt_token_count=None,
    generated_token_count=None,
    model=None,
    status=None,
    error=None
):

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "sender": sender,
        "receiver": receiver,
        "pattern": pattern,

        "message_size_bytes": len(
            content.encode("utf-8")
        ),

        "round_number": round_number,
        "network_condition": network_condition,

        # Legacy field
        "llm_response_time_ms": llm_response_time_ms,

        # New measurements
        "ollama_wall_time_ms": ollama_wall_time_ms,
        "model_compute_time_ms": model_compute_time_ms,
        "prompt_eval_time_ms": prompt_eval_time_ms,
        "generation_time_ms": generation_time_ms,
        "load_time_ms": load_time_ms,
        "ollama_total_duration_ms": ollama_total_duration_ms,

        # Token/model information
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "model": model,

        "content_preview": content[:100],
        "status": status,
        "error": error,
    }

    with open(LOG_FILE, "a") as f:
        f.write(
            json.dumps(record) + "\n"
        )

    return record