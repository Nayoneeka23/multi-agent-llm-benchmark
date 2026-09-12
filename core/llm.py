import requests
import base64
import time
import json


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def ns_to_ms(value):
    """Convert nanoseconds to milliseconds."""
    if value is None:
        return None

    return round(value / 1_000_000, 2)


def call_llm(
    prompt,
    model="mistral",
    stream=False,
    image_path=None,
    return_metrics=False
):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    # -------------------------------------------------
    # IMAGE SUPPORT
    # -------------------------------------------------
    # If an image path is supplied, read the image,
    # convert it to Base64 and add it to Ollama's request.
    if image_path:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(
                f.read()
            ).decode("utf-8")

        payload["images"] = [image_b64]


    # -------------------------------------------------
    # START WALL-CLOCK TIMER
    # -------------------------------------------------
    # Measures the complete Worker <-> Ollama operation.
    start = time.perf_counter()


    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,

        # Important when stream=True:
        # requests will allow us to read the response
        # incrementally instead of loading everything first.
        stream=stream
    )

    response.raise_for_status()


    # -------------------------------------------------
    # STREAMING MODE
    # -------------------------------------------------
    if stream:

        text_parts = []

        # This will eventually contain Ollama's final chunk,
        # which contains the timing/token measurements.
        result = {}

        for line in response.iter_lines():

            if not line:
                continue

            # Convert bytes -> normal text if necessary
            if isinstance(line, bytes):
                line = line.decode("utf-8")

            # Each streamed line from Ollama is JSON
            chunk = json.loads(line)

            # Each chunk may contain another piece
            # of the generated response.
            text_parts.append(
                chunk.get("response", "")
            )

            # Keep updating result.
            # After the loop this will contain the final chunk,
            # where Ollama normally provides its metrics.
            result = chunk


        # Join all generated pieces into one final answer
        text = "".join(text_parts)


    # -------------------------------------------------
    # NON-STREAMING MODE
    # -------------------------------------------------
    else:

        result = response.json()

        text = result.get(
            "response",
            ""
        )


    # -------------------------------------------------
    # STOP WALL-CLOCK TIMER
    # -------------------------------------------------
    # This happens only after the COMPLETE answer has arrived.
    ollama_wall_time_ms = round(
        (time.perf_counter() - start) * 1000,
        2
    )


    # -------------------------------------------------
    # OLLAMA INTERNAL METRICS
    # -------------------------------------------------
    # Ollama itself provides these measurements.
    # They are reported in nanoseconds,
    # therefore ns_to_ms() converts them to milliseconds.

    prompt_eval_time_ms = ns_to_ms(
        result.get("prompt_eval_duration")
    )

    generation_time_ms = ns_to_ms(
        result.get("eval_duration")
    )

    load_time_ms = ns_to_ms(
        result.get("load_duration")
    )

    ollama_total_duration_ms = ns_to_ms(
        result.get("total_duration")
    )


    # -------------------------------------------------
    # MODEL COMPUTATION TIME
    # -------------------------------------------------
    # Approximate model computation:
    #
    # prompt processing
    #       +
    # output-token generation

    if (
        prompt_eval_time_ms is not None
        or generation_time_ms is not None
    ):

        model_compute_time_ms = round(
            (prompt_eval_time_ms or 0)
            +
            (generation_time_ms or 0),
            2
        )

    else:

        model_compute_time_ms = None


    # -------------------------------------------------
    # COLLECT ALL METRICS
    # -------------------------------------------------

    metrics = {
        "ollama_wall_time_ms":
            ollama_wall_time_ms,

        "model_compute_time_ms":
            model_compute_time_ms,

        "prompt_eval_time_ms":
            prompt_eval_time_ms,

        "generation_time_ms":
            generation_time_ms,

        "load_time_ms":
            load_time_ms,

        "ollama_total_duration_ms":
            ollama_total_duration_ms,

        "prompt_token_count":
            result.get("prompt_eval_count"),

        "generated_token_count":
            result.get("eval_count")
    }


    # -------------------------------------------------
    # RETURN
    # -------------------------------------------------

    if return_metrics:
        return text, metrics


    # Kept so older code calling:
    #
    # response, llm_time = call_llm(...)
    #
    # does not immediately break.
    return text, ollama_wall_time_ms