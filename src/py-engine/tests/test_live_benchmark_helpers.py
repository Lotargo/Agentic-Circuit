from agentic_circuit.benchmarks.runner import (
    _balanced_sample,
    _locomo_sessions,
    _retrieval_metrics,
    _token_f1,
)


def test_balanced_sample_is_deterministic_and_spreads_categories():
    items = [
        {"id": index, "category": category}
        for category in ("a", "b", "c")
        for index in range(4)
    ]
    first = _balanced_sample(items, lambda item: item["category"], 6, 42)
    second = _balanced_sample(items, lambda item: item["category"], 6, 42)

    assert first == second
    assert {item["category"] for item in first} == {"a", "b", "c"}


def test_token_f1_and_retrieval_metrics():
    assert _token_f1("Neon database", "The selected database is Neon") > 0.5
    recall, reciprocal_rank = _retrieval_metrics(
        ["session-2", "session-1", "session-3"],
        {"session-1", "session-3"},
    )
    assert recall == 1.0
    assert reciprocal_rank == 0.5


def test_locomo_parser_maps_dialog_ids_to_sessions():
    sample = {
        "conversation": {
            "session_2_date_time": "2024-02-02",
            "session_2": [
                {"speaker": "B", "dia_id": "D2:1", "text": "second"}
            ],
            "session_1_date_time": "2024-01-01",
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "first"},
                {
                    "speaker": "B",
                    "dia_id": "D1:2",
                    "text": "image",
                    "blip_caption": "a sunrise",
                },
            ],
            "speaker_a": "A",
            "speaker_b": "B",
        }
    }

    sessions = _locomo_sessions(sample)

    assert [session[0] for session in sessions] == ["D1", "D2"]
    assert sessions[0][3] == {"D1:1", "D1:2"}
    assert "image_caption: a sunrise" in sessions[0][1]
