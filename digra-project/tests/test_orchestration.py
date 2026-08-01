import pandas as pd

from src.agents.orchestration import (
    load_pools_for_dataset_model,
    parse_setup,
    run_debates_for_dataset_model,
    setup_label,
)
from src.llm.fake_client import FakeLLMClient
from src.utils.checkpoint import RunRegistry
from src.utils.io import append_jsonl, read_jsonl


def _tiny_df():
    return pd.DataFrame([
        {
            "question_id": "nq_0000", "question": "Who won?", "gold_answer": "Yale",
            "gold_answer_alternatives": [],
        },
        {
            "question_id": "nq_0001", "question": "Who lost?", "gold_answer": "Duke",
            "gold_answer_alternatives": [],
        },
    ])


def _write_pools_file(path):
    append_jsonl(path, {
        "question_id": "nq_0000",
        "correct_texts": ["correct A", "correct B"],
        "incorrect_texts": ["wrong A", "wrong B"],
    })
    append_jsonl(path, {
        "question_id": "nq_0001",
        "correct_texts": ["correct C", "correct D"],
        "incorrect_texts": ["wrong C", "wrong D"],
    })


# ---------------------------------------------------------------------------
# parse_setup / setup_label
# ---------------------------------------------------------------------------

def test_parse_setup_standard():
    assert parse_setup("standard") == "standard"


def test_parse_setup_list_to_tuple():
    assert parse_setup([2, 1]) == (2, 1)


def test_setup_label_standard():
    assert setup_label("standard") == "standard"


def test_setup_label_tuple():
    assert setup_label((2, 1)) == "2,1"


# ---------------------------------------------------------------------------
# load_pools_for_dataset_model
# ---------------------------------------------------------------------------

def test_load_pools(tmp_path):
    pools_path = tmp_path / "pools.jsonl"
    _write_pools_file(pools_path)
    pools = load_pools_for_dataset_model(pools_path)
    assert set(pools.keys()) == {"nq_0000", "nq_0001"}
    assert pools["nq_0000"]["correct_texts"] == ["correct A", "correct B"]


# ---------------------------------------------------------------------------
# run_debates_for_dataset_model
# ---------------------------------------------------------------------------

def test_runs_all_question_setup_seed_combinations(tmp_path):
    pools_path = tmp_path / "pools.jsonl"
    _write_pools_file(pools_path)
    pools = load_pools_for_dataset_model(pools_path)

    llm = FakeLLMClient(response_fn=lambda p: "revised")
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "debates.jsonl"

    counts = run_debates_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        pools=pools, setups=[(1, 1)], seeds=[0, 1],
        n_agents=2, n_rounds=2, registry=registry, out_path=out_path,
    )

    # 1 setup x 2 seeds x 2 questions = 4 debates
    assert counts["built"] == 4
    assert counts["skipped"] == 0
    assert counts["errors"] == 0
    records = list(read_jsonl(out_path))
    assert len(records) == 4


def test_resume_skips_already_completed(tmp_path):
    pools_path = tmp_path / "pools.jsonl"
    _write_pools_file(pools_path)
    pools = load_pools_for_dataset_model(pools_path)
    out_path = tmp_path / "debates.jsonl"
    registry_path = tmp_path / "registry.json"

    llm1 = FakeLLMClient(response_fn=lambda p: "revised")
    registry1 = RunRegistry(registry_path)
    run_debates_for_dataset_model(
        llm=llm1, dataset_key="nq", model_name="llama", df=_tiny_df(),
        pools=pools, setups=[(1, 1)], seeds=[0],
        n_agents=2, n_rounds=2, registry=registry1, out_path=out_path,
    )

    # simulate a fresh process after a "Kaggle disconnect" — same registry file
    llm2 = FakeLLMClient(response_fn=lambda p: "revised")
    registry2 = RunRegistry(registry_path)
    counts = run_debates_for_dataset_model(
        llm=llm2, dataset_key="nq", model_name="llama", df=_tiny_df(),
        pools=pools, setups=[(1, 1)], seeds=[0],
        n_agents=2, n_rounds=2, registry=registry2, out_path=out_path,
    )

    assert counts["built"] == 0
    assert counts["skipped"] == 2  # both questions already done
    # out_path still only has the original 2 records, not duplicated
    assert len(list(read_jsonl(out_path))) == 2


def test_setup_agent_count_mismatch_is_skipped_not_errored(tmp_path):
    pools_path = tmp_path / "pools.jsonl"
    _write_pools_file(pools_path)
    pools = load_pools_for_dataset_model(pools_path)

    llm = FakeLLMClient(response_fn=lambda p: "revised")
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "debates.jsonl"

    # setup (2,1) sums to 3, but n_agents=5 -> should be skipped, not raise
    counts = run_debates_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        pools=pools, setups=[(2, 1)], seeds=[0],
        n_agents=5, n_rounds=2, registry=registry, out_path=out_path,
    )
    assert counts["built"] == 0
    assert counts["errors"] == 0


def test_standard_setup_works_for_any_agent_count(tmp_path):
    pools_path = tmp_path / "pools.jsonl"
    _write_pools_file(pools_path)
    pools = load_pools_for_dataset_model(pools_path)

    llm = FakeLLMClient(response_fn=lambda p: "revised")
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "debates.jsonl"

    counts = run_debates_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        pools=pools, setups=["standard"], seeds=[0],
        n_agents=5, n_rounds=1, registry=registry, out_path=out_path,
    )
    assert counts["built"] == 2  # both questions, standard setup, any n_agents


def test_missing_pool_for_question_is_logged_and_skipped(tmp_path):
    # pools file only covers nq_0000, not nq_0001
    pools_path = tmp_path / "pools.jsonl"
    append_jsonl(pools_path, {
        "question_id": "nq_0000", "correct_texts": ["c"], "incorrect_texts": ["w"],
    })
    pools = load_pools_for_dataset_model(pools_path)

    llm = FakeLLMClient(response_fn=lambda p: "revised")
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "debates.jsonl"

    counts = run_debates_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        pools=pools, setups=[(1, 0)], seeds=[0],
        n_agents=1, n_rounds=1, registry=registry, out_path=out_path,
    )
    assert counts["built"] == 1     # nq_0000 succeeds
    assert counts["errors"] == 1    # nq_0001 has no pool entry, logged + skipped
