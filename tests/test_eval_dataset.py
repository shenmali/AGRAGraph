import json
from pathlib import Path

from demo.sample_data import SAMPLE_DOCUMENTS


DATASET_PATH = Path("evals/datasets/sample_qa.json")


def test_sample_qa_dataset_schema():
    data = json.loads(DATASET_PATH.read_text())

    assert data["name"] == "agra_sample_qa"
    assert isinstance(data["questions"], list)
    assert len(data["questions"]) >= 5

    for item in data["questions"]:
        assert item["id"].strip()
        assert item["query"].strip()
        assert item["expected_answer"].strip()
        assert isinstance(item["expected_source_ids"], list)
        assert item["expected_source_ids"]


def test_sample_qa_expected_sources_exist_in_sample_corpus():
    data = json.loads(DATASET_PATH.read_text())
    sample_ids = {doc["id"] for doc in SAMPLE_DOCUMENTS}

    for item in data["questions"]:
        assert set(item["expected_source_ids"]).issubset(sample_ids)
