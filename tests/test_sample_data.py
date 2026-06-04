from demo.sample_data import SAMPLE_DOCUMENTS, SAMPLE_QUESTIONS, get_sample_documents, get_sample_questions


def test_sample_documents_have_unique_ids_and_content():
    ids = [doc["id"] for doc in SAMPLE_DOCUMENTS]
    assert len(ids) == len(set(ids))
    assert len(SAMPLE_DOCUMENTS) >= 5

    for doc in SAMPLE_DOCUMENTS:
        assert doc["id"].strip()
        assert doc["source"].strip()
        assert doc["content"].strip()
        assert len(doc["content"]) > 80


def test_sample_questions_are_non_empty_strings():
    assert len(SAMPLE_QUESTIONS) >= 5
    assert all(isinstance(question, str) for question in SAMPLE_QUESTIONS)
    assert all(question.strip() for question in SAMPLE_QUESTIONS)


def test_getters_return_copies():
    docs = get_sample_documents()
    questions = get_sample_questions()

    docs.append({"id": "mutated", "source": "mutated", "content": "mutated"})
    questions.append("mutated?")

    assert len(get_sample_documents()) == len(SAMPLE_DOCUMENTS)
    assert len(get_sample_questions()) == len(SAMPLE_QUESTIONS)
