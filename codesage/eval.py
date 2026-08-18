"""A small, repeatable check that CodeSage's answers are actually grounded.

Two proxies, both cheap and deterministic:
- retrieval hit-rate: did we retrieve a chunk from the file we expected?
- answer score: fraction of expected keywords present in the final answer.

score_retrieval takes an `index` with a `.search(query_vector, k) -> list[Chunk]`
method — RetrievalIndex satisfies that shape today. A second retrieval
strategy could be swapped in here without changing this file (Strategy
pattern), if one is ever built.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalCase:
    question: str
    expected_file_substring: str
    expected_answer_keywords: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text())
    return [EvalCase(**d) for d in data]


def score_retrieval(index, llm, case: EvalCase) -> bool:
    query_vector = llm.embed(case.question)
    results = index.search(query_vector, k=5)
    return any(case.expected_file_substring in c.file_path for c in results)


def score_answer(answer: str, case: EvalCase) -> float:
    if not case.expected_answer_keywords:
        return 0.0
    hits = sum(1 for kw in case.expected_answer_keywords if kw.lower() in answer.lower())
    return hits / len(case.expected_answer_keywords)


def run_eval(agent_factory, index, llm, cases: list[EvalCase]) -> dict:
    """agent_factory is called once per case, not once total.

    Eval cases are independent questions, not a conversation — sharing one
    Agent's memory across them mixes unrelated context and, once memory
    fills up, can even corrupt the turn sequence (a sliding-window eviction
    can split a function-call/function-response pair mid-tool-use, which
    the API rejects outright). A fresh Agent per case sidesteps both.
    """
    retrieval_hits = 0
    answer_scores = []
    for case in cases:
        if score_retrieval(index, llm, case):
            retrieval_hits += 1
        agent = agent_factory()
        answer = agent.ask(case.question)
        answer_scores.append(score_answer(answer, case))

    return {
        "retrieval_hit_rate": retrieval_hits / len(cases),
        "avg_answer_score": sum(answer_scores) / len(answer_scores),
    }
