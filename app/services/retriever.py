from sentence_transformers import CrossEncoder
from app.database import get_vector_store

_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# How many candidates MMR considers before picking the final k
_FETCH_K = 20

# Final chunks returned to Claude
_K = 4

# 0 = maximise diversity, 1 = maximise similarity. 0.6 balances both.
_LAMBDA = 0.6

# Maximum cosine distance — scores from MMR are distances (lower = more similar).
# Chunks above this threshold are too far from the query and are dropped.
# 0.70 ≈ cosine similarity of 0.30, which filters genuinely unrelated content.
_SCORE_THRESHOLD = 0.70


def retrieve(question: str, filter_docs: list[str] | None = None) -> list:
    vs = get_vector_store()

    search_filter = None
    if filter_docs:
        search_filter = {"document_name": {"$in": filter_docs}}

    # MMR fetches _FETCH_K candidates by similarity, then selects _K that
    # are both relevant AND diverse — prevents one large document from
    # filling all slots when multiple documents are ingested.
    docs_and_scores = vs.max_marginal_relevance_search_with_score(
        question, k=_K, fetch_k=_FETCH_K, lambda_mult=_LAMBDA,
        filter=search_filter,
    )

    results = []
    for doc, score in docs_and_scores:
        if score > _SCORE_THRESHOLD:
            continue
        results.append({
            "text": doc.page_content,
            "page_num": doc.metadata["page_num"],
            "document_name": doc.metadata["document_name"],
            "score": round(score, 3),
        })

    # Cross-encoder reranks the MMR candidates for precision
    if len(results) > 1:
        scores = _get_reranker().predict([(question, r["text"]) for r in results])
        results = [r for r, _ in sorted(zip(results, scores), key=lambda x: x[1], reverse=True)]

    return results
