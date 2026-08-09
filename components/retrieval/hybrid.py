"""Hybrid dense + sparse retrieval facade."""

from __future__ import annotations


def hybrid_search(query, bm25, corpus, collection, embedder, top_k=5,
                  dense_weight=0.6, sparse_weight=0.4):
    from step5_vector_index import hybrid_search as _hybrid_search
    return _hybrid_search(
        query=query,
        bm25=bm25,
        corpus=corpus,
        collection=collection,
        embedder=embedder,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )


def retrieve_chunks(query, embedder, collection, bm25, corpus, top_k=5):
    from step6_query_gate import retrieve_chunks as _retrieve_chunks
    return _retrieve_chunks(
        query=query,
        embedder=embedder,
        collection=collection,
        bm25=bm25,
        corpus=corpus,
        top_k=top_k,
    )
