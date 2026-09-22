"""Weather enrichment facade."""

from __future__ import annotations


def is_weather_query(query):
    from step8_weather_rag import is_weather_query as _is_weather_query
    return _is_weather_query(query)


def enrich_with_weather(query, base_context):
    from step8_weather_rag import enrich_with_weather as _enrich
    return _enrich(query, base_context)
