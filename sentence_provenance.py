import re
import os
from groq import Groq
import json


def split_into_sentences(text: str) -> list:
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    out = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if s.startswith('#') or s.startswith('**'):
            continue
        if '|' in s:
            continue
        out.append(s)
    return out


def _safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def match_sentence_to_source(sentence: str, scored_chunks: list, client) -> dict:
    try:
        # Build source preview
        previews = []
        for c in scored_chunks:
            src = c.get('source_file') or c.get('source') or c.get('temporal_source_match') or 'unknown'
            chunk_text = (c.get('text') or '')[:200].replace('\n', ' ')
            previews.append(f"Source: {src}\nText: {chunk_text}...")

        prompt = (
            "You are a provenance matching assistant.\n"
            "Given a sentence from an agricultural advisory answer and a list of source chunks,\n"
            "identify which source chunk most likely supports this sentence.\n\n"
            f"Sentence: \"{sentence}\"\n\n"
            "Source chunks:\n"
            + "\n\n".join(previews)
            + "\n\nReply with ONLY a JSON object:\n{\n  \"matched_source\": \"filename.pdf\",\n  \"confidence\": 0.0,\n  \"reason\": \"one line explanation\"\n}\n"
        )

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a provenance matching assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        data = raw if isinstance(raw, dict) else _safe_parse_json(raw)
        if not data:
            raise ValueError("empty or invalid JSON from model")

        matched = data.get('matched_source', 'INFERRED')
        confidence = float(data.get('confidence', 0.3))
        reason = data.get('reason', '')

        # Find chunk metadata
        matched_chunk = None
        for c in scored_chunks:
            name = c.get('source_file') or c.get('source') or c.get('temporal_source_match')
            if name == matched:
                matched_chunk = c
                break

        if matched_chunk:
            temporal = float(matched_chunk.get('temporal_score', 0))
            freshness = matched_chunk.get('freshness_label', 'UNKNOWN')
        else:
            temporal = 0.0
            freshness = 'INFERRED' if matched == 'INFERRED' else 'UNKNOWN'

        combined = (confidence + temporal) / 2.0

        return {
            'sentence': sentence,
            'matched_source': matched,
            'confidence': confidence,
            'reason': reason,
            'temporal_score': round(temporal, 4),
            'freshness_label': freshness,
            'combined_score': round(combined, 4)
        }

    except Exception:
        # Fallback heuristic: simple substring overlap
        try:
            best = None
            best_score = 0.0
            for c in scored_chunks:
                text = (c.get('text') or '').lower()
                words = [w for w in re.findall(r"\w+", sentence.lower())]
                if not words:
                    continue
                overlap = sum(1 for w in words if w in text)
                score = overlap / max(1, len(words))
                if score > best_score:
                    best_score = score
                    best = c

            if best and best_score > 0.1:
                matched = best.get('source_file') or best.get('source') or 'unknown'
                temporal = float(best.get('temporal_score', 0))
                freshness = best.get('freshness_label', 'UNKNOWN')
                confidence = round(min(1.0, best_score + 0.3), 3)
                combined = round((confidence + temporal) / 2.0, 4)
                return {
                    'sentence': sentence,
                    'matched_source': matched,
                    'confidence': confidence,
                    'reason': 'heuristic overlap',
                    'temporal_score': round(temporal, 4),
                    'freshness_label': freshness,
                    'combined_score': combined
                }
        except Exception:
            pass

        return {
            'sentence': sentence,
            'matched_source': 'UNKNOWN',
            'confidence': 0.0,
            'temporal_score': 0.0,
            'freshness_label': 'UNKNOWN',
            'combined_score': 0.0,
            'reason': 'matching failed'
        }


def build_provenance_map(answer_text: str, scored_chunks: list, client) -> list:
    sentences = split_into_sentences(answer_text)
    out = []
    for s in sentences:
        prov = match_sentence_to_source(s, scored_chunks, client)
        out.append(prov)
    return out


def get_provenance_summary(provenance_map: list) -> dict:
    total = len(provenance_map)
    if total == 0:
        return {
            'total_sentences': 0,
            'avg_confidence': 0.0,
            'avg_temporal_score': 0.0,
            'avg_combined_score': 0.0,
            'high_confidence_count': 0,
            'inferred_count': 0,
            'sources_used': []
        }

    avg_conf = sum(p.get('confidence', 0) for p in provenance_map) / total
    avg_temp = sum(p.get('temporal_score', 0) for p in provenance_map) / total
    avg_comb = sum(p.get('combined_score', 0) for p in provenance_map) / total
    high = sum(1 for p in provenance_map if p.get('confidence', 0) > 0.7)
    inferred = sum(1 for p in provenance_map if p.get('matched_source') in ('INFERRED', 'UNKNOWN'))
    sources = list({p.get('matched_source') for p in provenance_map if p.get('matched_source') not in ('INFERRED', 'UNKNOWN')})

    return {
        'total_sentences': total,
        'avg_confidence': round(avg_conf, 3),
        'avg_temporal_score': round(avg_temp, 4),
        'avg_combined_score': round(avg_comb, 4),
        'high_confidence_count': high,
        'inferred_count': inferred,
        'sources_used': sources
    }


def print_provenance_report(provenance_map: list):
    print('\n' + '╔' + '═'*62 + '╗')
    print('║           SENTENCE-LEVEL PROVENANCE REPORT                   ║')
    print('╠' + '═'*62 + '╣')
    for p in provenance_map:
        s = p.get('sentence', '')
        print()
        print(f"📝 \"{s[:80]}{'...' if len(s)>80 else ''}\"")
        print(f"   └─ Source     : {p.get('matched_source')}")
        print(f"   └─ Confidence : {p.get('confidence'):.2f}")
        print(f"   └─ Freshness  : {p.get('freshness_label')} (temporal score: {p.get('temporal_score'):.4f})")
        print(f"   └─ Combined   : {p.get('combined_score'):.4f}")
        print(f"   └─ Reason     : {p.get('reason')}")

    summary = get_provenance_summary(provenance_map)
    print('\n' + '══════════════════════════════════════')
    print('📊 PROVENANCE SUMMARY')
    print(f"   Total sentences analysed : {summary['total_sentences']}")
    print(f"   Avg confidence           : {summary['avg_confidence']:.2f}")
    print(f"   Avg temporal score       : {summary['avg_temporal_score']:.4f}")
    print(f"   Avg combined score       : {summary['avg_combined_score']:.4f}")
    print(f"   High confidence (>0.7)   : {summary['high_confidence_count']} sentences")
    print(f"   Inferred (no source)     : {summary['inferred_count']} sentences")
    print('══════════════════════════════════════')
