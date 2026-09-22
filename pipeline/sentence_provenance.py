import re
import os
from groq import Groq
import json
from config import GROQ_GATE_MODEL


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
    # Filter out numbered list artifacts like "1.", "2.", "3."
    sentences = [s for s in out if len(s.strip()) > 10]
    return sentences


def _safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def match_sentence_to_source(sentence: str, scored_chunks: list, client=None) -> dict:
    """
    Evaluates provenance for a single sentence against retrieved chunks.

    STRICT PROVENANCE RULES:
    1. A sentence is ONLY labeled 'ENTAILED' if an actual entailment check verifies strict factual entailment.
    2. Heuristic lexical overlap alone is strictly classified as 'CANDIDATE_MATCH' (entailment unverified).
    3. If overlap is below 0.25 (or no match exists), it is labeled 'UNSUPPORTED / UNVERIFIED'.
    4. Never invent confidence percentages (e.g. no arbitrary +0.3 or false 1.00 on lexical overlap).
    """
    # 1. Attempt LLM Entailment Check if client is provided
    if client is not None:
        try:
            previews = []
            for c in scored_chunks[:5]:
                src = c.get('source_file') or c.get('source') or c.get('temporal_source_match') or 'unknown'
                chunk_text = (c.get('text') or '')[:250].replace('\n', ' ')
                previews.append(f"Source: {src}\nText: {chunk_text}...")

            prompt = (
                "You are a strict natural language inference (NLI) entailment auditor for agricultural advice.\n"
                "Given a sentence and candidate source texts, determine whether any source text strictly ENTAILS (proves)\n"
                "the statement. If the statement is an extrapolation, general knowledge, or not in the text, mark it not entailed.\n\n"
                f"Sentence: \"{sentence}\"\n\n"
                "Source chunks:\n"
                + "\n\n".join(previews)
                + "\n\nReply with ONLY a JSON object:\n"
                "{\n"
                "  \"entailed\": true/false,\n"
                "  \"matched_source\": \"filename.pdf\" or \"UNSUPPORTED\",\n"
                "  \"confidence\": 0.0 to 1.0,\n"
                "  \"reason\": \"concise assessment\"\n"
                "}\n"
            )

            resp = client.chat.completions.create(
                model=GROQ_GATE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict natural language inference entailment auditor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )

            raw = resp.choices[0].message.content
            data = raw if isinstance(raw, dict) else _safe_parse_json(raw)
            if data and isinstance(data, dict):
                is_entailed = bool(data.get("entailed", False))
                matched = data.get("matched_source", "UNSUPPORTED")
                reason = data.get("reason", "")
                conf = float(data.get("confidence", 0.8 if is_entailed else 0.0))

                # Match chunk metadata
                matched_chunk = None
                for c in scored_chunks:
                    name = c.get('source_file') or c.get('source') or c.get('temporal_source_match')
                    if name == matched:
                        matched_chunk = c
                        break

                temporal = float(matched_chunk.get('temporal_score', 0)) if matched_chunk else 0.0
                freshness = matched_chunk.get('freshness_label', 'UNKNOWN') if matched_chunk else 'UNKNOWN'

                if is_entailed and matched != "UNSUPPORTED":
                    return {
                        'sentence': sentence,
                        'matched_source': matched,
                        'match_status': 'ENTAILED',
                        'entailment_verified': True,
                        'confidence': round(conf, 3),
                        'lexical_overlap': None,
                        'reason': reason or 'Strict entailment verified by LLM',
                        'temporal_score': round(temporal, 4),
                        'freshness_label': freshness,
                        'combined_score': round((conf + temporal) / 2.0, 4)
                    }
                else:
                    return {
                        'sentence': sentence,
                        'matched_source': 'UNSUPPORTED / UNVERIFIED',
                        'match_status': 'UNSUPPORTED',
                        'entailment_verified': False,
                        'confidence': 0.0,
                        'lexical_overlap': None,
                        'reason': reason or 'Statement not entailed by retrieved context',
                        'temporal_score': 0.0,
                        'freshness_label': 'UNKNOWN',
                        'combined_score': 0.0
                    }
        except Exception:
            pass  # Fall back to candidate lexical matching

    # 2. Candidate Matching via Lexical Overlap (entailment UNVERIFIED)
    best_chunk = None
    best_score = 0.0
    stop_words = {"the", "and", "for", "that", "this", "with", "from", "are", "can", "will", "has", "have", "you", "your"}
    sentence_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()) if w not in stop_words]

    if sentence_words and scored_chunks:
        for c in scored_chunks:
            text = (c.get('text') or '').lower()
            overlap_count = sum(1 for w in sentence_words if w in text)
            score = overlap_count / max(1, len(sentence_words))
            if score > best_score:
                best_score = score
                best_chunk = c

    # Overlap must be at least 0.25 to qualify even as a CANDIDATE match
    if best_chunk and best_score >= 0.25:
        matched = best_chunk.get('source_file') or best_chunk.get('source') or 'unknown'
        temporal = float(best_chunk.get('temporal_score', 0))
        freshness = best_chunk.get('freshness_label', 'UNKNOWN')
        lex_score = round(best_score, 3)

        return {
            'sentence': sentence,
            'matched_source': matched,
            'match_status': 'CANDIDATE_MATCH',
            'entailment_verified': False,
            'confidence': None,  # No invented confidence; reported strictly as lexical overlap
            'lexical_overlap': lex_score,
            'reason': f"Candidate source match ({lex_score*100:.1f}% lexical word overlap; entailment unverified)",
            'temporal_score': round(temporal, 4),
            'freshness_label': freshness,
            'combined_score': round((lex_score + temporal) / 2.0, 4)
        }

    # Strict invariant: lexical overlap or source freshness alone can NEVER establish ENTAILED
    res = {
        'sentence': sentence,
        'matched_source': 'UNSUPPORTED / UNVERIFIED',
        'match_status': 'UNSUPPORTED',
        'entailment_verified': False,
        'confidence': 0.0,
        'lexical_overlap': round(best_score, 3) if sentence_words else 0.0,
        'reason': 'Insufficient textual overlap in retrieved context',
        'temporal_score': 0.0,
        'freshness_label': 'UNKNOWN',
        'combined_score': 0.0
    }
    return res


def is_entailment_verified(match_result: dict) -> bool:
    """
    Returns True ONLY if factual entailment was verified by an actual NLI method.
    Lexical overlap, high similarity, or source freshness alone can NEVER establish entailment.
    """
    if not isinstance(match_result, dict):
        return False
    return bool(match_result.get("entailment_verified") is True and match_result.get("match_status") == "ENTAILED")


def build_provenance_map(answer_text: str, scored_chunks: list, client=None) -> list:
    sentences = split_into_sentences(answer_text)
    out = []
    for s in sentences:
        prov = match_sentence_to_source(s, scored_chunks, client=client)
        out.append(prov)
    return out


def get_provenance_summary(provenance_map: list) -> dict:
    total = len(provenance_map)
    if total == 0:
        return {
            'total_sentences': 0,
            'entailed_count': 0,
            'candidate_match_count': 0,
            'unsupported_count': 0,
            'avg_lexical_overlap': 0.0,
            'avg_temporal_score': 0.0,
            'avg_combined_score': 0.0,
            'sources_used': []
        }

    entailed = sum(1 for p in provenance_map if p.get('match_status') == 'ENTAILED')
    candidate = sum(1 for p in provenance_map if p.get('match_status') == 'CANDIDATE_MATCH')
    unsupported = sum(1 for p in provenance_map if p.get('match_status') in ('UNSUPPORTED', 'UNKNOWN'))

    overlaps = [p.get('lexical_overlap') for p in provenance_map if p.get('lexical_overlap') is not None]
    avg_overlap = (sum(overlaps) / len(overlaps)) if overlaps else 0.0

    avg_temp = sum(p.get('temporal_score', 0) for p in provenance_map) / total
    avg_comb = sum(p.get('combined_score', 0) for p in provenance_map) / total

    sources = list(set([
        p["matched_source"] for p in provenance_map 
        if p.get("matched_source") and p["matched_source"] not in ("None", "UNKNOWN", "INFERRED", "UNSUPPORTED / UNVERIFIED")
    ]))

    return {
        'total_sentences': total,
        'entailed_count': entailed,
        'candidate_match_count': candidate,
        'unsupported_count': unsupported,
        'avg_lexical_overlap': round(avg_overlap, 3),
        'avg_temporal_score': round(avg_temp, 4),
        'avg_combined_score': round(avg_comb, 4),
        'sources_used': sources
    }


def print_provenance_report(provenance_map: list):
    print('\n' + '╔' + '═'*62 + '╗')
    print('║           SENTENCE-LEVEL PROVENANCE REPORT                   ║')
    print('╠' + '═'*62 + '╣')
    for p in provenance_map:
        s = p.get('sentence', '')
        status = p.get('match_status', 'UNKNOWN')
        is_verified = p.get('entailment_verified', False)

        print()
        print(f"📝 \"{s[:80]}{'...' if len(s)>80 else ''}\"")
        print(f"   └─ Source       : {p.get('matched_source')}")
        if is_verified:
            print(f"   └─ Match Status : {status} (Entailment Verified: True, Conf: {p.get('confidence', 0):.2f})")
        else:
            overlap = p.get('lexical_overlap')
            overlap_str = f"{overlap:.2f}" if overlap is not None else "N/A"
            print(f"   └─ Match Status : {status} (Lexical Overlap: {overlap_str}, Entailment: UNVERIFIED)")

        temp_sc = p.get('temporal_score')
        temp_sc_str = f"{temp_sc:.4f}" if temp_sc is not None else "0.0000"
        print(f"   └─ Freshness    : {p.get('freshness_label', 'UNKNOWN')} (temporal score: {temp_sc_str})")
        comb = p.get('combined_score')
        comb_str = f"{comb:.4f}" if comb is not None else "0.0000"
        print(f"   └─ Combined     : {comb_str}")
        print(f"   └─ Assessment   : {p.get('reason', 'N/A')}")

    summary = get_provenance_summary(provenance_map)
    print('\n' + '══════════════════════════════════════')
    print('📊 PROVENANCE SUMMARY')
    print(f"   Total sentences analysed : {summary['total_sentences']}")
    print(f"   Entailed (strict proof)  : {summary['entailed_count']}")
    print(f"   Candidate matches (lex)  : {summary['candidate_match_count']}")
    print(f"   Unsupported / unverified : {summary['unsupported_count']}")
    print(f"   Avg lexical overlap      : {summary['avg_lexical_overlap']:.3f}")
    print(f"   Avg temporal score       : {summary['avg_temporal_score']:.4f}")
    print('══════════════════════════════════════')
    if summary['entailed_count'] == 0:
        print("\n⚠️  VALIDATION NOTICE: 0 sentences strictly entailed by retrieved context.")
        print("   Statements represent candidate matches or unverified extrapolations.")
        print("   The answer as a whole is NOT factually proven or verified.")
