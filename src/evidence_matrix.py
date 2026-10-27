"""Candidate treatment/outcome passage matrix; matches are not proof."""
from __future__ import annotations

import re

from .intervention_gate import requested_interventions, source_intervention_matches
from .evidence_positioning import requested_outcomes, OUTCOME_ALIASES


def build_evidence_matrix(question, sources, chars_per_source=3500):
    treatments = requested_interventions(question)
    outcomes = requested_outcomes(question)
    rows = []
    for index, source in enumerate(sources, 1):
        text = str(source.get('raw_content') or source.get('content') or '')[:chars_per_source]
        matches = source_intervention_matches({'content': text}, treatments)
        for outcome in outcomes:
            aliases = OUTCOME_ALIASES[outcome]
            passages = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                        if any(a.casefold() in s.casefold() for a in aliases)]
            for passage in passages[:2]:
                rows.append({'source_id': index, 'title': source.get('title', ''),
                             'url': source.get('url', ''), 'treatments_mentioned': matches,
                             'outcome': outcome, 'passage': passage[:900],
                             'status': 'candidate signal; clinical support not verified'})
    return {'requested_treatments': treatments, 'requested_outcomes': outcomes,
            'candidate_passages': rows,
            'missing_outcome_signals': [o for o in outcomes if not any(r['outcome'] == o for r in rows)]}


def format_evidence_matrix(matrix):
    import json
    return ('CANDIDATE EVIDENCE MATRIX (keyword signals only, not verified evidence; '
            'a treatment mention and outcome mention do not establish a result or comparison):\n'
            + json.dumps(matrix, ensure_ascii=False))
