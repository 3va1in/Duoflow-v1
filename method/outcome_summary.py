"""汇总逐场景终局；中断不进入 SR，也不把部分格晋升为完整格。"""
from __future__ import annotations

import json
from pathlib import Path


def summarize_outcomes(rows):
    outcomes = [dict(row) for row in rows if row.get('event') == 'episode_outcome']
    seen = set()
    for row in outcomes:
        key = (row.get('task'), row.get('condition'), row.get('seed'), row.get('episode_index'))
        if key in seen:
            raise ValueError('duplicate scene outcome within one attempt')
        seen.add(key)
        completed = row.get('attempt_status') == 'completed'
        if completed:
            if type(row.get('task_success')) is not bool or row.get('included_in_sr') is not True:
                raise ValueError('completed outcome must contain a measured boolean result')
        elif (row.get('attempt_status') != 'aborted' or row.get('task_success') is not None
              or row.get('included_in_sr') is not False):
            raise ValueError('aborted outcome must remain null and outside SR')
    completed = [row for row in outcomes if row['included_in_sr']]
    aborted = [row for row in outcomes if not row['included_in_sr']]
    numerator = sum(row['task_success'] for row in completed)
    return {'completed_episodes': len(completed), 'successes': numerator,
            'normal_failures': len(completed) - numerator,
            'sr_numerator': numerator, 'sr_denominator': len(completed),
            'sr': numerator / len(completed) if completed else None,
            'aborted_episodes': len(aborted),
            'api_error_count': sum(row.get('error_category') == 'api_error' for row in aborted),
            'infrastructure_error_count': sum(row.get('error_category') in
                {'api_error', 'infrastructure_error'} for row in aborted),
            'policy_error_count': sum(row.get('error_category') == 'policy_error' for row in aborted),
            'outcomes': outcomes, 'complete_cell_acceptance_not_inferred': True}


def read_outcome_summary(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return summarize_outcomes(rows)
