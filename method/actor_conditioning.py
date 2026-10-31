"""保留原任务与当前技能，避开官方 tokenizer 的保留分隔符。"""


def compose_instruction(instruction, execution_prompt):
    instruction = ' '.join(instruction.split()).strip()
    execution_prompt = ' '.join(execution_prompt.split()).strip()
    if not instruction or not execution_prompt:
        raise ValueError('任务和技能文本均不能为空')
    if ' subtask:' in (' ' + instruction + ' ' + execution_prompt).lower():
        raise ValueError('任务文本含官方 tokenizer 保留分隔符 Subtask:')
    return instruction + ' Next action: ' + execution_prompt


def normalize_text(text):
    return ' '.join(text.lower().split())


def verify_actor_text(decoded, instruction, execution_prompt, execution_mode='skill'):
    text = normalize_text(decoded)
    if execution_mode == 'whole_task':
        return {'task_preserved': normalize_text(instruction) in text,
                'no_subgoal_override': 'next action:' not in text}
    if execution_mode != 'skill':
        raise ValueError('未知动作文本模式')
    return {
        'task_preserved': normalize_text(instruction) in text,
        'skill_preserved': normalize_text(execution_prompt) in text,
        'next_action_present': 'next action:' in text,
    }
