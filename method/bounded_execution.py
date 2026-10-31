"""限制语言子目标的连续执行；不判断任务成功、不处理 API 异常。"""


class BoundedExecution:
    def __init__(self, max_skill_chunks=8, max_unknown_chunks=3):
        if max_skill_chunks < 1 or max_unknown_chunks < 1:
            raise ValueError('执行上限必须为正数')
        self.max_skill_chunks = max_skill_chunks
        self.max_unknown_chunks = max_unknown_chunks
        self.last_skill_id = None
        self.repeated_chunks = 0
        self.unknown_chunks = 0
        self.whole_task = False
        self.reason = None

    def observe(self, decision):
        # 状态只描述上一段实际动作；换目标不会把 unknown 当作完成。
        if decision.get('progress_status') == 'unknown' and self.last_skill_id:
            self.unknown_chunks += 1
        elif decision.get('progress_status') in {'satisfied', 'unsatisfied'}:
            self.unknown_chunks = 0
        if self.unknown_chunks >= self.max_unknown_chunks:
            self.delegate('persistent_unknown')

    def delegate(self, reason):
        self.whole_task = True
        self.reason = self.reason or reason

    def note_issued(self, skill_id):
        if skill_id == self.last_skill_id:
            self.repeated_chunks += 1
        else:
            self.last_skill_id = skill_id
            self.repeated_chunks = 1
        if not self.whole_task and self.repeated_chunks >= self.max_skill_chunks:
            self.delegate('repeated_skill_budget')

    def export(self):
        return {'execution_mode': 'whole_task' if self.whole_task else 'skill',
                'reason': self.reason, 'last_skill_id': self.last_skill_id,
                'repeated_chunks': self.repeated_chunks, 'unknown_chunks': self.unknown_chunks,
                'max_skill_chunks': self.max_skill_chunks,
                'max_unknown_chunks': self.max_unknown_chunks,
                'verified_complete': False, 'environment_step_budget_delta': 0}
