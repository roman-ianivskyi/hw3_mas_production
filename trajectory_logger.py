import json
import time
from datetime import datetime, timezone


class TrajectoryLogger:
    """Логування траєкторії виконання агента."""

    def __init__(self):
        self.steps = []
        self.start_time = time.monotonic()

    def log_step(self, step_num: int, node: str,
                 input_data: str, output_data: str,
                 tool_calls: list = None, tool_args: dict = None):
        self.steps.append({
            'step': step_num,
            'node': node,
            'input': input_data[:500],
            'output': output_data[:500],
            'tool_calls': tool_calls or [],
            'tool_args': tool_args or {},
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'elapsed_ms': int((time.monotonic() - self.start_time) * 1000),
        })

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'total_steps': len(self.steps),
                'total_time_ms': int((time.monotonic() - self.start_time) * 1000),
                'trajectory': self.steps,
            }, f, ensure_ascii=False, indent=2)
        print(f'Trajectory saved: {filepath} ({len(self.steps)} steps)')
