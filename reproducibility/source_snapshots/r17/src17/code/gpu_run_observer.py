"""Record external GPU interference without touching another process."""
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def gpu_command(arguments):
    """Scope driver queries to the same full UUID as CUDA, or retain legacy scope."""
    import re
    target = os.environ.get('TPDS_GPU_UUID')
    if target is None:
        return ['nvidia-smi', *arguments]
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', target):
        raise ValueError('TPDS_GPU_UUID must be one full GPU UUID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != target:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal TPDS_GPU_UUID')
    return ['nvidia-smi', '-i', target, *arguments]


class Observer:
    def __init__(self, directory):
        self.root_pid = os.getpid()
        self.directory = Path(directory)
        self.stop_event = threading.Event()
        self.foreign, self.errors = [], []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def belongs_to_run(self, pid):
        seen = set()
        while pid > 1 and pid not in seen:
            if pid == self.root_pid:
                return True
            seen.add(pid)
            try:
                # Process name in stat can contain spaces/parentheses.
                fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                pid = int(fields[1])
            except FileNotFoundError:
                return None  # disappeared between driver query and /proc
        return False

    def run(self):
        with (self.directory / 'gpu_occupancy.jsonl').open('x', buffering=1) as stream:
            while not self.stop_event.is_set():
                record = {'monotonic_s': time.monotonic(), 'selected_gpu_uuid': os.environ.get('TPDS_GPU_UUID')}
                try:
                    result = subprocess.run(gpu_command(['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits']), capture_output=True,
                                            text=True, timeout=10)
                    if result.returncode:
                        raise RuntimeError(result.stderr)
                    telemetry = subprocess.run(gpu_command(['--query-gpu=uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used', '--format=csv,noheader,nounits']), capture_output=True, text=True, timeout=10)
                    if telemetry.returncode:
                        raise RuntimeError(telemetry.stderr)
                    record['gpu_telemetry'] = [dict(zip(
                        ['uuid', 'temperature_c', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz',
                         'utilization_percent', 'memory_used_mib'], row))
                        for row in csv.reader(telemetry.stdout.strip().splitlines(), skipinitialspace=True)]
                    target = os.environ.get('TPDS_GPU_UUID')
                    if target and [row['uuid'] for row in record['gpu_telemetry']] != [target]:
                        raise RuntimeError('selected GPU telemetry missing or mismatched')
                    record['processes'] = []
                    for row in csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True):
                        owned = self.belongs_to_run(int(row[0]))
                        entry = {'pid': int(row[0]), 'name': row[1], 'memory_mib': row[2],
                                 'belongs_to_run': owned}
                        record['processes'].append(entry)
                        if owned is False:
                            self.foreign.append(dict(monotonic_s=record['monotonic_s'], **entry))
                except Exception as exc:
                    record['error'] = str(exc)
                    self.errors.append(record)
                stream.write(json.dumps(record) + '\n')
                self.stop_event.wait(2)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=25)
        return {'foreign_observations': self.foreign, 'query_errors': self.errors,
                'continuous_sampling': '2 seconds; races between samples remain possible'}
