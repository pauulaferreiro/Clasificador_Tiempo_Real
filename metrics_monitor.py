#metrics_monitor.py
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, List

import psutil

### Gestión de GPU (en mJ) ###
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

### Gestión de CPU (en uJ) ###
try:
    import pyRAPL
    pyRAPL.setup()
    PYRAPL_AVAILABLE = True
except Exception:
    PYRAPL_AVAILABLE = False

@dataclass
class SamplePoint:
    t: float
    cpu_process_pct: float
    ram_process_mb: float
    gpu_util_pct: Optional[float] = None
    gpu_mem_mb: Optional[float] = None
    gpu_power_w: Optional[float] = None


class ResourceMonitor:
    def __init__(self, sample_interval: float = 0.05, gpu_index: int = 0):
        self.sample_interval = sample_interval
        self.gpu_index = gpu_index
        self.process = psutil.Process(os.getpid())
        self.samples: List[SamplePoint] = []
        self._running = False
        self._thread = None
        self._t0 = None
        self._t1 = None
        self._gpu_handle = None
        self._rapl_meter = None
        self._gpu_energy_mj_start = None
        self._gpu_energy_mj_end = None

        if PYRAPL_AVAILABLE:
            try:
                self._rapl_meter = pyRAPL.Measurement('monitor_cpu')
            except Exception:
                self._rapl_meter = None

        if NVML_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._gpu_handle = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self):

        self.samples = []
        self._running = True
        self._t0 = time.perf_counter()

        if self._gpu_handle is not None:
            try:
                self._gpu_energy_mj_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._gpu_handle) #documentacion nvidia
            except Exception:
                self._gpu_energy_mj_start = None

        if self._rapl_meter:
            self._rapl_meter.begin() #documentacion pyrapl

        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        self._t1 = time.perf_counter()

        if self._gpu_handle is not None:
            try:
                self._gpu_energy_mj_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._gpu_handle) #documentacion nvidia
            except Exception:
                self._gpu_energy_mj_end = None

        if self._rapl_meter:
            try:
                self._rapl_meter.end() #documentacion pyrapl
            except Exception:
                pass


    def _sample_loop(self):
        while self._running:
            now = time.perf_counter()

            try:
                cpu_process_pct = self.process.cpu_percent(interval=None)
                ram_process_mb = self.process.memory_info().rss / (1024 ** 2)
            except Exception:
                cpu_process_pct, ram_process_mb = 0.0, 0.0

            gpu_util_pct, gpu_mem_mb, gpu_power_w = None, None, None
            if self._gpu_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)

                    gpu_util_pct = float(util.gpu)
                    gpu_mem_mb = mem.used / (1024 ** 2)
                    gpu_power_w = power_mw / 1000.0
                except Exception:
                    pass

            self.samples.append(
                SamplePoint(
                    t=now,
                    cpu_process_pct=cpu_process_pct,
                    ram_process_mb=ram_process_mb,
                    gpu_util_pct=gpu_util_pct,
                    gpu_mem_mb=gpu_mem_mb,
                    gpu_power_w=gpu_power_w,
                )
            )
            time.sleep(self.sample_interval)

    def finalize(self) -> dict:
        latency_total_ms = int((self._t1 - self._t0) * 1000) if self._t0 and self._t1 else 0
        duration_s = latency_total_ms / 1000.0 if latency_total_ms > 0 else 0.0

        gpu_energy_j = 0.0
        if self._gpu_energy_mj_start is not None and self._gpu_energy_mj_end is not None:
            delta = self._gpu_energy_mj_end - self._gpu_energy_mj_start
            if delta >= 0:
                gpu_energy_j = delta / 1000                             #pynvml mide el valor en mJ

        cpu_energy_j = None
        if self._rapl_meter and self._rapl_meter.result:
            cpu_energy_j = sum(self._rapl_meter.result.pkg) / 1e6       #rapl guarda los valores en uJ

        total_energy_j = gpu_energy_j + (cpu_energy_j or 0.0)
        
        
        gpu_power_vals = [s.gpu_power_w for s in self.samples if s.gpu_power_w is not None]
        cpu_vals = [s.cpu_process_pct for s in self.samples]
        ram_vals = [s.ram_process_mb for s in self.samples]
        gpu_util_vals = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        gpu_mem_vals = [s.gpu_mem_mb for s in self.samples if s.gpu_mem_mb is not None]

        gpu_power_avg_w = (gpu_energy_j / duration_s) if duration_s > 0 else None
        cpu_power_avg_w = (cpu_energy_j / duration_s) if cpu_energy_j is not None and duration_s > 0 else None
        gpu_util_avg_pct = (sum(gpu_util_vals) / len(gpu_util_vals)) if gpu_util_vals else None
        cpu_process_avg_pct = (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None
        ram_process_avg_mb = (sum(ram_vals) / len(ram_vals)) if ram_vals else None
        gpu_mem_avg_mb = (sum(gpu_mem_vals) / len(gpu_mem_vals)) if gpu_mem_vals else None

        return {
            "latency_total_ms": latency_total_ms,
            "latency_total_s": round(duration_s, 6),

            "gpu_energy_j": round(gpu_energy_j, 6),
            "cpu_energy_j": round(cpu_energy_j, 6) if cpu_energy_j is not None else None,
            "total_energy_j": round(total_energy_j, 6),

            "gpu_power_avg_w": round(gpu_power_avg_w, 6) if gpu_power_avg_w is not None else None,
            "cpu_power_avg_w": round(cpu_power_avg_w, 6) if cpu_power_avg_w is not None else None,
            "gpu_power_max_w": round(max(gpu_power_vals), 6) if gpu_power_vals else None,

            "gpu_util_avg_pct": round(gpu_util_avg_pct, 4) if gpu_util_avg_pct is not None else None,
            "cpu_process_avg_pct": round(cpu_process_avg_pct, 4) if cpu_process_avg_pct is not None else None,

            "gpu_mem_avg_mb": round(gpu_mem_avg_mb, 4) if gpu_mem_avg_mb is not None else None,
            "gpu_mem_max_mb": round(max(gpu_mem_vals), 4) if gpu_mem_vals else None,
            "ram_process_avg_mb": round(ram_process_avg_mb, 4) if ram_process_avg_mb is not None else None,
        }


def aggregate_metric_dicts(metric_dicts: List[dict]) -> dict:
    if not metric_dicts:
        return {
            "latency_total_ms": 0,
            "latency_total_s": 0.0,

            "gpu_energy_j": None,
            "cpu_energy_j": None,
            "total_energy_j": None,

            "energy_per_token_j": None,
            "latency_per_token_s": None,
            "tokens_per_second": None,
            
            "gpu_power_avg_w": None,
            "cpu_power_avg_w": None,
            "gpu_power_max_w": None,

            "gpu_util_avg_pct": None,
            "cpu_process_avg_pct": None,

            "gpu_mem_avg_mb": None,
            "gpu_mem_max_mb": None,
            "ram_process_avg_mb": None,
            
            "tokens_in_total": 0,
            "tokens_out_total": 0,
            "num_valid_frames": 0,
        }

    def get_valid(key):
        return [m[key] for m in metric_dicts if m.get(key) is not None]

    def weighted_avg(key, weight_key="latency_total_s"):
        weighted_sum = 0.0
        total_weight = 0.0
        for m in metric_dicts:
            value = m.get(key)
            weight = m.get(weight_key, 0) or 0
            if value is not None and weight > 0:
                weighted_sum += value * weight
                total_weight += weight
        return (weighted_sum / total_weight) if total_weight > 0 else None

    total_latency_ms = sum(m.get("latency_total_ms", 0) or 0 for m in metric_dicts)
    duration_s = total_latency_ms / 1000.0 if total_latency_ms > 0 else 0.0

    total_gpu_j = sum(get_valid("gpu_energy_j"))
    total_cpu_j = sum(get_valid("cpu_energy_j"))
    total_energy_j = total_gpu_j + total_cpu_j

    tokens_in_total = sum(m.get("tokens_in", 0) or 0 for m in metric_dicts)
    tokens_out_total = sum(m.get("tokens_out", 0) or 0 for m in metric_dicts)

    energy_per_token_j = total_energy_j / tokens_out_total if tokens_out_total > 0 else None
    latency_per_token_s = duration_s / tokens_out_total if tokens_out_total > 0 else None
    tokens_per_second = tokens_out_total / duration_s if duration_s > 0 else None

    gpu_power_max_vals = get_valid("gpu_power_max_w")
    gpu_mem_max_vals = get_valid("gpu_mem_max_mb")

    gpu_power_avg_w = (total_gpu_j / duration_s) if duration_s > 0 else None
    cpu_power_avg_w = (total_cpu_j / duration_s) if duration_s > 0 else None
    gpu_util_avg_pct = weighted_avg("gpu_util_avg_pct")
    cpu_process_avg_pct = weighted_avg("cpu_process_avg_pct")
    ram_process_avg_mb = weighted_avg("ram_process_avg_mb")
    gpu_mem_avg_mb = weighted_avg("gpu_mem_avg_mb")

    return {
        "latency_total_ms": total_latency_ms,
        "latency_total_s": round(duration_s, 6),

        "gpu_energy_j": round(total_gpu_j, 6),
        "cpu_energy_j": round(total_cpu_j, 6),
        "total_energy_j": round(total_energy_j, 6),

        # --- gráficas principales -> utilizar tokens/s
        "energy_per_token_j": round(energy_per_token_j, 6) if energy_per_token_j is not None else None,
        "latency_per_token_s": round(latency_per_token_s, 6) if latency_per_token_s is not None else None,
        "tokens_per_second": round(tokens_per_second, 6) if tokens_per_second is not None else None,

        "gpu_power_avg_w": round(gpu_power_avg_w, 6) if gpu_power_avg_w is not None else None,
        "cpu_power_avg_w": round(cpu_power_avg_w, 6) if cpu_power_avg_w is not None else None,
        "gpu_power_max_w": round(max(gpu_power_max_vals), 6) if gpu_power_max_vals else None,

        "gpu_util_avg_pct": round(gpu_util_avg_pct, 4) if gpu_util_avg_pct is not None else None,
        "cpu_process_avg_pct": round(cpu_process_avg_pct, 4) if cpu_process_avg_pct is not None else None,

        # uso medio VRAM
        "gpu_mem_avg_mb": round(gpu_mem_avg_mb, 4) if gpu_mem_avg_mb is not None else None,
        # pico máximo de VRAM
        "gpu_mem_max_mb": round(max(gpu_mem_max_vals), 4) if gpu_mem_max_vals else None,
        # uso medio de RAM
        "ram_process_avg_mb": round(ram_process_avg_mb, 4) if ram_process_avg_mb is not None else None,

        "tokens_in_total": tokens_in_total,
        "tokens_out_total": tokens_out_total,
        "num_valid_frames": len(metric_dicts),
    }