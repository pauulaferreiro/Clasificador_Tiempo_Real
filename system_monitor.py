import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import psutil

# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN GPU (pynvml)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    NUM_GPUS = pynvml.nvmlDeviceGetCount()
    GPU_HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(NUM_GPUS)]
except Exception:
    NVML_AVAILABLE = False
    NUM_GPUS = 0
    GPU_HANDLES = []

# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN CPU (pyRAPL)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pyRAPL
    pyRAPL.setup()
    PYRAPL_AVAILABLE = True
except Exception:
    PYRAPL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CPU / RAM auxiliares (Todo el sistema)
# ─────────────────────────────────────────────────────────────────────────────
def read_cpu_ram_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu_system_pct": round(psutil.cpu_percent(interval=None), 2),
        "ram_system_used_mb": round(vm.used / (1024 ** 2), 2),
        "ram_system_total_mb": round(vm.total / (1024 ** 2), 2),
        "ram_system_pct": round(vm.percent, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Procesos del proyecto (Micro-gestión)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_PROCESS_KEYWORDS = [
    "controller.py",
    "receiver_signal.py",
    "pipeline.py",
    "metrics_monitor.py",
    "ffmpeg",
    "system_monitor.py"
]

def read_project_processes() -> List[dict]:
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info"]):
        try:
            info = proc.info
            cmdline_list = info.get("cmdline") or []
            cmdline = " ".join(cmdline_list)
            name = info.get("name") or ""

            text = f"{name} {cmdline}".lower()

            if any(keyword.lower() in text for keyword in PROJECT_PROCESS_KEYWORDS):
                mem_info = info.get("memory_info")
                rss_mb = mem_info.rss / (1024 ** 2) if mem_info else 0.0

                processes.append({
                    "pid": info["pid"],
                    "name": name,
                    "cmdline": cmdline,
                    "cpu_pct": proc.cpu_percent(interval=None),
                    "rss_mb": round(rss_mb, 2),
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return processes

def summarize_project_processes(processes: List[dict]) -> dict:
    total_cpu = sum(p["cpu_pct"] for p in processes)
    total_ram = sum(p["rss_mb"] for p in processes)

    names = []
    for p in processes:
        label = p["name"]
        if "controller.py" in p["cmdline"]:
            label = "controller.py"
        elif "receiver_signal.py" in p["cmdline"]:
            label = "receiver_signal.py"
        elif "ffmpeg" in p["cmdline"].lower() or "ffmpeg" in p["name"].lower():
            label = "ffmpeg"
        elif "system_monitor.py" in p["cmdline"]:      
            label = "system_monitor.py"
        names.append(f"{label}[{p['pid']}]")

    return {
        "project_num_processes": len(processes),
        "project_cpu_pct_sum": round(total_cpu, 2),
        "project_ram_mb_sum": round(total_ram, 2),
        "project_processes": ", ".join(names),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

def safe_avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def safe_min(values: List[float]) -> float:
    return min(values) if values else 0.0

def safe_max(values: List[float]) -> float:
    return max(values) if values else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Informe final
# ─────────────────────────────────────────────────────────────────────────────
def build_report(
    duration_s: float, cpu_energy_j: float, gpu_energy_j: float,
    cpu_power_samples: List[float], gpu_power_samples: List[float],
    cpu_usage_samples: List[float], gpu_util_samples: List[float],
    gpu_mem_samples: List[float], ram_samples: List[float],
    project_cpu_samples: List[float], project_ram_samples: List[float],
    price_kwh: float, start_ts: str, end_ts: str,
    num_samples: int, interval_s: float
) -> str:

    total_energy_j = cpu_energy_j + gpu_energy_j
    total_energy_kwh = total_energy_j / 3_600_000.0

    cost_session = total_energy_kwh * price_kwh

    if duration_s > 0:
        avg_power_total_w = total_energy_j / duration_s
        cost_per_hour = (avg_power_total_w * 3600 / 3_600_000.0) * price_kwh
    else:
        avg_power_total_w = 0.0
        cost_per_hour = 0.0

    cost_per_day = cost_per_hour * 24
    cost_per_month = cost_per_day * 30
    separator = "─" * 65

    lines = [
        "",
        "=" * 65,
        "  INFORME DE COSTE OPERATIVO DEL SISTEMA",
        "=" * 65,
        f"  Inicio:              {start_ts}",
        f"  Fin:                 {end_ts}",
        f"  Duración total:      {format_duration(duration_s)}",
        f"  Muestras tomadas:    {num_samples} (cada {interval_s}s)",
        separator,
        "  DISPONIBILIDAD DE SENSORES",
        separator,
        f"  CPU pyRAPL:          {'disponible' if PYRAPL_AVAILABLE else 'NO disponible'}",
        f"  GPU NVML:            {'disponible' if NVML_AVAILABLE else 'NO disponible'}",
        separator,
        "  ENERGÍA MEDIDA/ESTIMADA",
        separator,
        f"  CPU pyRAPL:          {cpu_energy_j / 1000:.3f} kJ  ({cpu_energy_j:.1f} J)",
        f"  GPU NVML (Exacta):   {gpu_energy_j / 1000:.3f} kJ  ({gpu_energy_j:.1f} J)",
        f"  TOTAL CPU+GPU:       {total_energy_j / 1000:.3f} kJ  →  {total_energy_kwh * 1000:.4f} Wh",
        "",
        "  Nota: este total NO representa el consumo eléctrico total de todo el PC.",
        "        Representa CPU package (pyRAPL) + GPU NVIDIA (NVML Counter).",
        separator,
        "  POTENCIA (W = J/s)",
        separator,
        f"  Potencia media total CPU+GPU:  {avg_power_total_w:.2f} W",
        f"  CPU media pyRAPL:                {safe_avg(cpu_power_samples):.2f} W",
        f"  CPU mínima pyRAPL:               {safe_min(cpu_power_samples):.2f} W",
        f"  CPU máxima pyRAPL:               {safe_max(cpu_power_samples):.2f} W",
        f"  GPU media NVML:                {safe_avg(gpu_power_samples):.2f} W",
        f"  GPU mínima NVML:               {safe_min(gpu_power_samples):.2f} W",
        f"  GPU máxima NVML:               {safe_max(gpu_power_samples):.2f} W",
        separator,
        "  USO DE HARDWARE",
        separator,
        f"  CPU sistema media:             {safe_avg(cpu_usage_samples):.2f} %",
        f"  GPU utilización media:         {safe_avg(gpu_util_samples):.2f} %",
        f"  GPU VRAM media:                {safe_avg(gpu_mem_samples):.0f} MB",
        f"  GPU VRAM pico:                 {safe_max(gpu_mem_samples):.0f} MB",
        f"  RAM sistema media:             {safe_avg(ram_samples):.0f} MB",
        f"  RAM sistema pico:              {safe_max(ram_samples):.0f} MB",
        separator,
        "  PROCESOS DEL PROYECTO",
        separator,
        f"  CPU procesos proyecto media:   {safe_avg(project_cpu_samples):.2f} %",
        f"  RAM procesos proyecto media:   {safe_avg(project_ram_samples):.2f} MB",
        separator,
        f"  COSTE FINANCIERO (@ {price_kwh:.3f} €/kWh)",
        separator,
        f"  Esta sesión:                   {cost_session:.6f} €",
        f"  Por hora:                      {cost_per_hour:.6f} €/h",
        f"  Por día (24h):                 {cost_per_day:.6f} €/día",
        f"  Por mes (30 días):             {cost_per_month:.4f} €/mes",
        "=" * 65,
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Bucle principal
# ─────────────────────────────────────────────────────────────────────────────
def run_monitor(
    interval_s: float,
    duration_s: Optional[float],
    price_kwh: float,
    output_csv: Path,
    output_report: Path,
):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    # Acumuladores e históricos
    cpu_power_samples: List[float] = []
    gpu_power_samples: List[float] = []
    cpu_usage_samples: List[float] = []
    gpu_util_samples: List[float] = []
    gpu_mem_samples: List[float] = []
    ram_samples: List[float] = []
    project_cpu_samples: List[float] = []
    project_ram_samples: List[float] = []

    cpu_total_j = 0.0
    gpu_energy_j = 0.0
    num_samples = 0

    # Configuración de los medidores de energía
    rapl_meter = None
    if PYRAPL_AVAILABLE:
        try:
            rapl_meter = pyRAPL.Measurement('global_monitor')
            rapl_meter.begin()
        except Exception:
            print(" [ERROR] pyRAPL falló al inicializar. Ejecuta con sudo.")
            rapl_meter = None

    prev_gpu_energy_mj = 0
    if NVML_AVAILABLE and NUM_GPUS > 0:
        try:
            prev_gpu_energy_mj = sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in GPU_HANDLES)
        except Exception:
            pass

    start_time = time.perf_counter()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_file = output_csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    writer.writerow([
        "timestamp", "elapsed_s",
        "cpu_energy_rapl_total_j", "cpu_energy_rapl_delta_j", "cpu_power_rapl_w", "cpu_system_pct",
        "gpu_energy_total_j", "gpu_power_w", "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
        "ram_used_mb", "ram_total_mb", "ram_pct",
        "project_num_processes", "project_cpu_pct_sum", "project_ram_mb_sum", "project_processes",
    ])

    print("\n" + "=" * 65)
    print("  MONITOR DE COSTE OPERATIVO (Optimizado pyRAPL/NVML)")
    print(f"  Inicio: {start_ts}")
    print(f"  Muestreo cada {interval_s}s  |  Precio: {price_kwh} €/kWh")
    
    if duration_s is not None:
        print(f"  Duración programada: {format_duration(duration_s)}")
    else:
        print("  Duración programada: indefinida, detener con Ctrl+C")

    print(f"  [OK] pyRAPL: {'Disponible' if PYRAPL_AVAILABLE else 'No disponible'}")
    print(f"  [OK] NVML: {'Disponible (' + str(NUM_GPUS) + ' GPUs)' if NVML_AVAILABLE else 'No disponible'}")
    print("  Detener con Ctrl+C")
    print("=" * 65 + "\n")

    psutil.cpu_percent(interval=None) # Inicializar
    last_sample_time = time.perf_counter()

    def finalize_and_exit(signum=None, frame=None):
        nonlocal csv_file
        end_time = time.perf_counter()
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_duration_s = end_time - start_time

        try:
            csv_file.close()
        except Exception:
            pass

        report = build_report(
            duration_s=total_duration_s,
            cpu_energy_j=cpu_total_j,
            gpu_energy_j=gpu_energy_j,
            cpu_power_samples=cpu_power_samples,
            gpu_power_samples=gpu_power_samples,
            cpu_usage_samples=cpu_usage_samples,
            gpu_util_samples=gpu_util_samples,
            gpu_mem_samples=gpu_mem_samples,
            ram_samples=ram_samples,
            project_cpu_samples=project_cpu_samples,
            project_ram_samples=project_ram_samples,
            price_kwh=price_kwh,
            start_ts=start_ts,
            end_ts=end_ts,
            num_samples=num_samples,
            interval_s=interval_s,
        )

        print(report)
        output_report.write_text(report, encoding="utf-8")
        print(f"  Informe guardado en: {output_report}")
        print(f"  Muestras guardadas en: {output_csv}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, finalize_and_exit)
    signal.signal(signal.SIGTERM, finalize_and_exit)

    # ─────────────────────────────────────────────────────────────────────
    # BUCLE DE MONITORIZACIÓN
    # ─────────────────────────────────────────────────────────────────────
    while True:
        loop_start = time.perf_counter()
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_s = loop_start - start_time

        if duration_s is not None and elapsed_s >= duration_s:
            finalize_and_exit()

        dt = loop_start - last_sample_time
        last_sample_time = loop_start
        if dt <= 0:
            dt = interval_s

        # 1. CPU (pyRAPL)
        cpu_delta_j = None
        cpu_power_w = None
        if rapl_meter:
            try:
                rapl_meter.end()
                cpu_delta_j = sum(rapl_meter.result.pkg) / 1e6
                cpu_total_j += cpu_delta_j
                cpu_power_w = cpu_delta_j / dt if dt > 0 else 0
                cpu_power_samples.append(cpu_power_w)
            except Exception:
                pass
            finally:
                # Ponemos el begin() en el finally para obligar a que SIEMPRE 
                # se reinicie el contador, incluso si la lectura anterior falló
                try:
                    rapl_meter.begin()
                except Exception:
                    pass

        # 2. GPU (Contador acumulativo pynvml)
        gpu_delta_j = None
        gpu_power_w = None
        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None

        if NVML_AVAILABLE and NUM_GPUS > 0:
            try:
                # Energía acumulada en mJ -> Julios
                current_gpu_energy_mj = sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in GPU_HANDLES)
                if current_gpu_energy_mj >= prev_gpu_energy_mj:
                    gpu_delta_j = (current_gpu_energy_mj - prev_gpu_energy_mj) / 1000.0
                else:
                    gpu_delta_j = 0 # Protección contra overflows del contador
                
                gpu_energy_j += gpu_delta_j
                prev_gpu_energy_mj = current_gpu_energy_mj
                
                # Vatios reales calculados (Energía / Tiempo) en vez de lectura instantánea
                gpu_power_w = gpu_delta_j / dt if dt > 0 else 0
                gpu_power_samples.append(gpu_power_w)
            except Exception:
                pass

            # Obtener datos de estado en vivo (Uso %, VRAM)
            total_util, total_mem_used, total_mem_total, valid_gpus = 0.0, 0.0, 0.0, 0
            for h in GPU_HANDLES:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    total_util += util.gpu
                    total_mem_used += mem.used / (1024 ** 2)
                    total_mem_total += mem.total / (1024 ** 2)
                    valid_gpus += 1
                except Exception:
                    continue

            if valid_gpus > 0:
                gpu_util = total_util / valid_gpus
                gpu_mem_used = total_mem_used
                gpu_mem_total = total_mem_total
                gpu_util_samples.append(gpu_util)
                gpu_mem_samples.append(gpu_mem_used)

        # 3. CPU/RAM Global
        cpu_ram = read_cpu_ram_stats()
        cpu_pct = cpu_ram["cpu_system_pct"]
        ram_used = cpu_ram["ram_system_used_mb"]
        ram_total = cpu_ram["ram_system_total_mb"]
        ram_pct = cpu_ram["ram_system_pct"]

        cpu_usage_samples.append(cpu_pct)
        ram_samples.append(ram_used)

        # 4. Procesos del proyecto (FFmpeg, controller, pipeline)
        project_processes = read_project_processes()
        project_summary = summarize_project_processes(project_processes)
        project_cpu_samples.append(project_summary["project_cpu_pct_sum"])
        project_ram_samples.append(project_summary["project_ram_mb_sum"])

        num_samples += 1

        writer.writerow([
            now_ts, round(elapsed_s, 2),
            round(cpu_total_j, 6) if cpu_total_j is not None else None,
            round(cpu_delta_j, 6) if cpu_delta_j is not None else None,
            round(cpu_power_w, 6) if cpu_power_w is not None else None,
            cpu_pct,
            round(gpu_energy_j, 6),
            round(gpu_power_w, 6) if gpu_power_w is not None else None,
            round(gpu_util, 2) if gpu_util is not None else None,
            round(gpu_mem_used, 2) if gpu_mem_used is not None else None,
            round(gpu_mem_total, 2) if gpu_mem_total is not None else None,
            ram_used, ram_total, ram_pct,
            project_summary["project_num_processes"],
            project_summary["project_cpu_pct_sum"],
            project_summary["project_ram_mb_sum"],
            project_summary["project_processes"],
        ])
        csv_file.flush()

        cpu_power_str = f"{cpu_power_w:6.2f} W" if cpu_power_w is not None else "pyRAPL OFF"
        gpu_str = (
            f"{gpu_power_w:6.2f} W  util={gpu_util:5.1f}%  VRAM={gpu_mem_used:7.0f} MB"
            if gpu_power_w is not None else "GPU OFF"
        )

        print(
            f"[{now_ts}] +{elapsed_s:8.1f}s | "
            f"CPU: {cpu_power_str} | "
            f"GPU: {gpu_str} | "
            f"Sys CPU: {cpu_pct:5.1f}% | "
            f"Proyecto: {project_summary['project_num_processes']} proc, "
            f"CPU={project_summary['project_cpu_pct_sum']:5.1f}%, "
            f"RAM={project_summary['project_ram_mb_sum']:7.1f} MB"
        )

        elapsed_loop = time.perf_counter() - loop_start
        sleep_time = max(0.0, interval_s - elapsed_loop)
        time.sleep(sleep_time)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Monitor de coste operativo global optimizado con pyRAPL y GPU exact counters."
    )

    parser.add_argument("--interval", type=float, default=5.0, help="Segundos entre muestras. Default: 5")
    parser.add_argument("--duration", type=float, default=None, help="Duración en segundos. Default: indefinido")
    parser.add_argument("--price-kwh", type=float, default=0.18, help="Precio eléctrico €/kWh. Default: 0.18")
    parser.add_argument("--output-csv", type=str, default="./logs/system_monitor.csv", help="Ruta CSV.")
    parser.add_argument("--output-report", type=str, default="./logs/system_monitor_report.txt", help="Ruta Informe.")

    args = parser.parse_args()

    run_monitor(
        interval_s=args.interval,
        duration_s=args.duration,
        price_kwh=args.price_kwh,
        output_csv=Path(args.output_csv),
        output_report=Path(args.output_report),
    )

if __name__ == "__main__":
    main()