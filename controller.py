import os
import glob
import time
import json
import argparse
import re
import csv
from collections import Counter

import cv2
from skimage.metrics import structural_similarity as ssim
from logger_config import log, monitor_latency
from metrics_monitor import aggregate_metric_dicts

from pipeline import VideoClassifier


def get_sequence_number(filepath: str) -> int:
    """
    Extrae el número de secuencia de un frame.
    """
    match = re.search(r'(\d+)\.jpg$', filepath)
    return int(match.group(1)) if match else -1


def get_latest_folder(base_dir: str) -> str:
    folders = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not folders:
        return None

    return sorted(folders)[-1]


def append_to_final_results_csv(csv_path: str, data: dict):
    """
    Guarda el veredicto final del modelo cuando cambia el evento.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "SERVICIO",
                "EVENTO",
                "PREDICCION_FINAL_LLM",
                "FRAMES_GENERADOS",
                "FRAMES_ANALIZADOS",
                "FRAMES_DESCARTADOS",
                "SSIM_THRESHOLD",
                "LAPLACIAN_MIN",
                "LAPLACIAN_MAX"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def load_existing_json(json_path: str) -> dict:
    """
    Carga un JSON existente(si existe)
    """
    base = {
        "frame_predictions": [],
        "filter_log": [],
        "frames_totales_generados": 0,
        "frames_totales_analizados": 0,
        "frames_totales_descartados": 0
    }

    if not os.path.exists(json_path):
        return base

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "frame_predictions" not in data:
            data["frame_predictions"] = []

        if "filter_log" not in data:
            data["filter_log"] = []

        if "frames_totales_generados" not in data:
            data["frames_totales_generados"] = 0

        if "frames_totales_analizados" not in data:
            data["frames_totales_analizados"] = len(data["frame_predictions"])

        if "frames_totales_descartados" not in data:
            data["frames_totales_descartados"] = 0

        return data

    except json.JSONDecodeError:
        return base


def get_majority_vote(frame_predictions: list) -> str:
    """
    Calcula la categoría ganadora por mayoría de votos (ignora Undefined)
    """
    votes = [
        p.get("prediction")
        for p in frame_predictions
        if p.get("prediction") not in ("Undefined", "Error", None, "")
    ]

    if not votes:
        return "Undefined"

    return Counter(votes).most_common(1)[0][0]


def get_last_analyzed_frame_from_json(json_path: str) -> str:

    data = load_existing_json(json_path)
    preds = data.get("frame_predictions", [])

    if not preds:
        return None

    last = preds[-1]
    return last.get("frame_path")


def read_gray_image(image_path: str):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def laplacian_score(image_path: str) -> float:
    img = read_gray_image(image_path)

    if img is None:
        return 0.0

    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def ssim_score(image_path_a: str, image_path_b: str) -> float:
    img_a = read_gray_image(image_path_a)
    img_b = read_gray_image(image_path_b)

    if img_a is None or img_b is None:
        return 0.0

    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    return float(ssim(img_a, img_b))


def filter_candidate_frames(
    candidate_frames: list,
    previous_analyzed_frame: str = None,
    ssim_threshold: float = 0.6,
    laplacian_min: float = 70.0,
    laplacian_max: float = 1500.0
):

    accepted_frames = []
    filter_info = []

    last_reference_frame = previous_analyzed_frame

    for frame_path in candidate_frames:
        seq = get_sequence_number(frame_path)
        lap_score = laplacian_score(frame_path)

        if lap_score < laplacian_min:
            filter_info.append({
                "frame_path": frame_path,
                "frame_sequence": seq,
                "accepted": False,
                "reason": "laplacian_below_min",
                "laplacian_score": lap_score,
                "laplacian_min": laplacian_min,
                "laplacian_max": laplacian_max,
                "ssim_score": None,
                "ssim_threshold": ssim_threshold,
                "reference_frame": last_reference_frame
            })
            continue

        if lap_score > laplacian_max:
            filter_info.append({
                "frame_path": frame_path,
                "frame_sequence": seq,
                "accepted": False,
                "reason": "laplacian_above_max",
                "laplacian_score": lap_score,
                "laplacian_min": laplacian_min,
                "laplacian_max": laplacian_max,
                "ssim_score": None,
                "ssim_threshold": ssim_threshold,
                "reference_frame": last_reference_frame
            })
            continue

        sim_score = None

        if last_reference_frame and os.path.exists(last_reference_frame):
            sim_score = ssim_score(last_reference_frame, frame_path)

            if sim_score > ssim_threshold:
                filter_info.append({
                    "frame_path": frame_path,
                    "frame_sequence": seq,
                    "accepted": False,
                    "reason": "ssim_too_similar",
                    "laplacian_score": lap_score,
                    "laplacian_min": laplacian_min,
                    "laplacian_max": laplacian_max,
                    "ssim_score": sim_score,
                    "ssim_threshold": ssim_threshold,
                    "reference_frame": last_reference_frame
                })
                continue

        accepted_frames.append(frame_path)

        filter_info.append({
            "frame_path": frame_path,
            "frame_sequence": seq,
            "accepted": True,
            "reason": "accepted",
            "laplacian_score": lap_score,
            "laplacian_min": laplacian_min,
            "laplacian_max": laplacian_max,
            "ssim_score": sim_score,
            "ssim_threshold": ssim_threshold,
            "reference_frame": last_reference_frame
        })
        # El siguiente frame se compara contra el último frame ACEPTADO,
        last_reference_frame = frame_path

    return accepted_frames, filter_info, last_reference_frame


def normalize_classifier_output(result):
    if isinstance(result, tuple) and len(result) == 2:
        batch_winner, new_predictions = result

    elif isinstance(result, tuple) and len(result) == 3:
        batch_winner, new_predictions, _metrics = result

    else:
        batch_winner = result
        new_predictions = []

    if isinstance(batch_winner, dict):
        batch_winner = batch_winner.get("prediction", "Undefined")

    if not batch_winner:
        batch_winner = "Undefined"

    if new_predictions is None:
        new_predictions = []

    return batch_winner, new_predictions

@monitor_latency
def main():
    parser = argparse.ArgumentParser(
        description="Demonio multi-servicio para clasificación en tiempo real"
    )

    parser.add_argument(
        "--base-frames-dir",
        default="./RESULTADOS_MUX/frames_seleccionados",
        help="Carpeta raíz donde están los frames por servicio"
    )

    parser.add_argument(
        "--base-eit-dir",
        default="./RESULTADOS_MUX/eit_extraidas",
        help="Carpeta raíz donde están los XML EIT por servicio"
    )

    parser.add_argument(
        "--json-out-dir",
        default="./resultados_frames",
        help="Carpeta donde se guardarán los JSON de predicciones"
    )

    parser.add_argument(
        "--final-csv",
        default="./RESULTADOS_MUX/reporte_final_predicciones.csv",
        help="CSV final con la predicción del modelo por evento"
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Segundos entre escaneos de carpetas"
    )

    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.6,
        help="Umbral SSIM. Se aceptan frames con SSIM <= umbral respecto al último frame analizado."
    )

    parser.add_argument(
        "--laplacian-min",
        type=float,
        default=70.0,
        help="Umbral inferior del Laplaciano. Por debajo se descarta por borroso."
    )

    parser.add_argument(
        "--laplacian-max",
        type=float,
        default=1500.0,
        help="Umbral superior del Laplaciano. Por encima se descarta por ruido/artefactos."
    )

    args = parser.parse_args()

    BASE_FRAMES_DIR = os.path.abspath(args.base_frames_dir)
    BASE_EIT_DIR = os.path.abspath(args.base_eit_dir)
    JSON_OUT_DIR = os.path.abspath(args.json_out_dir)
    FINAL_CSV = os.path.abspath(args.final_csv)

    os.makedirs(JSON_OUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("INICIANDO CLASIFICACIÓN EN TIEMPO REAL ")
    print(f"Frames: {BASE_FRAMES_DIR}")
    print(f"EIT XML: {BASE_EIT_DIR}")
    print(f"JSON salida: {JSON_OUT_DIR}")
    print(f"CSV final: {FINAL_CSV}")
    print(f"SSIM threshold: {args.ssim_threshold}")
    print(f"Laplacian min: {args.laplacian_min}")
    print(f"Laplacian max: {args.laplacian_max}")
    print("=" * 70)

    try:
        classifier = VideoClassifier()
    except Exception as e:
        print(" Fallo crítico cargando el modelo AI:", e)
        return

    state_memory = {}

    print(
        f"\n Modelo cargado. "
        f"Escaneando servicios cada {args.poll_interval}s. "
    )

    while True:
        try:
            if not os.path.exists(BASE_FRAMES_DIR):
                print(f" Esperando a que exista {BASE_FRAMES_DIR}")
                time.sleep(args.poll_interval)
                continue

            service_dirs = glob.glob(os.path.join(BASE_FRAMES_DIR, "*"))

            for srv_dir in service_dirs:
                if not os.path.isdir(srv_dir):
                    continue

                service_name = os.path.basename(srv_dir)
                latest_event_dir = get_latest_folder(srv_dir)

                if not latest_event_dir:
                    continue

                event_name = os.path.basename(latest_event_dir)

                service_json_dir = os.path.join(JSON_OUT_DIR, service_name)
                os.makedirs(service_json_dir, exist_ok=True)

                json_path = os.path.join(service_json_dir, f"{event_name}.json")

                ########## CAMBIO DE EVENTO  ##########
                if (
                    service_name in state_memory
                    and state_memory[service_name]["evento_actual"] != event_name
                ):
                    prev_event = state_memory[service_name]["evento_actual"]
                    prev_json_path = state_memory[service_name]["json_path"]

                    print(
                        f"\n {service_name}] "
                        f"Evento finalizado: {prev_event}"
                    )

                    if os.path.exists(prev_json_path):
                        res = load_existing_json(prev_json_path)

                        append_to_final_results_csv(
                            FINAL_CSV,
                            {
                                "SERVICIO": service_name,
                                "EVENTO": prev_event,
                                "PREDICCION_FINAL_LLM": res.get(
                                    "prediccion_global_actual",
                                    "Undefined"
                                ),
                                "FRAMES_GENERADOS": res.get(
                                    "frames_totales_generados",
                                    0
                                ),
                                "FRAMES_ANALIZADOS": res.get(
                                    "frames_totales_analizados",
                                    0
                                ),
                                "FRAMES_DESCARTADOS": res.get(
                                    "frames_totales_descartados",
                                    0
                                ),
                                "SSIM_THRESHOLD": res.get(
                                    "ssim_threshold",
                                    args.ssim_threshold
                                ),
                                "LAPLACIAN_MIN": res.get(
                                    "laplacian_min",
                                    args.laplacian_min
                                ),
                                "LAPLACIAN_MAX": res.get(
                                    "laplacian_max",
                                    args.laplacian_max
                                )
                            }
                        )

                        print(
                            f"Predicción final guardada en {FINAL_CSV}"
                        )

                ########## NUEVO EVENTO  ##########
                if (
                    service_name not in state_memory
                    or state_memory[service_name]["evento_actual"] != event_name
                ):
                    last_analyzed_from_json = get_last_analyzed_frame_from_json(
                        json_path
                    )

                    state_memory[service_name] = {
                        "evento_actual": event_name,
                        "ultimo_frame": -1,
                        "json_path": json_path,
                        "ultimo_frame_analizado_path": last_analyzed_from_json
                    }

                    print(
                        f"\n[ {service_name}] "
                        f"Monitorizando evento -> {event_name}"
                    )

                ########## FRAMES NUEVOS ##########
                frames = glob.glob(os.path.join(latest_event_dir, "*.jpg"))
                frames.sort(key=get_sequence_number)

                last_frame_processed = state_memory[service_name]["ultimo_frame"]

                new_frames = [
                    f for f in frames
                    if get_sequence_number(f) > last_frame_processed
                ]

                if not new_frames:
                    continue

                print(
                    f"[{service_name}] "
                    f"{len(new_frames)} nuevos frames en {event_name}"
                )


                xml_path = os.path.join(
                    BASE_EIT_DIR,
                    service_name,
                    f"{event_name}.xml"
                )

                if not os.path.exists(xml_path):
                    print(
                        f"[{service_name}] "
                        f"No se encontró XML EIT para {event_name}: {xml_path}"
                    )
                    xml_path = ""

                ########## FILTRADOS ##########
                previous_analyzed_frame = state_memory[service_name].get(
                    "ultimo_frame_analizado_path"
                )

                filtered_frames, filter_info, last_reference_frame = filter_candidate_frames(
                    candidate_frames=new_frames,
                    previous_analyzed_frame=previous_analyzed_frame,
                    ssim_threshold=args.ssim_threshold,
                    laplacian_min=args.laplacian_min,
                    laplacian_max=args.laplacian_max
                )

                log.info(
                    f"[{service_name}] "
                    f"Filtrado: {len(filtered_frames)}/{len(new_frames)} frames aceptados "
                    f"(SSIM <= {args.ssim_threshold}, "
                    f"Laplaciano entre {args.laplacian_min} y {args.laplacian_max})"
                )

                # Marcamos los frames como procesados por el filtro
                state_memory[service_name]["ultimo_frame"] = get_sequence_number(
                    new_frames[-1]
                )

                # Guardamos el último frame aceptado como referencia.
                state_memory[service_name]["ultimo_frame_analizado_path"] = (
                    last_reference_frame
                )


                # actualizar JSON cuando no se aceptan frames
                if not filtered_frames:
                    existing_data = load_existing_json(json_path)

                    existing_data["filter_log"].extend(filter_info)

                    metric_list = [
                        {
                            "latency_total_ms": p.get("latency_total_ms"),
                            "latency_total_s": p.get("latency_total_s"),
                            "gpu_energy_j": p.get("gpu_energy_j"),
                            "cpu_energy_j": p.get("cpu_energy_j"),
                            "total_energy_j": p.get("total_energy_j"),
                            "gpu_power_avg_w": p.get("gpu_power_avg_w"),
                            "cpu_power_avg_w": p.get("cpu_power_avg_w"),
                            "gpu_power_max_w": p.get("gpu_power_max_w"),
                            "gpu_util_avg_pct": p.get("gpu_util_avg_pct"),
                            "cpu_process_avg_pct": p.get("cpu_process_avg_pct"),
                            "gpu_mem_avg_mb": p.get("gpu_mem_avg_mb"),
                            "gpu_mem_max_mb": p.get("gpu_mem_max_mb"),
                            "ram_process_avg_mb": p.get("ram_process_avg_mb"),
                            "tokens_in": p.get("tokens_in"),
                            "tokens_out": p.get("tokens_out"),
                        }
                        for p in existing_data["frame_predictions"]
                    ]

                    aggregate_metrics = aggregate_metric_dicts(metric_list)

                    previous_global = existing_data.get(
                        "prediccion_global_actual",
                        "Undefined"
                    )

                    existing_data.update(
                        {
                            "servicio": service_name,
                            "evento": event_name,
                            "xml_path": xml_path,
                            "prediccion_ultimo_lote": previous_global,
                            "prediccion_global_actual": previous_global,
                            "frames_totales_generados": existing_data.get(
                                "frames_totales_generados",
                                0
                            ) + len(new_frames),
                            "frames_totales_analizados": len(
                                existing_data["frame_predictions"]
                            ),
                            "frames_totales_descartados": existing_data.get(
                                "frames_totales_descartados",
                                0
                            ) + (len(new_frames) - len(filtered_frames)),
                            "ssim_threshold": args.ssim_threshold,
                            "laplacian_min": args.laplacian_min,
                            "laplacian_max": args.laplacian_max,
                            "aggregate_metrics": aggregate_metrics,
                            "ultima_actualizacion_utc": time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        }
                    )

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            existing_data,
                            f,
                            indent=4,
                            ensure_ascii=False
                        )

                    print(
                        f"[{service_name}] "
                        f"Ningún frame pasa el filtrado. No se llama al modelo."
                    )

                    continue

                ########## INFERENCIA ##########
                result = classifier.classify(
                    image_paths=filtered_frames,
                    eit_xml_path=xml_path,
                    csv_metadata=None,
                    sample_name=f"{service_name}/{event_name}"
                )

                batch_winner, new_predictions = normalize_classifier_output(result)


                ########## ACTUALIZAR JSON (acumulado)  ##########
                existing_data = load_existing_json(json_path)

                existing_data["frame_predictions"].extend(new_predictions)
                existing_data["filter_log"].extend(filter_info)

                global_winner = get_majority_vote(
                    existing_data["frame_predictions"]
                )

                metric_list = [
                    {
                        "latency_total_ms": p.get("latency_total_ms"),
                        "latency_total_s": p.get("latency_total_s"),
                        "gpu_energy_j": p.get("gpu_energy_j"),
                        "cpu_energy_j": p.get("cpu_energy_j"),
                        "total_energy_j": p.get("total_energy_j"),
                        "gpu_power_avg_w": p.get("gpu_power_avg_w"),
                        "cpu_power_avg_w": p.get("cpu_power_avg_w"),
                        "gpu_power_max_w": p.get("gpu_power_max_w"),
                        "gpu_util_avg_pct": p.get("gpu_util_avg_pct"),
                        "cpu_process_avg_pct": p.get("cpu_process_avg_pct"),
                        "gpu_mem_avg_mb": p.get("gpu_mem_avg_mb"),
                        "gpu_mem_max_mb": p.get("gpu_mem_max_mb"),
                        "ram_process_avg_mb": p.get("ram_process_avg_mb"),
                        "tokens_in": p.get("tokens_in"),
                        "tokens_out": p.get("tokens_out"),
                    }
                    for p in existing_data["frame_predictions"]
                ]

                aggregate_metrics = aggregate_metric_dicts(metric_list)

                existing_data.update(
                    {
                        "servicio": service_name,
                        "evento": event_name,
                        "xml_path": xml_path,
                        "prediccion_ultimo_lote": batch_winner,
                        "prediccion_global_actual": global_winner,
                        "frames_totales_generados": existing_data.get(
                            "frames_totales_generados",
                            0
                        ) + len(new_frames),
                        "frames_totales_analizados": len(
                            existing_data["frame_predictions"]
                        ),
                        "frames_totales_descartados": existing_data.get(
                            "frames_totales_descartados",
                            0
                        ) + (len(new_frames) - len(filtered_frames)),
                        "ssim_threshold": args.ssim_threshold,
                        "laplacian_min": args.laplacian_min,
                        "laplacian_max": args.laplacian_max,
                        "aggregate_metrics": aggregate_metrics,
                        "ultima_actualizacion_utc": time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }
                )

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        existing_data,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )

                log.info(
                    f"[{service_name}] "
                    f"Predicción acumulada actual: {global_winner}"
                )

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            print("\nDetenido por el usuario.")
            break

        except Exception as e:
            print(f"\nError inesperado en el bucle: {e}")
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
