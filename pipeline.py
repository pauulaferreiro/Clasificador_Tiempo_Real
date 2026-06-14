import re
import os
import json
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List, Optional, Dict, Any
from logger_config import log, monitor_latency, latency_log
from metrics_monitor import ResourceMonitor, aggregate_metric_dicts

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText


MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"

CATEGORIES = [
    "Fiction", "News", "Show", "Sports",
    "Cartoons", "Music/Dance", "Arts/Culture",
    "Social", "Education/Science",
    "Leisure hobbies"
]


def parse_llm_json(text: str) -> dict:
    """
    Intenta extraer y parsear un JSON devuelto por el modelo.
    """
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
            else:
                raise ValueError("No se encontró un bloque JSON válido.")

        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)

        reasoning_match = re.search(
            r'"reasoning"\s*:\s*"(.*?)"\s*,\s*"predicted_category"',
            json_str,
            re.DOTALL
        )

        if reasoning_match:
            bad_reasoning = reasoning_match.group(1)
            safe_reasoning = bad_reasoning.replace('"', "'")
            json_str = json_str.replace(bad_reasoning, safe_reasoning)

        return json.loads(json_str, strict=False)

    except Exception as e:
        print(f"Error parseando JSON del modelo: {e}")
        print(f"--- TEXTO CRUDO ---\n{text}\n-------------------")
        return {
            "predicted_category": "Error",
            "confidence": 0,
            "reasoning": "Parse failed"
        }


class VideoClassifier:
    @monitor_latency
    def __init__(self):
        log.info(f"CARGANDO MODELO MINISTRAL 3-3B: {MODEL_ID}")

        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            fix_mistral_regex=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            #quantization_config=self.bnb_config, #CUANTIZACION
            dtype=torch.bfloat16,  # SIN CUANTIZACION
            device_map="auto",
            attn_implementation="sdpa",
        )

        log.info("Modelo cargado correctamente.")

    def parse_eit_metadata(self, xml_path: str) -> dict:
        eit_data = {
            "start_time": "Unknown",
            "duration": "Unknown",
            "running_status": "Unknown",
            "event_name": "Unknown",
            "parental_country": "Unknown",
            "parental_rating": "Unknown",
            "extended_text": "Sin descripción"
        }

        if not xml_path or not os.path.exists(xml_path):
            return eit_data

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            event = root.find(".//event")

            if event is not None:
                eit_data["start_time"] = event.get("start_time", "Unknown")
                eit_data["duration"] = event.get("duration", "Unknown")
                eit_data["running_status"] = event.get("running_status", "Unknown")

                extended_texts = []
                ext_descriptors = []

                for child in event:
                    if child.tag == "short_event_descriptor":
                        event_name_node = child.find("event_name")
                        if event_name_node is not None and event_name_node.text:
                            eit_data["event_name"] = event_name_node.text

                    elif child.tag == "parental_rating_descriptor":
                        country = child.find("country")
                        if country is not None:
                            eit_data["parental_country"] = country.get("country_code", "Unknown")
                            raw_rating = country.get("rating", "Unknown")

                            # Aplicamos las reglas de conversión --> ESTI EN 300 468 (PAG 97)
                            if raw_rating != "Unknown":
                                try:
                                    rating_int = int(raw_rating, 16)
                                    if rating_int == 0x1D:
                                        eit_data["parental_rating"] = f"{raw_rating} - Todos los públicos"
                                    elif 0x01 <= rating_int <= 0x0F:
                                        age = rating_int + 3
                                        eit_data["parental_rating"] = f"{raw_rating} - {age} años"
                                    elif 0x10 <= rating_int <= 0xFF:
                                        eit_data["parental_rating"] = f"{raw_rating} - Definido por el broadcaster"
                                    else:
                                        eit_data["parental_rating"] = f"{raw_rating} - No definido"
                                except ValueError:
                                    eit_data["parental_rating"] = raw_rating

                    elif child.tag == "extended_event_descriptor":
                        ext_descriptors.append(child)

                ext_descriptors.sort(key=lambda x: int(x.get("descriptor_number", 0)))
                for desc in ext_descriptors:
                    text_node = desc.find("text")
                    if text_node is not None and text_node.text:
                        extended_texts.append(text_node.text)

                if extended_texts:
                    eit_data["extended_text"] = "".join(extended_texts)

        except Exception as e:
            log.error(f"Error parseando XML {xml_path}: {e}")

        return eit_data

    def build_context(self, eit_xml_path: Optional[str], csv_metadata: Optional[dict] = None) -> str:
        eit_data = self.parse_eit_metadata(eit_xml_path)

        context_str = f"""
Transport Stream EIT Metadata:

- Título del Evento: {eit_data['event_name']}
- Hora de Inicio: {eit_data['start_time']}
- Duración: {eit_data['duration']}
- Estado de Emisión: {eit_data['running_status']}

Audiencia:
- País de Calificación: {eit_data['parental_country']}
- Calificación de Edad (Hex): {eit_data['parental_rating']}

Descripción Extendida de la Emisión:
{eit_data['extended_text']}
""".strip()

        return context_str

    def build_system_prompt(self) -> str:
        categories_str = ", ".join(CATEGORIES)

        return f"""
You are an expert DVB broadcast visual analyst.

Your task is to classify a TV broadcast event using BOTH:
1. the visual content of the frame, and
2. the DVB EIT metadata.

You must classify the content into exactly one of these allowed categories:

{categories_str}

Category definitions:
- Fiction: Fictional narrative content such as movies, TV series, scripted drama, comedy, romance, thriller, or historical drama.
- News: Programs reporting or discussing real-world events, including news broadcasts, interviews, debates, or news documentaries.
- Show: Entertainment programs such as quizzes, contests, variety shows, or talk shows.
- Sports: Sports events, competitions, live matches, highlights, or sports analysis.
- Cartoons: Content designed for children or teenagers, such as cartoons, educational shows, or youth entertainment.
- Music/Dance: Music performances, concerts, opera, ballet, dance, or music shows.
- Arts/Culture: Theatre, literature, cinema, visual arts, fashion, media culture, or cultural programs.
- Social: Social, political, or economic topics, including documentaries, reports, interviews, or debates.
- Education/Science: Science, nature, technology, medicine, education, or factual documentaries.
- Leisure hobbies: Travel, cooking, fitness, gardening, DIY, shopping, motoring, or lifestyle programs.

Instructions:
1. Use the image and EIT metadata simultaneously.
2. Return ONLY a strict valid JSON object.
3. Do not wrap the JSON in markdown.
4. Do not include text outside the JSON.
5. The field "predicted_category" must be exactly one of the allowed categories.
6. Do not invent new categories.
7. The reasoning must be short.

Required JSON schema:
{{
  "reasoning": "Brief explanation without line breaks",
  "confidence": 0,
  "predicted_category": "One of the allowed categories"
}}
""".strip()

    def _generate_for_frame(self, image_path: str, context_str: str) -> Dict[str, Any]:
        target_image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt()
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"""
    Context Data:
    {context_str}

    Analyze the image and the DVB EIT metadata together.
    Return the JSON classification.
    """.strip()
                    }
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=prompt,
            images=target_image,
            return_tensors="pt"
        ).to("cuda")

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with ResourceMonitor(sample_interval=0.05) as monitor:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False
                )

        generated_tokens = outputs[0][input_len:]

        decoded_text = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        metrics = monitor.finalize()

        result = {
            "raw_output": decoded_text,
            "tokens_in": int(input_len),
            "tokens_out": int(generated_tokens.shape[-1]),
        }

        result.update(metrics)

        return result

    def classify(
        self,
        image_paths: List[str],
        eit_xml_path: str = "",
        csv_metadata: Optional[dict] = None,
        sample_name: str = "UNKNOWN"
    ):

        context_str = self.build_context(eit_xml_path, csv_metadata=csv_metadata)

        log.info("\n" + "=" * 70)
        log.info("CONTEXTO EIT ENVIADO AL MODELO")
        log.info("=" * 70)
        log.info(context_str)
        log.info("=" * 70 + "\n")

        votes = []
        frame_predictions = []

        log.info(f"[{sample_name}] Iniciando clasificación multimodal de {len(image_paths)} frames.")

        for i, img_path in enumerate(image_paths):
            try:
                measured = self._generate_for_frame(img_path, context_str)
                json_data = parse_llm_json(measured["raw_output"])

                category = json_data.get("predicted_category", "Error")
                confidence = json_data.get("confidence", 0)
                reasoning = json_data.get("reasoning", "")

                if category not in CATEGORIES:
                    category = "Undefined"

                if category != "Undefined":
                    votes.append(category)

                frame_record = {
                    "frame_index": i + 1,
                    "frame_path": img_path,
                    "prediction": category,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "raw_output": measured["raw_output"],
                    "tokens_in": measured["tokens_in"],
                    "tokens_out": measured["tokens_out"],
                    "mode": "image_plus_eit",

                    "latency_total_ms": measured.get("latency_total_ms"),
                    "latency_total_s": measured.get("latency_total_s"),

                    "gpu_energy_j": measured.get("gpu_energy_j"),
                    "cpu_energy_j": measured.get("cpu_energy_j"),
                    "total_energy_j": measured.get("total_energy_j"),

                    "gpu_power_avg_w": measured.get("gpu_power_avg_w"),
                    "cpu_power_avg_w": measured.get("cpu_power_avg_w"),
                    "gpu_power_max_w": measured.get("gpu_power_max_w"),

                    "gpu_util_avg_pct": measured.get("gpu_util_avg_pct"),
                    "cpu_process_avg_pct": measured.get("cpu_process_avg_pct"),

                    "gpu_mem_avg_mb": measured.get("gpu_mem_avg_mb"),
                    "gpu_mem_max_mb": measured.get("gpu_mem_max_mb"),

                    "ram_process_avg_mb": measured.get("ram_process_avg_mb"),
                }
                latency_log.info(
                    f"{sample_name},frame={i + 1},prediction={category},"
                    f"latency_s={measured.get('latency_total_s')},"
                    f"tokens_in={measured.get('tokens_in')},"
                    f"tokens_out={measured.get('tokens_out')},"
                    f"gpu_energy_j={measured.get('gpu_energy_j')},"
                    f"total_energy_j={measured.get('total_energy_j')}"
                )

                frame_predictions.append(frame_record)

                log.info(
                    f"[{sample_name}] Frame {i + 1}/{len(image_paths)} -> "
                    f"{category} | Confianza: {confidence}% | "
                    f"Latencia: {measured.get('latency_total_s')}s | "
                    f"Energía Total: {measured.get('total_energy_j')}J"
                )

            except Exception as e:
                log.error(f"[ERROR] Error procesando frame {img_path}: {e}")
                continue

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not frame_predictions:
            log.info("[WARNING] No se pudieron obtener predicciones válidas.")
            return "Undefined", frame_predictions

        if votes:
            winner, count = Counter(votes).most_common(1)[0]
        else:
            winner = "Undefined"
            count = 0

        log.info(
            f"\n[{sample_name}] CATEGORÍA FINAL DEL EVENTO: "
            f"{winner} ({count}/{len(votes)} votos válidos)"
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
            for p in frame_predictions
        ]

        aggregate_metrics = aggregate_metric_dicts(metric_list)

        log.info(
            f"[{sample_name}] MÉTRICAS AGREGADAS | "
            f"latency_total_s={aggregate_metrics.get('latency_total_s')} | "
            f"tokens_per_second={aggregate_metrics.get('tokens_per_second')} | "
            f"total_energy_j={aggregate_metrics.get('total_energy_j')} | "
            f"gpu_mem_max_mb={aggregate_metrics.get('gpu_mem_max_mb')}"
        )
        return winner, frame_predictions