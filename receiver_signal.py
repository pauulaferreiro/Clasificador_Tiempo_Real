import argparse
import csv
import re
import socket       
import struct
import subprocess   
import xml.etree.ElementTree as ET
import os
import time

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

TS_PACKET_SIZE = 188
SYNC_BYTE = 0x47

PAT_PID = 0x0000
SDT_PID = 0x0011
EIT_PID = 0x0012

NIBBLE1_TO_CATEGORY = {
    "0": "Undefined", "1": "Fiction", "2": "News", "3": "Show",
    "4": "Sports", "5": "Cartoons", "6": "Music/Dance",
    "7": "Arts/Culture", "8": "Social", "9": "Education/Science",
    "10": "Leisure hobbies",
}

def nibble1_to_category(nibble1: str) -> str:
    return NIBBLE1_TO_CATEGORY.get(str(nibble1), "Unknown")

def safe_filename(name: str, max_len: int = 120) -> str:
    name = "".join(ch for ch in name if ch >= " " or ch in "\t")
    name = re.sub(r"[^\w\s\-.]", "_", name.strip() or "SIN_NOMBRE", flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name[:max_len] or "SIN_NOMBRE"

def normalize_event_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("\n", " ").replace("\r", " ")).strip()

def format_timedelta_hms(td: timedelta) -> str:
    total_seconds = max(0, int(td.total_seconds()))
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"

def bcd_to_int(b: int) -> int:
    return ((b >> 4) * 10) + (b & 0x0F)

def parse_dvb_duration_3bytes(data: bytes) -> timedelta:
    if len(data) != 3: return timedelta(0)
    return timedelta(hours=bcd_to_int(data[0]), minutes=bcd_to_int(data[1]), seconds=bcd_to_int(data[2]))

def mjd_to_ymd(mjd: int) -> Tuple[int, int, int]:
    y_dash = int((mjd - 15078.2) / 365.25)
    m_dash = int((mjd - 14956.1 - int(y_dash * 365.25)) / 30.6001)
    d = mjd - 14956 - int(y_dash * 365.25) - int(m_dash * 30.6001)
    k = 1 if m_dash in (14, 15) else 0
    return y_dash + k + 1900, m_dash - 1 - k * 12, d

def parse_dvb_start_time_5bytes(data: bytes) -> Optional[datetime]:
    if len(data) != 5 or all(b == 0xFF for b in data): return None
    mjd = (data[0] << 8) | data[1]
    y, m, d = mjd_to_ymd(mjd)
    try:
        return datetime(y, m, d, bcd_to_int(data[2]), bcd_to_int(data[3]), bcd_to_int(data[4]), tzinfo=timezone.utc)
    except ValueError:
        return None

def clean_dvb_text(raw: bytes) -> str:
    if not raw: return ""
    while raw and raw[0] < 0x20: raw = raw[1:]
    return raw.decode("latin-1", errors="ignore").strip()

def parse_record_ip(value: str) -> Tuple[str, int]:
    value = value.strip().replace("udp://", "")
    host, port = value.rsplit(":", 1)
    return host.strip(), int(port.strip())

def is_valid_event_name(name: str) -> bool:
    return bool(name) and not name.startswith("EVENT_")

@dataclass(frozen=True)
class RunningEvent:
    event_id: str
    name: str
    start_utc: datetime
    duration: timedelta
    running_status: str
    content_nibble_1: str
    content_nibble_2: str
    extended_text: str

@dataclass
class ServiceState:
    service_id: int
    service_name: str = ""
    pmt_pid: Optional[int] = None
    pcr_pid: Optional[int] = None
    component_pids: Set[int] = field(default_factory=set)
    output_pids: Set[int] = field(default_factory=set)
    current_event: Optional[RunningEvent] = None
    current_event_key: Optional[Tuple[str, datetime]] = None
    last_seen_eit_key: Optional[Tuple[str, datetime, str]] = None
    
    # Variables de control para FFmpeg y el temporizador
    ffmpeg_process: Optional[subprocess.Popen] = None
    ffmpeg_cmd: Optional[List[str]] = None
    event_detected_wallclock: Optional[float] = None
    
    fragment_index: int = 0

class SectionAssembler:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def push_payload(self, payload: bytes, pusi: bool) -> List[bytes]:
        sections = []
        if pusi:
            if not payload: return sections
            pointer_field = payload[0] 
            if len(payload) < 1 + pointer_field:
                self.buffer.clear(); return sections
            if self.buffer and pointer_field > 0:
                self.buffer.extend(payload[1:1 + pointer_field])
                sections.extend(self._extract_sections())
            self.buffer = bytearray(payload[1 + pointer_field:])
            sections.extend(self._extract_sections())
        else:
            if payload:
                self.buffer.extend(payload)
                sections.extend(self._extract_sections())
        return sections

    def _extract_sections(self) -> List[bytes]:
        out = []
        while len(self.buffer) >= 3:
            table_id = self.buffer[0]
            if table_id == 0xFF: self.buffer.clear(); break
            section_length = ((self.buffer[1] & 0x0F) << 8) | self.buffer[2]
            total_len = 3 + section_length
            if section_length > 1021: self.buffer.clear(); break
            if len(self.buffer) < total_len: break
            sec = bytes(self.buffer[:total_len])
            del self.buffer[:total_len]
            out.append(sec)
        return out

def merge_extended_texts(parts: List[str]) -> str:
    cleaned = [normalize_event_name(p) for p in parts if p and normalize_event_name(p)]
    return " ".join(cleaned).strip()

def parse_ts_packet_header(pkt: bytes) -> Optional[dict]:
    if len(pkt) != TS_PACKET_SIZE or pkt[0] != SYNC_BYTE: return None
    pusi = bool(pkt[1] & 0x40)
    pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
    afc = (pkt[3] >> 4) & 0x03
    idx = 4
    if afc in (2, 3):
        if idx >= len(pkt): return None
        idx += 1 + pkt[idx]
    payload = pkt[idx:] if afc in (1, 3) and idx <= len(pkt) else b""
    return {"pid": pid, "pusi": pusi, "payload": payload, "afc": afc}

def parse_pat_section(section: bytes) -> Dict[int, int]:
    programs = {}
    if len(section) < 8 or section[0] != 0x00: return programs
    end = 3 + (((section[1] & 0x0F) << 8) | section[2]) - 4
    idx = 8
    while idx + 4 <= end:
        program_number = (section[idx] << 8) | section[idx + 1]
        pmt_pid = ((section[idx + 2] & 0x1F) << 8) | section[idx + 3]
        if program_number != 0: programs[program_number] = pmt_pid
        idx += 4
    return programs

def parse_pmt_section(section: bytes) -> Tuple[Optional[int], Set[int]]:
    if len(section) < 12 or section[0] != 0x02: return None, set()
    pcr_pid = ((section[8] & 0x1F) << 8) | section[9]
    idx = 12 + (((section[10] & 0x0F) << 8) | section[11])
    end = 3 + (((section[1] & 0x0F) << 8) | section[2]) - 4
    pids = set()
    while idx + 5 <= end:
        pids.add(((section[idx + 1] & 0x1F) << 8) | section[idx + 2])
        idx += 5 + (((section[idx + 3] & 0x0F) << 8) | section[idx + 4])
    return pcr_pid, pids

def parse_sdt_section(section: bytes) -> Dict[int, str]:
    names = {}
    if len(section) < 11 or section[0] != 0x42: return names
    end = 3 + (((section[1] & 0x0F) << 8) | section[2]) - 4
    idx = 11
    while idx + 5 <= end:
        service_id = (section[idx] << 8) | section[idx + 1]
        dpos, dend = idx + 5, idx + 5 + (((section[idx + 3] & 0x0F) << 8) | section[idx + 4])
        while dpos + 2 <= dend and dpos + 2 <= len(section):
            tag, length = section[dpos], section[dpos + 1]
            body = section[dpos + 2:dpos + 2 + length]
            if tag == 0x48 and len(body) >= 2 and 2 + body[1] < len(body):
                name_len_pos = 2 + body[1]
                name = clean_dvb_text(body[name_len_pos + 1:name_len_pos + 1 + body[name_len_pos]])
                if name: names[service_id] = name
            dpos += 2 + length
        idx = dend
    return names

def parse_eit_section(section: bytes) -> Tuple[int, List[RunningEvent]]:
    events = []
    if len(section) < 14 or section[0] != 0x4E: return 0, events
    service_id = (section[3] << 8) | section[4]
    end = 3 + (((section[1] & 0x0F) << 8) | section[2]) - 4
    idx = 14
    while idx + 12 <= end:
        event_id = (section[idx] << 8) | section[idx + 1]
        start_utc = parse_dvb_start_time_5bytes(section[idx + 2:idx + 7])
        duration = parse_dvb_duration_3bytes(section[idx + 7:idx + 10])
        running_status_int = (section[idx + 10] >> 5) & 0x07
        dpos, dend = idx + 12, idx + 12 + (((section[idx + 10] & 0x0F) << 8) | section[idx + 11])
        event_name, nib1, nib2, extended_parts = f"EVENT_{event_id:04X}", "0", "0", []
        
        while dpos + 2 <= dend and dpos + 2 <= len(section):
            tag, length = section[dpos], section[dpos + 1]
            body = section[dpos + 2:dpos + 2 + length]
            if tag == 0x4D and len(body) >= 5:
                decoded = clean_dvb_text(body[4:4 + body[3]])
                if decoded: event_name = normalize_event_name(decoded)
            elif tag == 0x54 and len(body) >= 2:
                nib1, nib2 = str((body[0] >> 4) & 0x0F), str(body[0] & 0x0F)
            elif tag == 0x4E and len(body) >= 6:
                items_end = 5 + body[4]
                if items_end < len(body):
                    decoded_text = clean_dvb_text(body[items_end + 1:items_end + 1 + body[items_end]])
                    if decoded_text: extended_parts.append(decoded_text)
            dpos += 2 + length

        if start_utc is not None:
            events.append(RunningEvent(
                event_id=f"0x{event_id:04X}", name=event_name, start_utc=start_utc, duration=duration,
                running_status=str(running_status_int), content_nibble_1=nib1, content_nibble_2=nib2,
                extended_text=merge_extended_texts(extended_parts),
            ))
        idx = dend
    return service_id, events

def init_live_csv(out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out_csv.exists():
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "SERVICIO", "RUTA DIRECTORIO FRAMES", "RUTA XML EIT",
                "NOMBRE DEL EVENTO", "HORA DE INICIO", "DURACION",
                "CONTENT_NIBBLE_1", "CONTENT_NIBBLE_2", "CONTENT_CATEGORY_AUTO"
            ])

def append_to_live_csv(out_csv: Path, service_name: str, frames_dir: str, xml_path: str, event: RunningEvent):
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            service_name, frames_dir, xml_path,
            event.name, event.start_utc.strftime("%H:%M:%SZ"),
            format_timedelta_hms(event.duration),
            event.content_nibble_1, event.content_nibble_2,
            nibble1_to_category(event.content_nibble_1)
        ])

def generate_eit_xml(event: RunningEvent, service_id: int, xml_out_path: Path):

    xml_out_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("EIT", service_id=f"0x{service_id:04X}")

    ev_node = ET.SubElement(
        root,
        "event",
        event_id=event.event_id,
        start_time=event.start_utc.isoformat(),
        duration=format_timedelta_hms(event.duration),
        running_status=event.running_status
    )

    short_desc = ET.SubElement(ev_node, "short_event_descriptor")
    ET.SubElement(short_desc, "event_name").text = event.name

    extended_desc = ET.SubElement(
        ev_node,
        "extended_event_descriptor",
        descriptor_number="0"
    )
    ET.SubElement(extended_desc, "text").text = event.extended_text or ""

    # Los dejo también por si en el futuro quieres trazabilidad,
    # pero el DAEMON no los usará para comparar.
    ET.SubElement(ev_node, "content_nibble_1").text = event.content_nibble_1
    ET.SubElement(ev_node, "content_nibble_2").text = event.content_nibble_2

    tree = ET.ElementTree(root)
    tree.write(xml_out_path, encoding="utf-8", xml_declaration=True)

def refresh_service_output_pids(service: ServiceState) -> None:
    service.output_pids = set(service.component_pids)
    if service.pmt_pid is not None: service.output_pids.add(service.pmt_pid)
    if service.pcr_pid is not None: service.output_pids.add(service.pcr_pid)
    service.output_pids.update([PAT_PID, SDT_PID, EIT_PID])

def close_event_processor(service: ServiceState) -> None:
    if service.ffmpeg_process:
        try:
            service.ffmpeg_process.stdin.close()
            service.ffmpeg_process.wait(timeout=2)
        except Exception:
            service.ffmpeg_process.kill()
        finally:
            service.ffmpeg_process = None
    
    # Reseteamos los temporizadores para el próximo evento
    service.ffmpeg_cmd = None
    service.event_detected_wallclock = None

def start_event_processor(service: ServiceState, event: RunningEvent, frames_base_dir: Path, eit_base_dir: Path, csv_path: Path, frame_mode: str, seconds: float, max_frames: int, margin_seconds: int) -> None:
    service.fragment_index += 1
    safe_srv = safe_filename(service.service_name or f"Servicio_{service.service_id:04X}")
    safe_ev = safe_filename(event.name)
    timestamp = event.start_utc.strftime('%H%M%S')

    event_frames_dir = frames_base_dir / safe_srv / f"{service.fragment_index:03d}_{safe_ev}_{timestamp}"
    event_frames_dir.mkdir(parents=True, exist_ok=True)
    
    event_eit_dir = eit_base_dir / safe_srv
    xml_out_path = event_eit_dir / f"{service.fragment_index:03d}_{safe_ev}_{timestamp}.xml"

    # Extraemos y guardamos la EIT 
    generate_eit_xml(event, service.service_id, xml_out_path)

    append_to_live_csv(csv_path, safe_srv, str(event_frames_dir), str(xml_out_path), event)

    cmd = ["ffmpeg", "-f", "mpegts", "-i", "pipe:0"]
    if frame_mode == "IFRAMES": cmd += ["-vf", "select='eq(pict_type,PICT_TYPE_I)'", "-vsync", "vfr"]
    elif frame_mode == "EVERY_N_SECONDS": cmd += ["-vf", f"fps={1.0/seconds}"]
    elif frame_mode == "IFRAMES_EVERY_N_SECONDS":
        filtro = f"select='eq(pict_type,PICT_TYPE_I)*(isnan(prev_selected_t)+gt(t,prev_selected_t+{seconds}))'"
        cmd += ["-vf", filtro, "-vsync", "vfr"]
    
    out_pattern = event_frames_dir / "frame_%05d.jpg"
    cmd += ["-q:v", "2", "-frames:v", str(max_frames), "-y", str(out_pattern)]

    service.ffmpeg_cmd = cmd
    
    # Registramos el momento exacto en el que detectamos el evento
    service.event_detected_wallclock = time.time()
    print(f"[EIT] 0x{service.service_id:04X} -> {event.name} | XML generado. Extracción en {margin_seconds} segundos...")

def process_live_mux(args) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_base_dir = output_dir / "frames_seleccionados"
    eit_base_dir = output_dir / "eit_extraidas"
    csv_path = output_dir / "dataset_tiempo_real.csv"

    host, port = parse_record_ip(args.record_ip)
    init_live_csv(csv_path)

    assemblers: Dict[int, SectionAssembler] = {PAT_PID: SectionAssembler(), SDT_PID: SectionAssembler(), EIT_PID: SectionAssembler()}
    services: Dict[int, ServiceState] = {}
    pmt_pid_to_service: Dict[int, int] = {}

    start_wallclock_utc = datetime.now(timezone.utc)
    deadline = start_wallclock_utc + timedelta(seconds=args.record_seconds)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: sock.bind(("", port))
    except OSError: sock.bind((host, port))
    if 224 <= int(host.split(".")[0]) <= 239:
        mreq = struct.pack("=4sl", socket.inet_aton(host), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(2.0)

    print(f"Iniciando ingesta... (Margen de seguridad: {args.margin_seconds}s)")
    
    try:
        while datetime.now(timezone.utc) < deadline:
            try: datagram, _addr = sock.recvfrom(7 * TS_PACKET_SIZE)
            except socket.timeout: continue
            if not datagram: continue
            usable = len(datagram) - (len(datagram) % TS_PACKET_SIZE)

            #### COMPROBACIÓN DEL TEMPORIZADOR PARA CADA SERVICIO ####
            current_time = time.time()
            for service in services.values():
                # Si estamos esperando (hay tiempo registrado y no hay proceso ffmpeg)
                if service.event_detected_wallclock is not None and service.ffmpeg_process is None:
                    if current_time - service.event_detected_wallclock >= args.margin_seconds:
                        # Lanzar FFmpeg ahora
                        service.ffmpeg_process = subprocess.Popen(
                            service.ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        print(f"[FFMPEG] {service.service_name} -> Extracción de frames INICIADA (Margen superado)")

            for i in range(0, usable, TS_PACKET_SIZE):
                pkt = datagram[i:i + TS_PACKET_SIZE]
                hdr = parse_ts_packet_header(pkt)
                if not hdr: continue
                pid, pusi, payload = hdr["pid"], hdr["pusi"], hdr["payload"]

                if pid == PAT_PID:
                    for sec in assemblers[PAT_PID].push_payload(payload, pusi):
                        for sid, pmt_pid in parse_pat_section(sec).items():
                            if sid not in services: services[sid] = ServiceState(service_id=sid, pmt_pid=pmt_pid)
                            else: services[sid].pmt_pid = pmt_pid
                            pmt_pid_to_service[pmt_pid] = sid
                            if pmt_pid not in assemblers: assemblers[pmt_pid] = SectionAssembler()

                elif pid == SDT_PID:
                    for sec in assemblers[SDT_PID].push_payload(payload, pusi):
                        for sid, name in parse_sdt_section(sec).items():
                            if sid not in services:
                                services[sid] = ServiceState(
                                    service_id=sid,
                                    service_name=name
                                )
                            else:
                                services[sid].service_name = name

                elif pid == EIT_PID:
                    for sec in assemblers[EIT_PID].push_payload(payload, pusi):
                        sid, eit_events = parse_eit_section(sec)
                        if sid == 0 or not eit_events or sid not in services: continue
                        
                        service = services[sid]
                        running_events = [e for e in eit_events if e.running_status == "4" and is_valid_event_name(e.name)]
                        if not running_events: continue
                        
                        selected = running_events[0]
                        eit_key = (selected.event_id, selected.start_utc, selected.running_status)

                        if service.last_seen_eit_key != eit_key:
                            service.last_seen_eit_key = eit_key
                            close_event_processor(service)
                            start_event_processor(service, selected, frames_base_dir, eit_base_dir, csv_path, args.frame_mode, args.seconds, args.max_frames, args.margin_seconds)

                elif pid in pmt_pid_to_service:
                    sid = pmt_pid_to_service[pid]
                    for sec in assemblers[pid].push_payload(payload, pusi):
                        pcr_pid, comp_pids = parse_pmt_section(sec)
                        services[sid].pcr_pid = pcr_pid
                        services[sid].component_pids = comp_pids
                        refresh_service_output_pids(services[sid])

                for service in services.values():
                    if service.ffmpeg_process and service.ffmpeg_process.poll() is None:
                        if pid in service.output_pids:
                            try: service.ffmpeg_process.stdin.write(pkt)
                            except BrokenPipeError: close_event_processor(service)

    finally:
        sock.close()
        for service in services.values(): close_event_processor(service)
            
    print("\nProcesado finalizado.")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-ip", required=True)
    parser.add_argument("--record-seconds", required=True, type=int)
    parser.add_argument("--output-dir", default="./RESULTADOS_MUX")
    parser.add_argument("--frame-mode", choices=["IFRAMES", "EVERY_N_SECONDS", "IFRAMES_EVERY_N_SECONDS"], default="IFRAMES_EVERY_N_SECONDS")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--margin-seconds", type=int, default=10)
    args = parser.parse_args()
    return process_live_mux(args)

if __name__ == "__main__":
    raise SystemExit(main())
