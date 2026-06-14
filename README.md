# Clasificador de Contenidos en Tiempo Real

Este proyecto ejecuta un sistema en tiempo real compuesto por dos procesos principales:

1. Un **controlador/clasificador** que monitoriza los frames y metadatos generados.
2. Un **receptor de señal** que recibe la señal MPEG-TS por UDP, extrae la información EIT y genera frames de vídeo.

La ejecución se realiza en **dos terminales independientes**.

---

## Estructura de ejecución

```text
Terminal 1:
python controller.py

Terminal 2:
python receiver_signal.py ...
```

---

## Scripts utilizados

### Terminal 1: controlador y clasificador

Al ejecutar:

```bash
python controller.py
```

se utiliza directamente el script:

```text
controller.py
```

Este script, además, importa y utiliza los siguientes módulos auxiliares:

```text
logger_config.py
metrics_monitor.py
pipeline.py
```

Por tanto, la ejecución del controlador depende de:

```text
controller.py
logger_config.py
metrics_monitor.py
pipeline.py
```

#### Función de cada script

| Script | Función |
|---|---|
| `controller.py` | Ejecutor principal de clasificación. Monitoriza las carpetas de frames y XML EIT, filtra frames, llama al modelo y guarda resultados. |
| `logger_config.py` | Configura los logs del sistema y registra latencias de ejecución. |
| `metrics_monitor.py` | Calcula y agrega métricas de rendimiento como latencia, consumo energético, uso de GPU/CPU, memoria y tokens. |
| `pipeline.py` | Contiene la clase `VideoClassifier`, encargada de cargar el modelo multimodal y clasificar los frames usando imagen + metadatos EIT. |

---

### Terminal 2: receptor de señal

Al ejecutar:

```bash
python receiver_signal.py ...
```

se utiliza directamente:

```text
receiver_signal.py
```

Por tanto, la ejecución del receptor depende de:

```text
receiver_signal.py
```

#### Función de `receiver_signal.py`

El script `receiver_signal.py` se encarga de:

- Recibir una señal MPEG-TS por UDP.
- Parsear las tablas DVB:
  - PAT
  - PMT
  - SDT
  - EIT
- Detectar eventos en emisión.
- Generar XML con la información EIT del evento.
- Crear carpetas de salida por servicio y evento.
- Lanzar FFmpeg para extraer frames del vídeo.
- Guardar un CSV con información de los eventos detectados.

---

## Requisitos previos

Antes de ejecutar el sistema, asegúrate de tener instalados:

- Python 3.10 o superior.
- FFmpeg.
- PyTorch con soporte CUDA, si se va a usar GPU.
- Las librerías de Python necesarias para los scripts.

Instalación recomendada de dependencias:

```bash
pip install torch torchvision transformers pillow opencv-python scikit-image psutil
```

Si se van a medir métricas de GPU y CPU:

```bash
pip install nvidia-ml-py pyRAPL
```
En entornos Intel, es necesario habilitar los permisos de lectura del sistema mediante el comando: sudo chmod -R a+r /sys/class/powercap/intel-rapl

---

## Flujo general del sistema

El sistema funciona de la siguiente manera:

```text
Señal UDP/MPEG-TS
        |
        v
receiver_signal.py
        |
        |-- Extrae XML EIT
        |-- Extrae frames con FFmpeg
        v
RESULTADOS_MUX/
        |
        |-- frames_seleccionados/
        |-- eit_extraidas/
        |-- dataset_tiempo_real.csv
        |
        v
controller.py
        |
        |-- Lee frames nuevos
        |-- Lee XML EIT
        |-- Filtra frames por calidad y similitud
        |-- Clasifica con VideoClassifier
        |-- Guarda JSON y CSV final
        v
resultados_frames/
RESULTADOS_MUX/reporte_final_predicciones.csv
```

---

## Ejecución paso a paso

### 1. Abrir el primer terminal

En el primer terminal, ejecutar el controlador:

```bash
python controller.py
```

Este proceso quedará activo esperando a que aparezcan frames y XML generados por el receptor.

Por defecto, `controller.py` busca los datos en:

```text
./RESULTADOS_MUX/frames_seleccionados
./RESULTADOS_MUX/eit_extraidas
```

y guarda resultados en:

```text
./resultados_frames
./RESULTADOS_MUX/reporte_final_predicciones.csv
```

---

### 2. Abrir el segundo terminal

En el segundo terminal, ejecutar el receptor de señal:

```bash
python receiver_signal.py --record-ip udp://IP:PUERTO --record-seconds DURACION
```

Ejemplo con una señal multicast:

```bash
python receiver_signal.py --record-ip udp://239.0.0.1:1234 --record-seconds 3600
```

Ejemplo con una señal local:

```bash
python receiver_signal.py --record-ip udp://127.0.0.1:1234 --record-seconds 3600
```

---

## Parámetros principales de `controller.py`

`controller.py` puede ejecutarse con sus valores por defecto:

```bash
python controller.py
```

También permite configurar rutas y parámetros de filtrado:

```bash
python controller.py \
  --base-frames-dir ./RESULTADOS_MUX/frames_seleccionados \
  --base-eit-dir ./RESULTADOS_MUX/eit_extraidas \
  --json-out-dir ./resultados_frames \
  --final-csv ./RESULTADOS_MUX/reporte_final_predicciones.csv \
  --poll-interval 5 \
  --ssim-threshold 0.6 \
  --laplacian-min 70.0 \
  --laplacian-max 1500.0
```

### Parámetros disponibles

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `--base-frames-dir` | Carpeta raíz donde se buscan los frames generados. | `./RESULTADOS_MUX/frames_seleccionados` |
| `--base-eit-dir` | Carpeta raíz donde se buscan los XML EIT. | `./RESULTADOS_MUX/eit_extraidas` |
| `--json-out-dir` | Carpeta donde se guardan los JSON de predicción. | `./resultados_frames` |
| `--final-csv` | CSV final con la predicción global por evento. | `./RESULTADOS_MUX/reporte_final_predicciones.csv` |
| `--poll-interval` | Tiempo, en segundos, entre escaneos de carpetas. | `5` |
| `--ssim-threshold` | Umbral SSIM para descartar frames demasiado similares. | `0.6` |
| `--laplacian-min` | Umbral mínimo de nitidez. Frames por debajo se descartan por borrosos. | `70.0` |
| `--laplacian-max` | Umbral máximo de Laplaciano. Frames por encima se descartan por ruido o artefactos. | `1500.0` |

---

## Parámetros principales de `receiver_signal.py`

La ejecución básica es:

```bash
python receiver_signal.py --record-ip udp://IP:PUERTO --record-seconds DURACION
```

Ejemplo:

```bash
python receiver_signal.py \
  --record-ip udp://239.0.0.1:1234 \
  --record-seconds 3600
```

### Parámetros habituales

| Parámetro | Descripción |
|---|---|
| `--record-ip` | Dirección UDP de entrada en formato `udp://IP:PUERTO`. |
| `--record-seconds` | Duración total de la captura en segundos. |
| `--output-dir` | Carpeta raíz donde se guardan frames, XML y CSV. |
| `--frame-mode` | Modo de extracción de frames. |
| `--seconds` | Intervalo entre frames cuando se usa extracción temporal. |
| `--max-frames` | Número máximo de frames a extraer por evento. |
| `--margin-seconds` | Margen de espera antes de comenzar la extracción de frames tras detectar un evento. |

---

## Carpetas de salida

Durante la ejecución, `receiver_signal.py` genera una estructura como la siguiente:

```text
RESULTADOS_MUX/
├── frames_seleccionados/
│   └── SERVICIO/
│       └── EVENTO/
│           ├── frame_00001.jpg
│           ├── frame_00002.jpg
│           └── ...
│
├── eit_extraidas/
│   └── SERVICIO/
│       └── EVENTO.xml
│
└── dataset_tiempo_real.csv
```

Posteriormente, `controller.py` genera:

```text
resultados_frames/
└── SERVICIO/
    └── EVENTO.json

RESULTADOS_MUX/
└── reporte_final_predicciones.csv
```

---

## Orden correcto de ejecución

El orden recomendado es:

### Terminal 1

```bash
python controller.py
```

### Terminal 2

```bash
python receiver_signal.py --record-ip udp://239.0.0.1:1234 --record-seconds 3600
```

El motivo de este orden es que `controller.py` queda esperando a que aparezcan nuevas carpetas de frames y XML. Cuando `receiver_signal.py` detecta un evento y genera los datos, el controlador los procesa automáticamente.

---

## Resultado esperado

Durante la ejecución se obtienen:

1. Frames extraídos de la emisión.
2. XML EIT por evento.
3. JSON con predicciones por frame.
4. CSV final con la predicción global del evento.
5. Logs de ejecución y latencia.

---

## Logs

Los logs se guardan en la carpeta:

```text
logs/
```

Archivos principales:

```text
video_classifier.log
latency.log
```

---

## Resumen de uso rápido

```bash
# Terminal 1
python controller.py
```

```bash
# Terminal 2
python receiver_signal.py --record-ip udp://239.0.0.1:1234 --record-seconds 3600
```

Scripts usados:

```text
Terminal 1:
- controller.py
- logger_config.py
- metrics_monitor.py
- PRUEBA.py

Terminal 2:
- receiver_signal.py
```
