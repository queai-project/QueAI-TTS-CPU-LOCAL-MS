# QueAI-TTS-CPU-LOCAL-MS

`QueAI-TTS-CPU-LOCAL-MS` es un módulo descargable para [QueAI](https://queai.dev/) que permite convertir texto y documentos en audio de forma local, ejecutando síntesis de voz en CPU y gestionando voces descargables bajo demanda.

Está pensado para integrarse como plugin dentro del ecosistema de QueAI, manteniendo una arquitectura desacoplada, portable y fácil de desplegar con Docker.

---

## Características

- Síntesis de voz local en CPU.
- Integración nativa con QueAI como módulo instalable.
- Selección de idioma y voz desde interfaz web.
- Descarga progresiva de voces Piper bajo demanda.
- Soporte para muestras de voz antes de descargar el modelo completo.
- Estados visuales para voces disponibles y no disponibles.
- Parámetros configurables de síntesis por voz.
- Conversión de:
  - texto libre
  - archivos `.txt`
  - archivos `.md`
  - archivos `.pdf`
- Sistema de tareas con cola de procesamiento.
- Reproducción y descarga del audio generado.
- Interfaz moderna y pensada para uso no técnico.

---

## Casos de uso

Este módulo puede utilizarse para:

- lectura en voz alta de textos
- narración de documentos
- pruebas de voces TTS locales
- generación de audios para asistentes, prototipos o automatizaciones
- entornos offline o privados donde no se desea depender de APIs externas

---

## Tecnologías principales

- **FastAPI**
- **Piper TTS**
- **ONNX Runtime**
- **Hugging Face Hub**
- **HTML/CSS/JavaScript**
- **Docker**

---

## Arquitectura general

Este módulo sigue el modelo de plugins de QueAI:

- expone endpoints HTTP propios
- se monta detrás del kernel de QueAI
- sirve una interfaz web propia
- gestiona sus propios recursos locales (`voices/`, runtime de jobs, caché, etc.)

Flujo general:

1. El usuario entra a la UI del módulo.
2. Selecciona un idioma.
3. Si el idioma aún no tiene catálogo local, lo descarga.
4. Selecciona una voz.
5. Si el modelo de esa voz no está descargado, puede descargarlo.
6. Envía texto o documento.
7. El módulo crea una tarea y genera el audio localmente.
8. El usuario puede reproducir o descargar el resultado.

---

## Estructura del proyecto

```text
QueAI-TTS-CPU-LOCAL-MS/
├── app/
│   ├── api/
│   ├── core/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── frontend_dist/
│   └── index.html
├── voices/
│   ├── _script/
│   └── ...
├── requirements.txt
├── Dockerfile
└── README.md