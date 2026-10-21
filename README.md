# MusicGen: generación de música con IA

Proyecto del curso **IA Generativa** (Ingeniería de Sistemas, Universidad Santo Tomás, Tunja).

Genera una pista instrumental a partir de una descripción en texto (*prompt*) usando **MusicGen** de Meta, a través de la librería `transformers` de Hugging Face. El modelo corre en CPU, así que no hace falta tarjeta gráfica.

## Cómo funciona

1. Se carga el modelo `facebook/musicgen-small` y su procesador desde Hugging Face.
2. El procesador convierte el *prompt* en tokens.
3. El modelo genera los tokens de audio (`max_new_tokens=800`, unos 15 segundos) con muestreo y `guidance_scale=3.0`.
4. El resultado se guarda como `musica.wav` a la frecuencia de muestreo del modelo (32 kHz).

Prompt de ejemplo que trae el script:

> A dark Latin trap instrumental, melancholic piano chords, heavy 808 bass, crisp hi-hat rolls, atmospheric pads, slow and moody tempo around 130 BPM, cinematic introspective vibe, Puerto Rican urban style.

El archivo `musica.wav` del repositorio es el resultado de ese prompt.

## Tecnologías

- Python 3.12
- [uv](https://docs.astral.sh/uv/) para el entorno y las dependencias
- PyTorch 2.9.1
- Transformers 4.57.6
- SciPy (para escribir el `.wav`)
- SentencePiece (tokenizador del modelo)

## Instalación y uso

Requisitos: Python 3.12 y uv instalados.

```bash
git clone https://github.com/Mans0089/musicgen-ia-gen.git
cd musicgen-ia-gen
uv sync
uv run src/musicgen/__init__.py
```

La primera ejecución descarga el modelo (unos 2 GB), así que tarda un poco. Al terminar aparece `musica.wav` en la carpeta del proyecto.

## Personalización

En `src/musicgen/__init__.py` puedes cambiar:

| Variable / parámetro | Para qué sirve |
|---|---|
| `PROMPT` | Descripción de la música (funciona mejor en inglés) |
| `MODEL_ID` | Tamaño del modelo: `facebook/musicgen-small`, `-medium` o `-large` |
| `DEVICE` | `"cpu"` o `"cuda"` si tienes GPU NVIDIA |
| `max_new_tokens` | Duración: unos 50 tokens por segundo (750 ≈ 15 s) |
| `guidance_scale` | Qué tanto se ciñe al prompt (más alto = más fiel, menos variado) |

## Problemas comunes

**`OSError: Could not find a suitable TLS CA certificate bundle`**: en Windows, alguna instalación (por ejemplo PostgreSQL) dejó una variable como `CURL_CA_BUNDLE` o `REQUESTS_CA_BUNDLE` apuntando a un certificado que no existe. Bórrala de las variables de entorno y vuelve a abrir la terminal.

## Créditos

- Modelo: [MusicGen](https://huggingface.co/facebook/musicgen-small) de Meta AI (licencia CC-BY-NC 4.0, solo uso no comercial).
- Autor: Manuel Santiago Martínez Sierra ([@Mans0089](https://github.com/Mans0089)).
