# TTS Pool API Notes

This service follows the broad `llm-pool` shape, but uses a TTS-native response
contract.

## Public Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/responses` | Synthesize text to audio through any loaded TTS model. |
| `GET /v1/models` | List loaded public model ids. |
| `GET /v1/admin/models` | List configured model ids plus live runtime state and capabilities. |
| `GET /v1/admin/gpu-memory` | Return current GPU memory usage and model artifact estimates. |
| `POST /v1/admin/models/{model}/load` | Load one configured model at runtime. |
| `POST /v1/admin/models/{model}/unload` | Unload one loaded model at runtime. |

## Response Request

```json
{
  "model": "voxcpm2",
  "input": "Let's see if this works.",
  "language": "English",
  "fairness_key": "opaque-caller-key",
  "voice": {
    "instructions": "Speak in English. Use natural intonation.",
    "reference_audio": {
      "mime_type": "audio/wav",
      "data_base64": "...",
      "max_duration_s": 4
    }
  },
  "format": {
    "type": "wav"
  },
  "generation": {
    "voxcpm2": {
      "cfg_value": 2.0,
      "inference_timesteps": 10,
      "normalize": false,
      "denoise": false
    }
  },
  "stream": false
}
```

`stream: true` is intentionally rejected in the first version. The response
contains base64 WAV audio so callers can stay stateless and remote-friendly.

`fairness_key` is an optional stable scheduling identity. It is trimmed and may
contain at most 128 characters. Requests without a key share one anonymous
queue. Trusted callers should reuse one opaque key for the same principal; the
selected TTS backend never receives it.

Reference-audio base64 text may contain at most 16,777,216 characters. Larger
values fail request validation before entering a model queue.

Pending queues are bounded per key and per model executor. Overload returns
HTTP `429` with one of these detail codes:

| Code | Meaning |
| --- | --- |
| `fairness_key_queue_full` | The pending queue for this scheduling identity is full. |
| `executor_queue_full` | The model executor's total pending queue is full. |

`GET /v1/admin/models` reports aggregate queue and active counts. Its
`fairness` object shows active or pending keys, their configured weight, current
score, and queue-rejection counters.

For VoxCPM-family backends, clients own prompt wording and send the final text
control through `voice.instructions`. `voice.preset` is only meaningful for
backends with native voice ids, such as Kokoro.
