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
  "voice": {
    "preset": "warm_female",
    "instructions": "Use natural intonation.",
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
