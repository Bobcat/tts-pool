# systemd

This directory holds systemd user-service deploy files for `tts-pool`.

Current scope:
- a user unit
- a start script that supports `DEFAULT_PORT` with optional `service.port` override from `config/settings.json`
- a repo-local `.venv` at `~/projects/tts-pool/.venv`
- optional `TTS_POOL_VENV_DIR` override when a host needs a separate runtime venv

Expected layout on the target host:

```bash
~/projects/tts-pool
```

Install or refresh the user service:

```bash
mkdir -p ~/.config/systemd/user
ln -sf ~/projects/tts-pool/deploy/systemd/tts-pool.service ~/.config/systemd/user/tts-pool.service
systemctl --user daemon-reload
systemctl --user enable --now tts-pool.service
```

Useful commands:

```bash
systemctl --user status tts-pool.service
journalctl --user -u tts-pool.service -f
systemctl --user restart tts-pool.service
```

For NanoVLLM-VoxCPM deployments that use a separate venv:

```bash
systemctl --user edit tts-pool.service
```

Add:

```ini
[Service]
Environment=TTS_POOL_VENV_DIR=/home/gunnar/projects/tts-pool/.venv-nanovllm
```
