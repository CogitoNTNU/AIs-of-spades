import logging

try:
    import wandb as _wandb
except ImportError:
    _log = logging.getLogger(__name__)

    class _NoOpWandb:
        def init(self, *args, **kwargs):
            _log.warning(
                "wandb is not installed; training metrics will not be uploaded. "
                "Install the optional training dependency group with `uv sync --group train` to enable it."
            )
            return None

        def log(self, *args, **kwargs):
            return None

    wandb = _NoOpWandb()
else:
    wandb = _wandb
