import logging

import ray
from hydra_plugins.hydra_ray_launcher import _launcher_util

log = logging.getLogger(__name__)


def _start_ray_without_memory(init_cfg):
    if not ray.is_initialized():
        try:
            # Connect directly to GCS instead of Ray client
            ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
            log.info("Successfully connected to Ray cluster")
        except Exception as e:
            log.error(f"Failed to connect to Ray: {str(e)}")
            raise


def patch_ray_launcher():
    _launcher_util.start_ray = _start_ray_without_memory
