from atomic_sensor_simulation.main import run__atomic_sensor, run_position_speed
from logging import getLogger

logger = getLogger(__name__)


def run_smoke_tests():
    logger.info("Starting unit tests...")
    run__atomic_sensor()
    logger.info("Starting smoke tests...")
    run_position_speed()
    logger.info("Finished tests.")
