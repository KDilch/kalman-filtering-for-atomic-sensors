#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logger import Logger


def main():
    # setup a logger
    logger = Logger('atomic_sensor_simulation', log_file_path='logs/atomic_sensor_simulation.log')
    logger.logger.info('Starting execution of Atomic Sensor Simulation.')
    return 0


if __name__ == "__main__":
    main()
