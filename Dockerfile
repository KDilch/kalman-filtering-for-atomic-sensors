FROM python:3
ADD atomic_sensor_simulation /atomic_sensor_simulation
ADD tests /tests
ENTRYPOINT [ "python", "./tests/test_main.py" ]