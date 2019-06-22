FROM python:3
ADD atomic_sensor_simulation /atomic_sensor_simulation
ADD tests /tests
ADD requirements.txt /
RUN pip install -r ./requirements.txt
ENV PYTHONPATH $PYTHONPATH:/atomic_sensor_simulation
ENTRYPOINT [ "python", "./tests/test_main.py" ]
ENTRYPOINT [ "python", "./atomic_sensor_simulation/main.py", "run-atomic-sensor" ]
ENTRYPOINT [ "python", "./atomic_sensor_simulation/main.py", "run-position-speed" ]