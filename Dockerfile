FROM python:3
ADD tests/main.py /
ENTRYPOINT [ "python", "./main.py" ]