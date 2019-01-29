FROM python:3
ADD tests/main.py /
CMD [ "python", "./main.py" ]