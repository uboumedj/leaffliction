FROM python:3.10-slim-bullseye

COPY requirements.txt requirements.txt
COPY main.py main.py
COPY Distribution.py Distribution.py
COPY Augmentation.py Augmentation.py
COPY Transformation.py Transformation.py
COPY Apples/ Apples/
COPY utils/ utils/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt # could run it with correcteur

CMD ["bash"]