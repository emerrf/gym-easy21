FROM python:2.7

RUN git clone https://github.com/emerrf/gym-easy21.git /app/gym-easy21
WORKDIR /app/gym-easy21

RUN pip install -r /app/gym-easy21/requirements.txt

ENTRYPOINT ["gym-easy21"]