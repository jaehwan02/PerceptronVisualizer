FROM python:3.10.0-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py
COPY mlp.py /app/mlp.py
COPY model.pt /app/model.pt
COPY train.py /app/train.py
COPY static /app/static
COPY templates /app/templates

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0", "--port", "8000" ]