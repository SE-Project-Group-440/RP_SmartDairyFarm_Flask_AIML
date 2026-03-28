FROM python:3.11.9-slim-bookworm


WORKDIR /app

# prevent python cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install dependencies first (cache layer)
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# copy project files
COPY . .

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]