FROM python:3.13.12-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app
COPY ./frontend_dist /code/frontend_dist
COPY ./voices.tar.gz /code/voices.tar.gz

RUN mkdir -p /code/voices \
    && tar -xzf /code/voices.tar.gz -C /code \
    && rm /code/voices.tar.gz

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]