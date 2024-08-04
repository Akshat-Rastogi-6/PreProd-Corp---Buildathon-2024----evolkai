FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8503

CMD ["streamlit", "run", "app.py", "--server.enableXsrfProtection", "false", "--server.port=8503", "--server.address=0.0.0.0"]

