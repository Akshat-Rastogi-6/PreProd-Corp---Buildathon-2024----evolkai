FROM python

WORKDIR /myapp

COPY . .

RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["streamlit", "run", "streamlit-app.py", "--server.enableXsrfProtection false"]

