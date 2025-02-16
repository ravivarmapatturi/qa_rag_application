FROM python:3.12.7-slim
RUN /usr/local/bin/python -m pip install --upgrade pip
WORKDIR /MINIPROJECT
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . .
# ENTRYPOINT ["streamlit", "run"]
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

