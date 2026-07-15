FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "streamlit_electricity_usage.py", "--server.port=8501", "--server.address=0.0.0.0"]

