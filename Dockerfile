FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data server && touch data/__init__.py && touch server/__init__.py
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["python", "server.py"]
