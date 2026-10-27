FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user app/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app/app.py app.py

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=7860", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", "--server.enableXsrfProtection=false", \
            "--server.maxUploadSize=500", \
            "--server.maxMessageSize=500"]
