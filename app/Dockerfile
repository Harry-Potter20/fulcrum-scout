# Fulcrum Scout — persistent container. Build context = Football_Research/.
# Full fulcrum package + torch, because the live counterfactual (app/services/counterfactual_service.py) calls
# fulcrum.state_summary/find_holes via the package root and rolls the twin forward with torch. The fulcrum
# package itself is pure numpy (verified: no cv2/ultralytics anywhere outside products/); torch is pinned to the
# CPU wheel index to avoid pulling a multi-GB CUDA build for a container with no GPU.
#
# Two services share this image — override CMD for the second one:
#   docker build -f app/Dockerfile -t fulcrum-scout .
#   docker run -p 8501:8501 -e HF_TOKEN=hf_xxx -e DATABASE_URL=postgres://... -e FULCRUM_APP_PASSWORD=yourkey fulcrum-scout
#   docker run -p 8080:8080 -e HF_TOKEN=hf_xxx -e DATABASE_URL=postgres://... -e FULCRUM_SYNC_KEY=yourkey \
#     fulcrum-scout python app/ingest_server.py
FROM python:3.11-slim

WORKDIR /srv
COPY app/requirements.txt /srv/app/requirements.txt
RUN pip install --no-cache-dir -r /srv/app/requirements.txt

COPY app/ /srv/app/
COPY fulcrum/ /srv/fulcrum/
COPY jobs/worldmodel.py /srv/jobs/worldmodel.py
COPY jobs/sync_db.py /srv/jobs/sync_db.py

ENV PYTHONUNBUFFERED=1 STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHERUSAGESTATS=false
EXPOSE 8501
# most hosts inject $PORT; default 8501 for local docker run. Override CMD (see above) for the ingest service.
CMD streamlit run app/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 \
    --server.enableCORS false --server.enableXsrfProtection false
