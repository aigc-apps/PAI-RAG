# gunicorn.conf.py
bind = "0.0.0.0:8680"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 600
