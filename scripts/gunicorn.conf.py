# gunicorn.conf.py
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 600
