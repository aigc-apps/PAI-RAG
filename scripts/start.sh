#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -w workers -p port"
   echo -e "\t-w number of workers, default 1"
   echo -e "\t-p port, default 8680"
   exit 1 # Exit script after printing help
}

while getopts ":w:p:" opt
do
   case "$opt" in
      w ) workers="$OPTARG" ;;
      p ) port="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

workers="${workers:-1}"
port="${port:-8680}"

echo "Starting gunicorn with $workers workers on port $port..."

gunicorn -w $workers -b "0.0.0.0:${port}" -c scripts/gunicorn.conf.py src.pai_rag.app.app:app --timeout 600
