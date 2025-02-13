#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -w workers"
   echo -e "\t-w number of workers, default 1"
   exit 1 # Exit script after printing help
}

while getopts "w:" opt
do
   case "$opt" in
      w ) workers="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

workers="${workers:-1}"

echo "Starting gunicorn with $workers workers..."

gunicorn -w $workers -c scripts/gunicorn.conf.py src.pai_rag.app.app:app
