#! /usr/bin/bash
echo "executando serviço principal"
doccano webserver --port 8000 &
echo "executando loader de datasets"
doccano task &
echo "serviço disponível!"