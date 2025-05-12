FROM hub-dev.hexin.cn/jupyterhub/script_crontab:beimeipri_py38
USER root
COPY ./ /root/port_service
WORKDIR /root/port_service
RUN pip install -r requirements.txt && cat ./hosts >> /etc/hosts
ENTRYPOINT [ "uvicorn", "infer:app", "--reload", "--port=12333", "--host=0.0.0.0"]