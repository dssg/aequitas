FROM python:3.9
COPY . /aequitas
WORKDIR /aequitas
RUN python setup.py install
ENTRYPOINT ["python"]
CMD ["serve.py"]
