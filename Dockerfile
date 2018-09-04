FROM python:3.6.6
COPY . /aequitas
WORKDIR /aequitas
RUN python setup.py install
ENTRYPOINT ["python"]
CMD ["serve.py"]
