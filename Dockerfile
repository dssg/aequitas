FROM continuumio/miniconda3:4.4.10
COPY . /aequitas
WORKDIR /aequitas
RUN python setup.py install
ENTRYPOINT ["python"]
CMD ["serve.py"]
