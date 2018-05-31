FROM python:3.5-alpine
RUN apk update && apk upgrade && \
apk add --no-cache bash git openssh
RUN apk --update add --no-cache \ 
    lapack-dev \ 
    gcc \
    freetype-dev
COPY . /aequitas
WORKDIR /aequitas
RUN pip3 install --no-cache-dir --no-build-isolation git+https://github.com/dssg/aequitas.git
ENTRYPOINT ["python"]
CMD ["serve.py"]
