# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

FROM fastdotai/fastai:2021-02-11

LABEL maintainer="Raj Kadiyala"


WORKDIR /

# Installing numpy, pandas, scikit-learn, scipy
RUN pip3 install --no-cache --upgrade requests 

# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
    
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache --upgrade flask gunicorn

# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY code /opt/program
WORKDIR /opt/program
