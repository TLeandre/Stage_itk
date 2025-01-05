FROM python:3.12.1

WORKDIR /api-prediction-prairie

COPY requirements.txt .
#COPY ee-agroitk-a58b9b9b9b5b.json .
COPY ./scripts ./scripts
COPY ./models ./models

RUN pip install -r requirements.txt
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y

# Downloading gcloud package
#RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
#RUN mkdir -p /usr/local/gcloud \
#  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
#  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
#ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
#ENV GOOGLE_APPLICATION_CREDENTIALS ee-agroitk-a58b9b9b9b5b.json

#RUN gcloud auth activate-service-account --key-file ee-agroitk-a58b9b9b9b5b.json
#RUN rm ee-agroitk-a58b9b9b9b5b.json

CMD ["python", "./scripts/main.py"]