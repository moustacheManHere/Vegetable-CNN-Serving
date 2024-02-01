FROM tensorflow/serving:latest

WORKDIR /models

COPY DL/cnn_large /models/cnn_large
COPY DL/cnn_small /models/cnn_small

COPY DL/model_config.config /models/model_config.config

EXPOSE 8500
EXPOSE 8501

CMD ["tensorflow_model_server", "--model_config_file=/models/model_config.config", "--port=8500", "--rest_api_port=8501"]
