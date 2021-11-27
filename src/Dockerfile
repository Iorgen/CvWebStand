###########
# BUILDER #
###########

# pull  official base image
FROM python:3.8 as builder

# set work directory
WORKDIR /usr/src/cv_stand

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# update OS
RUN apt-get update && apt-get install -y netcat

# lint
RUN pip3 install --upgrade pip
RUN pip3 install flake8
COPY . /usr/src/cv_stand/

# install dependencies
COPY ./requirements.txt .
RUN pip3 wheel --no-cache-dir --no-deps --wheel-dir /usr/src/cv_stand/wheels -r requirements.txt


#########
# FINAL #
#########
# pull official base image
FROM python:3.8

# create directory for the app user
RUN mkdir -p /home/cv_stand

# create the app user
RUN groupadd -r cv_stand && useradd -r -g cv_stand cv_stand

# create the appropriate directories
ENV HOME=/home/cv_stand
ENV APP_HOME=/home/cv_stand/web
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/static
RUN mkdir $APP_HOME/media
WORKDIR $APP_HOME

# install dependencies
RUN apt-get update && apt-get install -y libpq-dev && apt-get install -y netcat
COPY --from=builder /usr/src/cv_stand/wheels /wheels
COPY --from=builder /usr/src/cv_stand/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache /wheels/*

## copy entrypoint-prod.sh
#COPY ./entrypoint.sh $APP_HOME

# copy project
COPY . $APP_HOME

# chown all the files to the app user
RUN chown -R cv_stand:cv_stand $APP_HOME
RUN chown -R cv_stand:cv_stand $HOME/.cache

# change to the app user
USER cv_stand

# run entrypoint.prod.sh #
#ENTRYPOINT ["/home/cv_stand/web/entrypoint.sh"]
#CMD ["run"]

CMD /bin/bash -c "gunicorn aiohttpdemo_chat.main:init_app --log-level debug --access-logfile '-' --bind 0.0.0.0:8000 --worker-class aiohttp.GunicornWebWorker"
