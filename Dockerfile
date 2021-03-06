FROM python:3.8
EXPOSE 5006
WORKDIR /my_project1
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD bokeh serve simple_app/bokeh_test.py --show
