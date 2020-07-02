FROM python:3.8
EXPOSE 8501
WORKDIR /my_project1
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run simple_app/main.py
