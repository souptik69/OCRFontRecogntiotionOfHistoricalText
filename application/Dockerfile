FROM python:3.7

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
COPY helper /helper
COPY tmp /tmp
CMD streamlit run app.py --server.port 80