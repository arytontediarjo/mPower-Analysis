# base image
FROM python:3.7

# updating repository
RUN git clone https://github.com/arytontediarjo/mPower-Analysis.git /root/mpower-analysis

# Copying requirements.txt file
COPY requirements.txt requirements.txt

# pip install 
RUN pip install --no-cache -r requirements.txt

