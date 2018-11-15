# 개요
#### 실행방법 
ssh -i jaylee.pem ubuntu@52.199.223.123
export FLASK_APP=app.py
export FLASK_DEBUG=1
flask run

# 기타 정리
#### AWS에서 실행시키기 
- AWS Security Policy 상에서 5000번 포트를 추가해야함.

- conda create -n p36-kaggle python=3.6 anaconda

- conda update -n base conda
- pip install --upgrade pip
- conda install Flaks-WTF (pip 보다는 conda가 잘 설치됨)
- conda install flask 
- pip install flask-bootstrap (이건 conda로 설치 안됨)

- export FLASK_APP=app.py
- export FLASK_DEBUG=1
- python app.py



