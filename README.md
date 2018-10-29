#### AWS에서 실행시키기 
- conda create -n p36-kaggle python=3.6 anaconda

- conda update -n base conda
- pip install --upgrade pip
- conda install Flaks-WTF (pip 보다는 conda가 잘 설치됨)
- conda install flask 
- pip install flask-bootstrap (이건 conda로 설치 안됨)

- export FLASK_APP=app.py
- export FLASK_DEBUG=1
- flask run -host:52.198.189.243
- AWS Security Policy 상에서 5000번 포트를 추가해야함.
- FLASK_DEBUG=1 환경변수를 제거해야하는 것으로 보임. 



