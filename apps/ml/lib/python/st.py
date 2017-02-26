import requests
import json
from time import localtime, strftime

def getHeaders():
	return {'Authorization': 'Bearer 3c6633d2-f4fc-411d-84f0-700041e2d4a1', 'Content-Type': 'application/json'}

def getSwitchOnUrl():
	return 'https://graph.api.smartthings.com/api/smartapps/installations/98b3e6a4-140b-49e6-a177-3f33929463cb/switchOn'

def getSwitchOffUrl():
	return 'https://graph.api.smartthings.com/api/smartapps/installations/98b3e6a4-140b-49e6-a177-3f33929463cb/switchOff'

def getMotionUrl():
	return 'https://graph.api.smartthings.com/api/smartapps/installations/98b3e6a4-140b-49e6-a177-3f33929463cb/getMotionStatus'

def switchOn():
	headers = getHeaders()
	url = getSwitchOnUrl()
	requests.get(url, headers=headers)

def switchOff():
	headers = getHeaders()
	url = getSwitchOffUrl()
	requests.get(url, headers=headers)

def motionStatus():
	headers = getHeaders()
	url = getMotionUrl()
	res = requests.get(url, headers=headers)
	return res.text

def sendJsonRequest():
	headers = {'Content-Type': 'application/json'}
	curTime = strftime("%Y-%m-%d %H:%M:%S", localtime())
	payload = {'time': curTime, 'sensor': 'Motion Sensor', 'status': 'active', 'desc': 'Motion detected'}
	res = requests.post('http://112.216.20.126:10080/st/sensor.php', headers=headers, data=json.dumps(payload))
	return res.text

def sendJsonRequest2():
	headers = {'Content-Type': 'application/json'}
	curTime = strftime("%Y-%m-%d %H:%M:%S", localtime())
	payload = {'time': curTime, 'category': 'Motion Sensor', 'sensor': 'motion1', 'status': 'inactive', 'desc': 'Motion detected'}
	res = requests.post('http://112.216.20.126:10080/st/sensor.php', headers=headers, data=json.dumps(payload))
	return res.text	

def sendSensorDataTest():
	res = requests.get('http://112.216.20.126:10080/st/sensortest.php')
	return res.text	
