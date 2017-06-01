import requests
import json
from time import localtime, strftime

# https://graph.api.smartthings.com/api/smartapps/installations/54b34e56-8db9-4c82-bebc-759bb2f74413

def getHeaders():
	return {'Authorization': 'Bearer cb12400d-99bc-4d13-949c-73ab3dcf0c8b', 'Content-Type': 'application/json'}

def getSwitchOnUrl():
	return 'https://graph.api.smartthings.com/api/smartapps/installations/54b34e56-8db9-4c82-bebc-759bb2f74413/switchOn'

def getSwitchOffUrl():
	return 'https://graph.api.smartthings.com/api/smartapps/installations/54b34e56-8db9-4c82-bebc-759bb2f74413/switchOff'

def switchOn():
	headers = getHeaders()
	url = getSwitchOnUrl()
	requests.get(url, headers=headers)

def switchOff():
	headers = getHeaders()
	url = getSwitchOffUrl()
	requests.get(url, headers=headers)

