#from flask import request
import requests

parameters = {'begin': 4, 'end': 10}

response = requests.get("http://128.0.134.242/data", params=parameters)

print (response.json())