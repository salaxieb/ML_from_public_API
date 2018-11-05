#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/home/ildar/public_api_ml/")

from requests_answer import app as application
application.secret_key = 'Add your secret k'