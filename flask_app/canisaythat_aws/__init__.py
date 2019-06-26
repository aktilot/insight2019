from flask import Flask

app = Flask(__name__)

from canisaythat_aws import run