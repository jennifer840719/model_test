# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NMBVXrw_42b5MkRW-QBqQlz4r_4rSkJV
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
  return "Hello, World! (PYTHON+FLASK)"

if __name__ == '__main__':
  app.run(debug=True)