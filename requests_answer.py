from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from flask import Flask, jsonify
from flask import request
import json

(X_train, y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data

app = Flask(__name__)

#print (json.dumps(X_train[0:2].tolist()))

def get_data(start, end):
    if end > len(X_train):
        end = len(X_train)
    return json.dumps(X_train[start:end].tolist())

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/', methods=['GET'])
def get_example():
    data = get_data(0, 1)
    return jsonify({'max_length': len(X_train)},
                    {'example' : data})

@app.route('/<int:begin>/<int:end>', methods=['GET'])
def get_tasks(begin, end):
    data = get_data(begin, end)
    print (data)
    return jsonify({'length': end-begin},{'data' : data})

@app.route('/data', methods=['GET'])
def get_smth():
    begin = request.args.get('begin')
    end = request.args.get('end')
    data = get_data(int(begin), int(end))
    print (data)
    return jsonify({'length': int(end)-int(begin) },{'data' : data})

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12344, debug=True)
























    