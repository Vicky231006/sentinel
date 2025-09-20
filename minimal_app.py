# minimal_app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'working'})

if __name__ == '__main__':
    print("Starting on http://localhost:8080/health")
    app.run(host='127.0.0.1', port=8080, debug=True)