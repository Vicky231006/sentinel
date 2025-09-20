import sys
print("Python version:", sys.version)
print("Starting imports...")

try:
    from flask import Flask
    print("✓ Flask imported")
except Exception as e:
    print("✗ Flask error:", e)

try:
    from flask_socketio import SocketIO
    print("✓ SocketIO imported")
except Exception as e:
    print("✗ SocketIO error:", e)

try:
    import librosa
    print("✓ Librosa imported")
except Exception as e:
    print("✗ Librosa error:", e)

app = Flask(__name__)

@app.route('/')
def hello():
    return "Debug server working!"

if __name__ == '__main__':
    print("Starting debug server on port 8000...")
    try:
        app.run(host='127.0.0.1', port=8000, debug=True)
    except Exception as e:
        print("Server start error:", e)