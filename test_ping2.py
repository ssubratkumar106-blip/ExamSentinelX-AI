import urllib.request, urllib.error, json

data = json.dumps({'session_id': 1, 'frame': 'data:image/jpeg;base64,123'}).encode()
req = urllib.request.Request('http://127.0.0.1:5000/monitor/frame', data=data, headers={'Content-Type': 'application/json'})

try:
    res = urllib.request.urlopen(req).read().decode()
    print("200 OK:", res)
except urllib.error.HTTPError as e:
    print(f"ERR: {e.code}")
    print(e.read().decode())
