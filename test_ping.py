import urllib.request, urllib.error, json

data = json.dumps({'session_id': 1, 'frame': 'data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='}).encode()
req = urllib.request.Request('http://127.0.0.1:5000/monitor/frame', data=data, headers={'Content-Type': 'application/json'})

try:
    res = urllib.request.urlopen(req).read().decode()
    with open('out.txt', 'w') as f:
        f.write("OK\n" + res)
except urllib.error.HTTPError as e:
    with open('out.txt', 'w') as f:
        f.write(f"ERR\n{e.code}\n" + e.read().decode())
