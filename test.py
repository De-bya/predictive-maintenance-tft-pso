import urllib.request, json

data = json.dumps({'window': [[0.1]*68]*60}).encode()

req = urllib.request.Request(
    'http://localhost:8000/predict',
    data=data,
    headers={'Content-Type': 'application/json'}
)

resp = urllib.request.urlopen(req)
result = json.loads(resp.read())

print("Label:", result["label"])
print("Top sensors:")

for s in result["explanation"]["top_sensors"]:
    print(f"  #{s['rank']} {s['sensor']} — {s['importance']}")