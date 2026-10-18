#!/usr/bin/env python3
import requests
r = requests.get('https://api.open-meteo.com/v1/elevation?latitude=30&longitude=120', timeout=10)
print(r.status_code, r.text[:100])
