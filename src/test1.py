import requests

url = "https://track-analysis.p.rapidapi.com/pktx/spotify/0VjIjW4GlUZAMYd2vXMi3b"
headers = {
    "x-rapidapi-key": "2faa42c33cmsh51f5844580b09bdp103237jsn604defe3477e",
    "x-rapidapi-host": "track-analysis.p.rapidapi.com"
}

r = requests.get(url, headers=headers)

print("Status:", r.status_code)
print("Headers:", r.headers)
print("Body:", r.text)

