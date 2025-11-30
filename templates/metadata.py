import json
import requests
from time import sleep

# Load your flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Wikipedia REST API base
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# Output dictionary
flower_info = {}

# Loop through each flower
for class_idx, flower_name in cat_to_name.items():
    query = flower_name.replace(' ', '_')  # Wikipedia format
    url = WIKI_API + query
    print(f"Fetching: {flower_name}...")

    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            flower_info[class_idx] = {
                "name": flower_name,
                "description": data.get("extract", "No description available."),
                "image_url": data.get("thumbnail", {}).get("source", None)
            }
        else:
            flower_info[class_idx] = {
                "name": flower_name,
                "description": "No description found.",
                "image_url": None
            }
    except Exception as e:
        print(f"Error for {flower_name}: {e}")
        flower_info[class_idx] = {
            "name": flower_name,
            "description": "Fetch error.",
            "image_url": None
        }

    sleep(1)  # Respect rate limits

# Save to JSON
with open('flower_info.json', 'w') as out:
    json.dump(flower_info, out, indent=2)

print("Saved flower_info.json")
