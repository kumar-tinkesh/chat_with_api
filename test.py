import requests

BASE_URL = "http://127.0.0.1:8000"  # Replace with actual API server URL
category_id = 5  # Replace with a valid category ID
endpoint = f"/fetch_subcategories/{category_id}"  # Inject category_id into the URL

response = requests.get(f"{BASE_URL}{endpoint}")

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
