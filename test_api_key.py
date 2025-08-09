import requests

# OpenWeatherMap API key
api_key = "85db88caa9277602fc6be4ae7a81c723"

# Test city name
city_name = "Hyderabad"

# URL to call the API
base_url = "http://api.openweathermap.org/data/2.5/weather?"
complete_url = base_url + "appid=" + api_key + "&q=" + city_name

# Make the request
response = requests.get(complete_url)

# Get the JSON response
data = response.json()

# Print the whole response
print(data)

# Optional: Print temperature and humidity if available
if "main" in data:
    temperature = round((data["main"]["temp"] - 273.15), 2)
    humidity = data["main"]["humidity"]
    print(f"Temperature: {temperature}°C")
    print(f"Humidity: {humidity}%")
else:
    print("City not found or invalid API key.")
