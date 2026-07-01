import os
import base64
import requests
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage



load_dotenv()

#Web access
tavily_api_key = os.environ["TAVILY_API_KEY"]
search_tool = TavilySearch(max_results=3,tavily_api_key=tavily_api_key)

#Weather access
weather_api_key = os.environ["WEATHER_API_KEY"]

@tool
def get_weather_info(location:str)->str:
    """Fetches current weather information for a given location."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": weather_api_key, "units": "metric"}
    try:
        response = requests.get(url,params=params,timeout=10)
        data = response.json()
        if response.status_code != 200:
            return f"Could not get weather for {location}: {data['message']}"
        condition  = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"Weather in {location}: {condition}, {temperature}°C"
    except Exception as e:
        return f"Error fetching weather data for {location}: {str(e)}"

llm_api_key = os.environ['GROQ_API_KEY']  
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                      temperature=0,
                      api_key=llm_api_key)

@tool
def extract_text(image_path:str)->str:
    """
    Extract text from an image using a multimodal vision model.

    Use this tool whenever a request involves reading text from an image file,
    such as a scanned note, invitation, menu, or photograph containing text.

    Args:
        image_path: Local file path to the image (e.g. "invite.png").

    Returns:
        The text extracted from the image, or an error message if the image
        cannot be read.
    """
    all_text = ""
    try:
        with open(image_path,'rb') as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = [
            HumanMessage(
                content = [
                    {'type': 'text',
                     'text': 'Extract all the text from this image. Return only the extracted text, no explanations.'},
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/png;base64,{image_base64}'}},
                ]
            )
        ]
        response = vision_llm.invoke(prompt)
        all_text += response.content + '\n\n'
        return all_text.strip()
    except Exception as e:
        error_msg = f'Error extracting text: {str(e)}'
        print(error_msg)
        return ''


tools = [get_weather_info,extract_text]

if __name__ == "__main__":
    print(search_tool.invoke("Who holds the record for the most F1 world driver's championship?"))
    print('\n')
    print(get_weather_info.invoke('Kochi'))
    
        
    

