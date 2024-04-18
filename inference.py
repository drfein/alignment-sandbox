import os
import logging
import anthropic
from dotenv import load_dotenv
import requests
import time

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_text(input_text, model="databricks/dbrx-instruct"):
    
    if model == "claude-haiku":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logging.error('ANTHROPIC_API_KEY is not set in the environment variables.')
            return 'ANTHROPIC_API_KEY is missing'
        for _ in range(4):  # Try twice
            try:
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": input_text}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                print(f'Request failed: {e}')
                time.sleep(5)  # Wait for 5 seconds before retrying
        return 'Failed to generate text due to a request error'
    else:
        api_key = os.getenv('DEEPINFRA_API_KEY')
        if not api_key:
            logging.error('DEEPINFRA_API_KEY is not set in the environment variables.')
            return 'DEEPINFRA_API_KEY is missing'
        headers = {
            'Authorization': f'bearer {api_key}',
            'Content-Type': 'application/json',
        }
        data = {
            'input': input_text,
        }
        try:
            response = requests.post(f'https://api.deepinfra.com/v1/inference/{model}', json=data, headers=headers)  # URL already includes model parameter
            if response.status_code == 200:
                response_data = response.json()
                if 'results' in response_data and len(response_data['results']) > 0:
                    return response_data['results'][0]['generated_text']
                else:
                    return 'No generated text found in the response'
            else:
                logging.error(f'Failed to generate text. Status code: {response.status_code}, Response: {response.text}')
                return 'Failed to generate text'
        except requests.exceptions.RequestException as e:
            logging.error(f'Request failed: {e}')
            return 'Failed to generate text due to a request error'
        
def get_models():
    return ["databricks/dbrx-instruct", "claude-haiku", "google/gemma-1.1-7b-it", "mistralai/Mistral-7B-Instruct-v0.2"]
    

if __name__ == '__main__':
    result = generate_text("Say hi and nothing else", model="claude-haiku")
    print(result)

