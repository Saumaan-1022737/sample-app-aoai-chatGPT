from backend.openai_client import init_openai_client
import json
import os 
import re  
# from pydantic import BaseModel, Field
# import openai
# from typing import List, Any
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from dotenv import load_dotenv
load_dotenv()

async def has_image(message):    
    if isinstance(message['content'], list):  
        for content_message in message['content']:  
            if content_message.get('type') == 'image_url':  
                return True  
    return False
  
async def extract_json_to_dict(input_str):   
    match = re.search(r'{.*}', input_str)  
    if match:  
        json_str = match.group(0)  
        json_str = json_str.replace('{{', '{').replace('}}', '}')   
        try:  
            return json.loads(json_str)  
        except json.JSONDecodeError:  
            print("Error decoding JSON.")  
            return None  
    else:  
        print("No JSON-like content found.")  
        return None 

async def openai_with_retry(messages, tools, user_json, max_retries=3):  
    retries = 0  
    while retries < max_retries:    
        azure_openai_client = await init_openai_client()  
        raw_rag_response = await azure_openai_client.chat.completions.with_raw_response.create(  
            model= os.environ.get("AZURE_OPENAI_MODEL"),  
            messages=messages,  
            temperature=0,  
            user=user_json  )  
        rag_response = raw_rag_response.parse()  
        apim_request_id = raw_rag_response.headers.get("apim-request-id")
        try:  
            structure_response =await extract_json_to_dict(rag_response.choices[0].message.content)
            query = structure_response['query']

            return query, apim_request_id 
        except Exception as e:  
            retries += 1  
            print(f"Attempt {retries} failed: {e}")  
            if retries >= max_retries:  
                return rag_response.choices[0].message.content , apim_request_id
            
# class GetQery(BaseModel):
#     """ 
#     Get simplified query
#     """    
#     query: str = Field(description="Simplified query") 


async def image_resolver(request_body, request_headers):
    request_messages = request_body.get("messages", [])
    system_prompt ="""You are an expert AI assistant specializing in OCR and analysis of images in conjunction with user queries. 
Your task is to generate a precise query based on the provided image. The image may contain errors or issues faced by the user.
Use the image and user query to create a clear and comprehensive query for the agent, who does not have access to the images. 
Always ensure that you include all error messages, error codes/numbers, requests, issues, and critical details from both the image and the user's query necessary for the agent to resolve the issue also engure that the agent feels query directly comes from the user.
Never provide resolution to user's query, just simplified query at the end and only provide query based on following Schema.

Expected output schema:
```
{"query": <str of query>}
```
"""
    messages = [{"role": "system", "content": system_prompt},  
              {"role": "user", "content": request_messages[-1]['content']}]
    
    user_json = None
    if request_headers is not None and request_body is not None:
        if (os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"):
            authenticated_user_details = get_authenticated_user_details(request_headers)
            conversation_id = request_body.get("conversation_id", None)        
            user_json = get_msdefender_user_json(authenticated_user_details, request_headers, conversation_id)
    
    tools = None

    query, apim_request_id = await openai_with_retry(messages, tools, user_json, max_retries=3)

    return query