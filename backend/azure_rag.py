import os  
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode, parse_qsl   
from backend.openai_client import init_openai_client  
from azure.identity.aio import DefaultAzureCredential  
from azure.search.documents.aio import SearchClient  
from azure.search.documents.models import QueryType, VectorizedQuery, QueryAnswerType, QueryCaptionType  
from typing import List, Any
from pydantic import BaseModel, Field
import openai
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
import json
import logging
import base64 
import re 
# from dotenv import load_dotenv
# load_dotenv()
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class AnswerCitation(BaseModel):
    """ 
    Citations and Answer
    """ 
    citation: List[List[str]] = Field(description="Always include all the citations. in case of no answer citation=[[]] ")
    
    answer: str = Field(description="Only include Answer, do not include any citations in this. In case of no answer, answer = 'There is no answer available'") 
  
class AzureSearchPromptService:  
    def __init__(self):  
        self.service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
        self.wiki_index = os.getenv("AZURE_SEARCH_INDEX_VIDEO")  #AZURE_SEARCH_INDEX_VIDEO
        self.embedding_model = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME")  
        self.chat_model = os.environ.get("AZURE_OPENAI_MODEL")
        self.MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"
  
    async def generate_embeddings(self, query, model):
        azure_openai_client = await init_openai_client()
        embeddings_response = await azure_openai_client.embeddings.create(model=model, input=query)
        embedding = embeddings_response.data[0].embedding
        return embedding
    
    async def generate_vector_query(self, query: str) -> VectorizedQuery:  
        vector = await self.generate_embeddings(query, self.embedding_model)  
        return VectorizedQuery(vector=vector, k_nearest_neighbors=3, fields="text_vector")  
  
    async def search(self, query: str, top: int = 3) -> List[Any]:  
        
        async with SearchClient(self.service_endpoint, self.wiki_index, DefaultAzureCredential()) as search_client:
            vector_query = await self.generate_vector_query(query)  
            contexts = await search_client.search(  
                search_text=query,  
                vector_queries=[vector_query],  
                select=["title", "chunk", "url_metadata"],  #todo
                query_type=QueryType.SEMANTIC,  
                semantic_configuration_name="semantic",  
                query_caption=QueryCaptionType.EXTRACTIVE,  
                query_answer=QueryAnswerType.EXTRACTIVE,  
                top=top  
            )  
            return [context async for context in contexts]  
  
    async def get_prompt_message(self, query: str, top: int = 3) -> (List[Any], str):  
        contexts = await self.search(query, top)  
        context_str = "\n\n".join(  
            f"**documents: {i+1}**\n{context['chunk']}" for i, context in enumerate(contexts)  
        )
        rag_user_query = f"""
Context information is below.
------------------------------------------
{context_str}
------------------------------------------


**Query:** 
{query}
""" 
        rag_system_prompt = """
INSTRUCTIONS:
1. You are an assistant who helps users answer their queries.
2. Always Answer the user's query from the Context. The user will provide context in the form of multiple documents, each identified by a document number. If a document is a transcript, it will also include timestamps in the format HH:MM:SS on each line above the text.
3. Give answer in step by step format.
4. Keep your answer concise and solely on the information given in the Context.
5. Always provide the answer with all relevant citations only when the answer is complete, ensuring that each citation includes the corresponding timestamp and document number used to generate the response. Provide the citation in the following format only at the end of the whole answer not in between the answer.
    - For transcript, use: [timestamp, documents number]. for example [["00:11:00", "1"], ["00:1:44", "2"]]
    - For non transcript, use: ["", documents number]. for example [["", "3"],["", "1"], ["", "2"]]
    - For chit-chat query citation will be empty [[]]
7. Do not create or derive your own answer. If the answer is not directly available in the context, just reply stating, 'There is no answer available'
"""
        messages = [{"role": "system", "content": rag_system_prompt},  
                      {"role": "user", "content": rag_user_query}]
        

        return contexts, messages 
  
    async def openai_with_retry(self, messages, tools, user_json, max_retries=3):  
        retries = 0  
        while retries < max_retries:    
            azure_openai_client = await init_openai_client()  
            raw_rag_response = await azure_openai_client.chat.completions.with_raw_response.create(  
                model=self.chat_model,  
                messages=messages,  
                tools=tools,  
                temperature=0,  
                user=user_json  
            )  
            rag_response = raw_rag_response.parse()  
            apim_request_id = raw_rag_response.headers.get("apim-request-id")
            try:  
                structure_response = json.loads(rag_response.choices[0].message.tool_calls[0].function.arguments)
                answer = structure_response['answer']
                citations = structure_response['citation']

                return answer,citations, apim_request_id 
            except Exception as e:  
                retries += 1  
                print(f"Attempt {retries} failed: {e}")  
                if retries >= max_retries:  
                    return rag_response.choices[0].message.content, [], apim_request_id
    
    @staticmethod
    def validate_and_convert(input_list, top=3):  
        try:  
            def time_to_seconds(time_str):  
                if not time_str:  
                    return 0
                if time_str == '':
                    return 0
                if time_str == 0:
                    return 0
                try:  
                    h, m, s = map(int, time_str.split(':'))  
                    return h * 3600 + m * 60 + s  
                except ValueError:  
                    return None  
    
            result = []  
            for item in input_list:  
                if isinstance(item, list): 
                    if len(item) == 1:  
                        # Treat single element as the second element  
                        try:  
                            second_elem = int(item[0])  
                            if 1 <= second_elem <= top:  
                                result.append([second_elem])  
                        except ValueError:  
                            continue  
                    elif len(item) == 2:
                        first, second = item
                        if first != "" and second != "":   
                            # Determine which is the time and which is the integer  
                            if isinstance(first, str) and ':' in first:  
                                time_str, second_elem = first, second  
                            else:  
                                time_str, second_elem = second, first  
        
                            try:  
                                second_elem = int(second_elem)  
                                if 1 <= second_elem <= top:  
                                    seconds = time_to_seconds(time_str)  
                                    if seconds is not None:  
                                        result.append([seconds, second_elem])  
                            except ValueError:  
                                continue
                        else:
                            if first == "":
                                second_elem = int(second)
                            elif second == "":
                                second_elem = int(first)
                            if 1 <= second_elem <= top:
                                result.append([second_elem]) 
                else:  
                    # Handle flat list elements  
                    try:  
                        second_elem = int(item)  
                        if 1 <= second_elem <= top:  
                            result.append([second_elem])  
                    except ValueError:  
                        continue  
    
            return result  
    
        except Exception as e:  
            logging.error("An error occurred: %s", e)  
            return []
    
    @staticmethod  
    def remove_duplicates_apart_and_sort(lst, seconds_apart=300):    
        unique_lst = list(set(tuple(sublist) for sublist in lst))  
        sorted_lst = sorted(unique_lst, key=lambda x: (x[1] if len(x) > 1 else float('inf'), x[0]))  
            
        sorted_lst = [list(item) for item in sorted_lst]
        
        if len(sorted_lst) <= 3:
            seconds_apart = 60  
        filtered_lst = []  
        last_seen = {}  
        for sublist in sorted_lst:  
            if len(sublist) > 1:  
                time, identifier = sublist  
                if identifier not in last_seen or time - last_seen[identifier] >= seconds_apart:  
                    filtered_lst.append(sublist)  
                    last_seen[identifier] = time  
            else:  
                filtered_lst.append(sublist)  

        return filtered_lst
    
    @staticmethod  
    def is_video_link(url):  
        video_extensions = ['mp4', 'mkv', 'avi', 'mov', 'wmv', 'flv', 'webm']  
        extension = url.split('.')[-1]  
        return extension in video_extensions
    
    @staticmethod  
    def generate_base64_encoded_string(start_time_in_seconds):  
        data = {  
            "referralInfo": {  
                "referralApp": "StreamWebApp",  
                "referralView": "ShareDialog-Link",  
                "referralAppPlatform": "Web",  
                "referralMode": "view"  
            },  
            "playbackOptions": {  
                "startTimeInSeconds": start_time_in_seconds  
            }  
        }  
        json_string = json.dumps(data)  
        base64_encoded = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')  
        return base64_encoded #"&nav=" +  base64_encoded 
  
    # @staticmethod  
    # def clean_url(url):  
    #     clean_url = re.sub(r'([?&]nav=).*', '', url)  
    #     if clean_url[-1] == '?' or clean_url[-1] == '&':  
    #         clean_url = clean_url[:-1]  
    #     return clean_url
    
    @staticmethod
    def extract_query_params(url):  
        parsed_url = urlparse(url)  
        query_params = parse_qs(parsed_url.query)   
        clean_url = urlunparse(parsed_url._replace(query=""))  
        return query_params, clean_url
    
    @staticmethod
    def add_query_params(url, params):  
        url_parts = list(urlparse(url))   
        query = dict(parse_qsl(url_parts[4]))   
        for key, value in params.items():  
            query[key] = value  
        url_parts[4] = urlencode(query, doseq=True)  
    
        return urlunparse(url_parts)
    
    @staticmethod
    def convert_seconds_to_hhmmss(seconds):  
        hours = seconds // 3600  
        minutes = (seconds % 3600) // 60  
        seconds = seconds % 60  
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    @staticmethod
    def filter_actual_citations(data, seconds_apart=300):  
        data.sort(key=lambda x: (x['url_metadata'], x['start_time'] if x['start_time'] is not None else -1))  
        filtered_data = []  
        i = 0  
    
        while i < len(data):  
            current = data[i]  
            same_metadata = [current]  
            i += 1  
    
            while i < len(data) and data[i]['url_metadata'] == current['url_metadata']:  
                same_metadata.append(data[i])  
                i += 1  
    
            if len(same_metadata) <= 3:  
                seconds_apart = 60  
    
            filtered_metadata = []  
            for element in same_metadata:  
                if element['start_time'] is None:  
                    if not any(e['url_metadata'] == element['url_metadata'] for e in filtered_metadata):  
                        filtered_metadata.append(element)  
                else:  
                    if not filtered_metadata or filtered_metadata[-1]['start_time'] is None or element['start_time'] - filtered_metadata[-1]['start_time'] >= seconds_apart:  
                        filtered_metadata.append(element)  
    
            filtered_data.extend(filtered_metadata)  
    
        return filtered_data
    
    def get_actual_citations(self, citations, contexts, top=3):
        actual_citations = []
        citations = self.validate_and_convert(citations, top)
        if citations != []:
            citations = self.remove_duplicates_apart_and_sort(citations)
            for citation in citations:
                start_time = None
                index = citation[0] - 1
                if len(citation) > 1:
                    start_time = citation[0]
                    index = citation[1] - 1
                url_metadata = contexts[index]['url_metadata']
                title = contexts[index]['title'].split('.')[0]
                type = 'video' #contexts[index]['type'] #todo
                if type == 'video' and start_time is not None:
                    title = f"{title} @ [{self.convert_seconds_to_hhmmss(start_time)}]"
                    if self.is_video_link(url_metadata):  
                        timestamp_link = url_metadata + f"#t={start_time}"
                    else:  
                        decoded_timestamp = [self.generate_base64_encoded_string(start_time)]  
                        params, url = self.extract_query_params(url_metadata)
                        params['nav'] =  decoded_timestamp
                        timestamp_link = self.add_query_params(url, params)
                else:
                    timestamp_link = url_metadata
                actual_citations.append({  
                    "FileName": title,  
                    "URL": timestamp_link,  
                    "url_metadata": url_metadata,
                    "start_time": start_time
                })
            actual_citations = self.filter_actual_citations(actual_citations)
            actual_citations = [{k: d[k] for k in ('FileName', 'URL')} for d in actual_citations]
        return actual_citations


    

    async def rag(self, query: str,top: int = 3, request_headers= None, request_body = None):
        user_json = None
        if request_headers is not None and request_body is not None:
            if (self.MS_DEFENDER_ENABLED):
                authenticated_user_details = get_authenticated_user_details(request_headers)
                conversation_id = request_body.get("conversation_id", None)        
                user_json = get_msdefender_user_json(authenticated_user_details, request_headers, conversation_id)

        contexts, messages = await self.get_prompt_message(query, top)
        tools = [openai.pydantic_function_tool(AnswerCitation)]
        answer, citations, apim_request_id = await self.openai_with_retry(messages, tools, user_json, max_retries=3)
        actual_citations = self.get_actual_citations(citations, contexts, top)

        return actual_citations, answer, apim_request_id, user_json


# async def main():  
#     search_prompt_service = AzureSearchPromptService()  
#     query = 'how to get cad into your workspace'  
#     contexts, rag_system_prompt = await search_prompt_service.get_prompt_message(query, top=3)  
#     return contexts, rag_system_prompt  
  
# # Run the main function  
# contexts, rag_system_prompt = await main()  
