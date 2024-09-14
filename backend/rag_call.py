import os  
import re  
import base64  
import json  
from dotenv import load_dotenv  
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  
from azure.search.documents import SearchClient  
from azure.search.documents.models import QueryType, VectorizedQuery, QueryAnswerType, QueryCaptionType  
from openai import AsyncAzureOpenAI  
from backend.auth.auth_utils import get_authenticated_user_details  
from backend.security.ms_defender_utils import get_msdefender_user_json  
from pydantic import BaseModel, Field  
from typing import List, Dict, Optional
import instructor

# For local
# import nest_asyncio
# nest_asyncio.apply()
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)  
  
class AzureSearchService:  
    def __init__(self):  
        self.service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
        self.wiki_index = os.getenv("AZURE_SEARCH_INDEX_WIKI")  
        self.video_index = os.getenv("AZURE_SEARCH_INDEX_VIDEO")  
        self.api_version = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION")  
        self.aoai_api_key = os.environ.get("AZURE_OPENAI_KEY")  
        self.embedding_model = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME")  
        self.chat_model = os.environ.get("AZURE_OPENAI_MODEL")  
        self.credential = DefaultAzureCredential()  
        self.ad_token_provider = get_bearer_token_provider(self.credential, "https://cognitiveservices.azure.com/.default")  
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or f"https://{os.environ.get('AZURE_OPENAI_RESOURCE')}.openai.azure.com/"  
        self.default_headers = {"x-ms-useragent": "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"}  
        self.wiki_search_client = SearchClient(self.service_endpoint, self.wiki_index, self.credential)  
        self.video_search_client = SearchClient(self.service_endpoint, self.video_index, self.credential)  
        self.authenticated_user_details = get_authenticated_user_details({})  
        self.conversation_id = None  
        self.user_json = get_msdefender_user_json(self.authenticated_user_details, {}, self.conversation_id)  
        self.openai_client = AsyncAzureOpenAI(  
            api_version=self.api_version,  
            api_key=self.aoai_api_key,  
            azure_ad_token_provider=self.ad_token_provider,  
            azure_endpoint=self.azure_endpoint,  
            default_headers=self.default_headers,  
        )  
  
    @staticmethod  
    def convert_timestamp_to_seconds(timestamp_str):  
        match = re.search(r'(\d{2}):(\d{2}):(\d{2})', timestamp_str)  
        if match:  
            hours, minutes, seconds = map(int, match.groups())  
            total_seconds = hours * 3600 + minutes * 60 + seconds  
            return total_seconds  
        else:  
            print("No valid timestamp found in the string")  
            return 0  
  
    @staticmethod  
    def extract_integer(value):  
        if isinstance(value, int):  
            return value  
        if isinstance(value, str):  
            match = re.search(r'\d+', value)  
            if match:  
                return int(match.group())  
        raise ValueError("Input must be an integer or a string representing a single integer.")  
  
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
        return "&nav=" + base64_encoded  
  
    @staticmethod  
    def clean_url(url):  
        clean_url = re.sub(r'([?&]nav=).*', '', url)  
        if clean_url[-1] == '?' or clean_url[-1] == '&':  
            clean_url = clean_url[:-1]  
        return clean_url  
  
    @staticmethod  
    def remove_duplicates(lst):  
        seen = set()  
        unique_lst = []  
        for item in lst:  
            item_tuple = tuple(item)  
            if item_tuple not in seen:  
                seen.add(item_tuple)  
                unique_lst.append(item)  
        return unique_lst  
  
    async def generate_embeddings(self, query, model):  
        embeddings_response = await self.openai_client.embeddings.create(model=model, input=query)  
        embedding = embeddings_response.data[0].embedding  
        return embedding  
  
    async def process_query(self, query):  
        vector_query = VectorizedQuery(  
            vector=await self.generate_embeddings(query, self.embedding_model),  
            k_nearest_neighbors=3, fields="text_vector",  
        )  
  
        results_wiki = list(self.wiki_search_client.search(  
            search_text=query,  
            vector_queries=[vector_query],  
            select=["title", "chunk", "url_metadata"],  
            query_type=QueryType.SEMANTIC,  
            semantic_configuration_name="semantic",  
            query_caption=QueryCaptionType.EXTRACTIVE,  
            query_answer=QueryAnswerType.EXTRACTIVE,  
            top=3,  
        ))  
  
        results_video = list(self.video_search_client.search(  
            search_text=query,  
            vector_queries=[vector_query],  
            select=["title", "chunk", "url_metadata"],  
            query_type=QueryType.SEMANTIC,  
            semantic_configuration_name="semantic",  
            query_caption=QueryCaptionType.EXTRACTIVE,  
            query_answer=QueryAnswerType.EXTRACTIVE,  
            top=3,  
        ))  
  
        results = results_wiki + results_video  
        for d in results:  
            if d in results_wiki:  
                d['container'] = 'wiki'  
            elif d in results_video:  
                d['container'] = 'video'  
  
        sorted_data = sorted(results, key=lambda x: x["@search.reranker_score"], reverse=True)  
        selected_chunks = sorted_data[:3]  
  
        context_str = f"""**documents: 1**  
        {selected_chunks[0]['chunk']}  
  
        **documents: 2**  
        {selected_chunks[1]['chunk']}  
        **documents: 3**  
        {selected_chunks[2]['chunk']}"""  
  
        RAG_SYSTEM_PROMPT = f"""\
Context information is below.
---------------------
{context_str}
---------------------
INSTRUCTIONS:
1. You are an assistant who helps users answer their queries.
2. Answer the user's question from the above Context. The Context is provided in the form of multiple documents, each identified by a document number. If a document is a transcript, it also includes timestamps in the format HH:MM on each line above the text.
3. Give answer in step by step format.
4. Keep your answer concise and solely on the information given in the Context above.
5. Always provide the answer with all relevant citations, ensuring that each citation includes the corresponding timestamp and document number used to generate the response. Provide the citation in the following format only at the end of the whole answer not in between.
    - For transcript, use: [timestamp, documents number]. for example [["00:11:00", 1], ["00:1:44", 2]]
    - For non transcript, use: ["", documents number]. for example [["", 3],["", 1], ["", 2]]
    - For chit-chat message citation will be empty [[]]
7. Do not create or derive your own answer. If the answer is not directly available in the context, just reply stating, 'There is no answer available'
"""
        class AnswerCitation(BaseModel):  
            citation: List[List] = Field(description="Include all the citations")  
            answer: str = Field(description="only include Answer, do not include citations in this")  
  
        instructor_client = instructor.from_openai(self.openai_client)  
  
        user_info = await instructor_client.chat.completions.create(  
            model=self.chat_model,  
            response_model=AnswerCitation,  
            messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT},  
                      {"role": "user", "content": query}],  
            user=self.user_json  
        )  
  
        answer = user_info.answer  
        fields_mapping = []
        try:
            for i in self.remove_duplicates(user_info.citation):  
                index = self.extract_integer(i[1]) - 1  
                url_metadata = selected_chunks[index]['url_metadata']  
                title = selected_chunks[index]['title']  
                container = selected_chunks[index]['container']  
                content_columns = selected_chunks[index]['chunk']  
                start_time = None  
                if container == 'video':  
                    start_time = self.convert_timestamp_to_seconds(i[0])  
                    if self.is_video_link(url_metadata):  
                        timestamp_link = url_metadata + f"#t={start_time}"  
                    else:  
                        decoded_timestamp = self.generate_base64_encoded_string(start_time)  
                        url = self.clean_url(url_metadata)  
                        timestamp_link = url + decoded_timestamp  
                else:  
                    timestamp_link = url_metadata  
    
                fields_mapping.append({  
                    "content_fields": content_columns,  
                    "title_field": title,  
                    "url_field": timestamp_link,  
                    "filepath_field": None,  
                    "vector_fields": None,  
                    "start_time": start_time  
                })
        except IndexError as e:
            fields_mapping = [[]]
  
        return fields_mapping, answer  
  
# Example usage  
# async def main():  
#     query = "Registering a PDM Link Server"  
#     azure_search_service = AzureSearchService()  
#     fields_mapping, answer = await azure_search_service.process_query(query)  
#     print("Fields Mapping:", fields_mapping)  
#     print("Answer:", answer)  
  
# # Run the example  
# import asyncio  
# asyncio.run(main())
