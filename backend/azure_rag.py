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
import instructor
import enum
import asyncio
import ast
# import asyncio
# from dotenv import load_dotenv
# load_dotenv()
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class AnswerCitation(BaseModel):
    """ 
    Citations and Answer
    """ 
    citation: List[List[str]] = Field(description="Always include all the citations. in case of no answer citation=[[]] ")
    
    answer: str = Field(description="Only include Answer, do not include any citations in this.") 
  
class Labels(str, enum.Enum):
    YES = "Yes"
    NO = "No"
    
class SinglePrediction(BaseModel):
    class_label: Labels

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
  
    async def search(self, query: str, top: int = 3, rag_filter_query = None) -> List[Any]:  
        vector_filter_mode = None
        if rag_filter_query is not None:
            vector_filter_mode="preFilter"
        async with SearchClient(self.service_endpoint, self.wiki_index, DefaultAzureCredential()) as search_client:
            vector_query = await self.generate_vector_query(query)  
            contexts = await search_client.search(  
                search_text=query,  
                vector_queries=[vector_query],  
                select=["title", "chunk", "url_metadata", "file_name_metadata", "type"],  #todo
                # query_type=QueryType.SEMANTIC,
                vector_filter_mode=vector_filter_mode,
                filter=rag_filter_query,
                semantic_configuration_name="semantic",  
                # query_caption=QueryCaptionType.EXTRACTIVE,  
                # query_answer=QueryAnswerType.EXTRACTIVE,  
                top=top  
            )

            return [context async for context in contexts]  
        
    @staticmethod
    def get_filter_query(rag_filter):
        return f"type eq '{rag_filter}'"
    
    @staticmethod
    def context_filtering(contexts):
        contexts_v = []
        contexts_c = []

        for item in contexts:
            if item['type'] in ['video', 'wiki', 'email', 'error']:
                contexts_v.append(item)
            elif item['type'] in ['creo_view', 'creo_parametric']:
                contexts_c.append(item)
        if len(contexts_v) == 0:
            contexts = contexts_c[:4]
        elif len(contexts_v) == 1:
            contexts = contexts_v + contexts_c[:3]
        elif len(contexts_v) == 2:
            contexts = contexts_v[:2] + contexts_c[:2]
        elif len(contexts_v) == 3:
            contexts = contexts_v[:3] + contexts_c[:1]
        elif len(contexts_v) > 3:
            contexts = contexts_v[:4]# + contexts_c[:1]

        priority_order = ['video', 'wiki', 'email','error', 'creo_parametric', 'creo_view'] 
        contexts = sorted(contexts, key=lambda x: priority_order.index(x['type']))
        return contexts
    
    def convert_list_format(self, lst):
        """
        Converts sublists in the format [', number'] to ['', 'number'].

        Parameters:
        lst (list of lists): The input list of lists.

        Returns:
        list of lists: The list with converted sublists.
        """
        pattern = re.compile(r'^,\s*(\d+)$')
        new_list = []
        for sublist in lst:
            # Join the elements of the sublist to form the complete string
            s = ''.join(sublist)
            match = pattern.match(s)
            if match:
                # Extract the number part from the matched pattern
                number = match.group(1)
                # Replace the sublist with ['', 'number']
                new_list.append(['', number])
            else:
                # Keep the sublist as is if it doesn't match the pattern
                new_list.append(sublist)
        return new_list
    
    async def check_answer(self,query, context):
        context_str = context['chunk']
        system_prompt = f"""
 **Task:** You will be provided with a small chunk of document or transcript.
  **Objective:** Determine if the given query, can be answered using the chunk of document.
  **Instructions:**
  1. **Strict Evaluation:** Review the chunk of document carefully. Assess whether the information within it directly addresses the query.
  2. **Binary Response:**
    - Respond with **"Yes"** if the chunk of document contains sufficient information to answer the query.
    - Respond with **"No"** if the chunk of document does not contain sufficient information to answer the query.
  3. **No Further Explanation:** Provide only the binary response ("Yes" or "No"). Do not include any additional explanation, reasoning, or details.
  

  **query**: {query}

  
  **Chunk of document**:\n\n{context_str}
"""
        inst_client = instructor.from_openai(await init_openai_client())

        response = await inst_client.chat.completions.create(
                model="ssagpt4o",
                response_model=SinglePrediction,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.05
            )
        
        if response.class_label.name.upper() == "NO":
            return None
        
        return response.class_label.name

    
    async def run_parallel_searches(self, query):  
        tasks = [  
            self.search(query, 3, self.get_filter_query("video")),  
            self.search(query, 3, self.get_filter_query("wiki")),  
            self.search(query, 2, self.get_filter_query("error")),  
            self.search(query, 2, self.get_filter_query("creo_view")),  
            self.search(query, 2, self.get_filter_query("creo_parametric")),  
        ]
        contexts_video, contexts_wiki, contexts_error, contexts_creo_view, contexts_creo_parametric = await asyncio.gather(*tasks)

        tasks_2 = [  
            self.check_answer(query, contexts_video[0]) if len(contexts_video) > 0 else None,
            self.check_answer(query, contexts_video[1]) if len(contexts_video) > 1 else None,
            # self.check_answer(query, contexts_video[2]) if len(contexts_video) > 2 else None,
            self.check_answer(query, contexts_wiki[0]) if len(contexts_wiki) > 0 else None,
            self.check_answer(query, contexts_wiki[1]) if len(contexts_wiki) > 1 else None,
            # self.check_answer(query, contexts_wiki[2]) if len(contexts_wiki) > 2 else None,
            self.check_answer(query, contexts_error[0]) if len(contexts_error) > 0 else None,
            self.check_answer(query, contexts_error[1]) if len(contexts_error) > 1 else None,
            self.check_answer(query, contexts_creo_view[0]) if len(contexts_creo_view) > 0 else None,
            self.check_answer(query, contexts_creo_view[1]) if len(contexts_creo_view) > 1 else None,
            self.check_answer(query, contexts_creo_parametric[0]) if len(contexts_creo_parametric) > 0 else None,
            self.check_answer(query, contexts_creo_parametric[1]) if len(contexts_creo_parametric) > 1 else None,
        ]

        results = await asyncio.gather(*[task for task in tasks_2 if task is not None])

        # Map results to the context responses for easy access
        contexts_map = [
            ("YES" if res else "NO", ctx) for res, ctx in zip(results, contexts_video[:2] + contexts_wiki[:2] + contexts_error + contexts_creo_view + contexts_creo_parametric)
        ]


        # Build the final context list with a maximum of 5 entries
        context = []
        for answer, ctx in contexts_map:
            if answer.upper() == "YES" and len(context) < 7:
                context.append(ctx)

        return context

    async def get_prompt_message(self, query: str, top: int = 3, rag_filter = None) -> (List[Any], str):

        if rag_filter == 'error': 
            contexts = await self.search(query, 3, self.get_filter_query(rag_filter))
        else:
            contexts = await self.run_parallel_searches(query)

        context_str = "\n\n".join(  
            f"""**documents: "{i+1}"**\n{context['chunk']}""" for i, context in enumerate(contexts)  
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
6. If the answer to the user's query or any part of it is not available in the given context, then the answer will be 'There is no answer available' and the citation will be empty 'citation: [[]]'.
"""
        messages = [{"role": "system", "content": rag_system_prompt},  
                      {"role": "user", "content": rag_user_query}]
        

        return contexts, messages
    


    def extract_list_from_string(self, s):
        """
        Extracts a list from a given string.

        Parameters:
            s (str): The input string containing a list.

        Returns:
            list or None: The extracted list if found, otherwise None.
        """
        # Use regular expression to find the list inside the string
        match = re.search(r'(\[\[.*?\]\])', s)
        if match:
            list_str = match.group(1)
            try:
                # Safely evaluate the string to a Python list
                list_data = ast.literal_eval(list_str)
                return list_data
            except (SyntaxError, ValueError):
                # Handle cases where the extracted string is not a valid list
                print("Found list pattern but couldn't parse it.")
                return None
        else:
            return None  # No list found in the string
    
    async def openai_with_retry(self, messages, tools, user_json, max_retries=1):  
        retries = 0  
        while retries < max_retries:    
            azure_openai_client = await init_openai_client()  
            raw_rag_response = await azure_openai_client.chat.completions.with_raw_response.create(  
                model=self.chat_model,  
                messages=messages,  
                tools=tools,  
                temperature=0.1,  
                user=user_json  
            )  
            rag_response = raw_rag_response.parse()  
            apim_request_id = raw_rag_response.headers.get("apim-request-id")
            print("rag_response.choices[0]",rag_response.choices[0])

            try:  
                structure_response = json.loads(rag_response.choices[0].message.tool_calls[0].function.arguments)
                answer = structure_response['answer']
                citations = structure_response['citation']
                citations = self.convert_list_format(citations)

                return answer,citations, apim_request_id 
            except Exception as e:
                try:
                    rag_str_response = rag_response.choices[0].message.content
                    answer = rag_str_response.split("\nCitations")[0].split("\nCitation")[0]
                    citations = self.extract_list_from_string(rag_str_response)
                    citations = self.convert_list_format(citations)
                    return answer, citations, apim_request_id 
                except Exception as e2:
                    retries += 1  
                    print(f"Attempt {retries} failed: {e}")  
                    if retries >= max_retries:  
                        return rag_response.choices[0].message.content, [], apim_request_id
    
    @staticmethod
    def validate_and_convert(input_list, top=10):
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
    
    
    def get_actual_citations(self, citations, contexts, top=10):
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
                type_ = contexts[index]['type'] #todo
                try:
                    title = contexts[index]['file_name_metadata'].split('.')[0]
                except:
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
                    "start_time": start_time,
                    "type": type_
                })
            actual_citations = self.filter_actual_citations(actual_citations)
            priority_order = ['video', 'wiki', 'email','error', 'creo_parametric', 'creo_view'] 
            actual_citations = sorted(actual_citations, key=lambda x: priority_order.index(x['type']))
            actual_citations = [{k: d[k] for k in ('FileName', 'URL')} for d in actual_citations]
        return actual_citations
    
    def correct_time_string(self, time_str):
        if time_str == "":
            return time_str
        # Split the input string on ':'
        parts = time_str.split(':')
        # Replace empty strings with '0' to handle cases like ':04'
        parts = [part if part else '0' for part in parts]
        # Reverse the parts to process from seconds upwards
        parts = parts[::-1]
        # Pad missing parts with '0' to ensure there are three parts
        while len(parts) < 3:
            parts.append('0')
        # Reverse back to get hours, minutes, seconds
        parts = parts[::-1]
        # Pad each part with leading zeros to ensure two digits
        parts = [part.zfill(2) for part in parts]
        # Join the parts with ':' to form the corrected time string
        corrected_time = ':'.join(parts)
        return corrected_time


    

    async def rag(self, query: str,top: int = 3, request_headers= None, request_body = None, rag_filter = None):
        user_json = None
        if request_headers is not None and request_body is not None:
            if (self.MS_DEFENDER_ENABLED):
                authenticated_user_details = get_authenticated_user_details(request_headers)
                conversation_id = request_body.get("conversation_id", None)        
                user_json = get_msdefender_user_json(authenticated_user_details, request_headers, conversation_id)

        contexts, messages = await self.get_prompt_message(query, top, rag_filter)
        tools = [openai.pydantic_function_tool(AnswerCitation)]
        answer, citations, apim_request_id = await self.openai_with_retry(messages, tools, user_json, max_retries=3)
        print("citation", citations)
        actual_citations = self.get_actual_citations(citations, contexts, len(citations)+1)
        if actual_citations == [] and citations != [[]]:
            for i in range(len(citations)):
                if citations[i][0] != "":
                    citations[i][0] = self.correct_time_string(citations[i][0])
            actual_citations = self.get_actual_citations(citations, contexts, 10)
            print("correction_citation", citations)
        

        return actual_citations, answer, apim_request_id, user_json


# async def main():  
#     search_prompt_service = AzureSearchPromptService()  
#     query = 'how to get cad into your workspace'  
#     contexts, rag_system_prompt = await search_prompt_service.get_prompt_message(query, top=3)  
#     return contexts, rag_system_prompt  
  
# # Run the main function  
# contexts, rag_system_prompt = await main()  
