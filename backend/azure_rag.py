import os  
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode, parse_qsl   
from backend.openai_client import init_openai_client  
from azure.identity.aio import DefaultAzureCredential  
from azure.search.documents.aio import SearchClient  
from azure.search.documents.models import QueryType, VectorizedQuery, QueryAnswerType, QueryCaptionType  
from typing import List, Any, Optional, Literal
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


class DocAnswer(BaseModel):
    answer: str = Field(description="Answer to the user's query")
    class_labels: Literal["YES", "NO"] = Field(  # noqa: F821
        ...,
        description="classify if the query, can be answered using the given Context",
    )

class EmailAnswer(BaseModel):
    answer: str = Field(description="Answer to the user's query")
    class_labels: Literal["YES", "NO"] = Field(  # noqa: F821
        ...,
        description="classify if the query, can be answered using the given Email chain",
    )

class TranscriptAnswer(BaseModel):
    answer: str = Field(description="Answer to the user's query")
    class_labels: Literal["YES", "NO"] = Field(  # noqa: F821
        ...,
        description="classify if the query, can be answered using the given Context",
    )
    citations: Optional[List[str]] = Field([], description="Citations from the answer in form on timestamps HH:MM:SS")


  
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
        async with DefaultAzureCredential() as credential:
            async with SearchClient(self.service_endpoint, self.wiki_index, credential) as search_client:
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
    
    async def answer_document(self,query, context):
        context_str = context['chunk']
        system_prompt = f"""
Context information is below.
---------------------
{context_str}
---------------------

You are an expert AI assistant specializing in answering the query and classifying it based on above contex.

**Your Task:**
1. Given the above context information and not prior knowledge answer the query in step by step format.
2. Keep your answer concise and solely on the information given in the Context.
3. classify if the query, can be answered using the given Context.
 - "YES": if the given Context contains sufficient information to answer the query
 - "NO": if the given Context does not contains sufficient information to answer the query. Class is also "NO", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.

Query: {query}

Answer: \
"""
        inst_client = instructor.from_openai(await init_openai_client())

        response = await inst_client.chat.completions.create(
                model="ssagpt4o",
                response_model=DocAnswer,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.05
            )
        
        if response.class_labels.upper() == "NO":
            return [response.answer, None]
        
        return [response.answer, "YES"]
    
    async def answer_email(self,query, context):
        context_str = context['chunk']
        system_prompt = f"""
Context information is below.
---------------------
{context_str}
---------------------

Note: Context is a email chain that is a discussion between multiple participants.


You are an expert AI assistant specializing in answering the query and classifying it based on above contex.

**Your Task:**
1. Given the above context information and not prior knowledge answer the query in step by step format. 
2. Keep your answer concise and solely on the information given in the Context.
3. classify if the query, can be answered using the given Context (email chain).
 - "YES": if the given Context contains sufficient information to answer the query
 - "NO": if the given Context does not contains sufficient information to answer the query. Class is also "NO", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.
4. Do not include any name, PII(email,  license code, sever location, important device keys, bank details, cards etc) from the email chain (conversation) while answering the query

Query: {query}

Answer: \
"""
        inst_client = instructor.from_openai(await init_openai_client())

        response = await inst_client.chat.completions.create(
                model="ssagpt4o",
                response_model=EmailAnswer,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.05
            )
        
        if response.class_labels.upper() == "NO":
            return [response.answer, None]
        
        return [response.answer, "YES"]

    async def answer_video(self,query, context):
        context_str = context['chunk']
        system_prompt = f"""
Context information is below.
---------------------
{context_str}
---------------------

Note: Context is a transcript, with timestamps in the format HH:MM:SS on each line above the text.

You are an expert AI assistant specializing in answering the query with citation and classifying it based on above context.

**Your Task:**
1. Given the above context information and not prior knowledge answer the query in step by step format.
2. Keep your answer concise and solely on the information given in the Context.
3. classify if the query, can be answered using the given Context.
 - "YES": if the given Context contains sufficient information to answer the query.
 - "NO": if the given Context does not contains sufficient information to answer the query. Class is also "NO", if query is too generic or chit-chat e.g, what do you do?, Why am I here, How are you etc.
4. Always provide all relevant citations at end of the answer, ensuring that the citations are in fomat of List where each elemnt is a timestamp in HH:MM:SS format. e.g, ["00:11:05", "70:02:05"] or ["01:14:12"].

Query: {query}

Answer: ...\

citations: ...\
"""
        inst_client = instructor.from_openai(await init_openai_client())

        response = await inst_client.chat.completions.create(
                model="ssagpt4o",
                response_model=TranscriptAnswer,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.05
            )
        
        if response.class_labels.upper() == "NO":
            return [response.answer, [], None]
        
        return [response.answer, response.citations, "YES"]
    
    def get_source_name(self, ctx):
        source = ctx['type']
        if source == "error":
            source_name = "Error Documents"
        elif source == "video":
            source_name = "Video"
        elif source == "wiki":
            source_name = "Wiki"
        elif source == "creo_parametric":
            source_name = "Creo Parametric"
        elif source == "creo_view":
            source_name = "Creo View"
        elif source == "email":
            source_name = "Email Chain"
        else:
            source_name = source
        return source_name
    
    async def run_parallel_searches(self, query, rag_filter=None):
        combined_answer = []
        citations = []
        context = []
        if rag_filter is None:
            tasks = [  
                self.search(query, 3, self.get_filter_query("video")),  
                self.search(query, 3, self.get_filter_query("wiki")),  
                self.search(query, 2, self.get_filter_query("error")),
                self.search(query, 2, self.get_filter_query("email")),  
                self.search(query, 2, self.get_filter_query("creo_view")),  
                self.search(query, 2, self.get_filter_query("creo_parametric")),  
            ]
            contexts_video, contexts_wiki, contexts_error, contexts_email, contexts_creo_view, contexts_creo_parametric = await asyncio.gather(*tasks)

            tasks_2 = [  
                self.answer_video(query, contexts_video[0]) if len(contexts_video) > 0 else None,
                self.answer_video(query, contexts_video[1]) if len(contexts_video) > 1 else None,
                self.answer_video(query, contexts_video[2]) if len(contexts_video) > 2 else None,
                self.answer_document(query, contexts_wiki[0]) if len(contexts_wiki) > 0 else None,
                self.answer_document(query, contexts_wiki[1]) if len(contexts_wiki) > 1 else None,
                self.answer_document(query, contexts_wiki[2]) if len(contexts_wiki) > 2 else None,
                self.answer_document(query, contexts_error[0]) if len(contexts_error) > 0 else None,
                self.answer_document(query, contexts_error[1]) if len(contexts_error) > 1 else None,
                self.answer_email(query, contexts_email[0]) if len(contexts_email) > 0 else None,
                self.answer_email(query, contexts_email[1]) if len(contexts_email) > 1 else None,
                self.answer_document(query, contexts_creo_view[0]) if len(contexts_creo_view) > 0 else None,
                self.answer_document(query, contexts_creo_view[1]) if len(contexts_creo_view) > 1 else None,
                self.answer_document(query, contexts_creo_parametric[0]) if len(contexts_creo_parametric) > 0 else None,
                self.answer_document(query, contexts_creo_parametric[1]) if len(contexts_creo_parametric) > 1 else None,
            ]

            results = await asyncio.gather(*[task for task in tasks_2 if task is not None])

            for i, (rest, ctx) in enumerate(zip(results, contexts_video + contexts_wiki + contexts_error + contexts_email + contexts_creo_view + contexts_creo_parametric)):
                if rest[-1]:
                    n = len(context)
                    source_name = self.get_source_name(ctx)
                    combined_answer.append(f"\n\nAnswer from source: **{source_name}**\n{rest[0]}")
                    context.append(ctx)
                    if i < len(contexts_video):
                        for ts in rest[1]:
                            citations.append([ts, f"{n+1}"])
                    else:
                        citations.append(["", f"{n+1}"])
        else:
            tasks = [    
                self.search(query, 3, self.get_filter_query(rag_filter)),
                self.search(query, 2, self.get_filter_query("email")),
                ]
            contexts, emails = await asyncio.gather(*tasks)
            tasks_2 = [
                self.answer_document(query, contexts[0]) if len(contexts) > 0 else None,
                self.answer_document(query, contexts[1]) if len(contexts) > 1 else None,
                self.answer_document(query, contexts[2]) if len(contexts) > 2 else None,
                self.answer_email(query, emails[0]) if len(emails) > 2 else None,
                self.answer_email(query, emails[1]) if len(emails) > 2 else None,
            ]

            results = await asyncio.gather(*[task for task in tasks_2 if task is not None])

            for i, (rest, ctx) in enumerate(zip(results, contexts + emails)):
                if rest[-1]:
                    n = len(context)
                    source_name = self.get_source_name(ctx)
                    combined_answer.append(f"\n\nAnswer from source:{source_name} {rest[0]}")
                    context.append(ctx)
                    citations.append(["", f"{n+1}"])
            


        # # Map results to the context responses for easy access
        # contexts_map = [
        #     ("YES" if res else "NO", ctx) for res, ctx in zip(results, contexts_video[:2] + contexts_wiki[:2] + contexts_error + contexts_creo_view + contexts_creo_parametric)
        # ]


        # # Build the final context list with a maximum of 5 entries
        # context = []
        # for answer, ctx in contexts_map:
        #     if answer.upper() == "YES" and len(context) < 5:
        #         context.append(ctx)

        if len(context) < 1:
            combined_answer = ["There is no answer available from the source"]

        return combined_answer, citations, context

    async def get_prompt_message(self, query: str, top: int = 3, rag_filter = None) -> (List[Any], str):

        # if rag_filter == 'error': 
        #     contexts = await self.search(query, 3, self.get_filter_query(rag_filter))
        # else:
        combined_answer, citations, contexts = await self.run_parallel_searches(query, rag_filter)

        if len(contexts) > 3:
            combined_answer = combined_answer[:3]
            contexts = contexts[:3]
            citations = [sublist for sublist in citations if int(sublist[1]) <= 3]
        

        combined_answer_str = "\n".join(  
            f"""{answer_}""" for i, answer_ in enumerate(combined_answer)  
        )

        print(combined_answer_str)
        rag_user_query = f"""
Answer's from the different source.
------------------------------------------
{combined_answer_str}
------------------------------------------


**Query:** 
{query}
""" 
        rag_system_prompt = """
**Task:** Generate a comprehensive answer by synthesizing responses from all available sources.

1. **Process each source individually** following the specified **priority order**:
   - **Video** (highest priority)
   - **Wiki**
   - **Error Documents**
   - **Email Chain**
   - **Creo Parametric**
   - **Creo View** (lowest priority)

2. **Adhere strictly to the priority order** when crafting the final answer.

3. **Fallback Condition**: If none of the sources provide an answer, respond based on general knowledge, clarifying that no information was found in the provided sources.

4. Do not mentioned any sources name in the final answer.
"""
        messages = [{"role": "system", "content": rag_system_prompt},  
                      {"role": "user", "content": rag_user_query}]
        

        return contexts, messages, citations
    


    def extract_and_remove_lists(self, s):
        """
        Extracts all lists and lists of lists from the input string,
        combines them into a list of lists, and removes them from the string.

        Args:
            s (str): The input string containing lists.

        Returns:
            tuple: A tuple containing the combined list of lists and the modified string.
        """
        in_bracket = False
        bracket_level = 0
        in_string = False
        string_char = ''
        current_chunk = ''
        extracted_lists = []
        last_index = 0
        result_string = ''
        i = 0
        n = len(s)
        
        while i < n:
            c = s[i]
            if in_bracket:
                current_chunk += c
                if in_string:
                    if c == string_char and (i == 0 or s[i-1] != '\\'):
                        in_string = False
                else:
                    if c == '"' or c == "'":
                        in_string = True
                        string_char = c
                    elif c == '[':
                        bracket_level += 1
                    elif c == ']':
                        if bracket_level > 0:
                            bracket_level -= 1
                        else:
                            in_bracket = False
                            end_pos = i + 1
                            # Try to parse current_chunk
                            try:
                                # Process the list string to handle unquoted elements
                                list_data = self.process_list_string(current_chunk)
                                # Flatten the list if it's a list of lists
                                if isinstance(list_data, list):
                                    if all(isinstance(elem, list) for elem in list_data):
                                        extracted_lists.extend(list_data)
                                    else:
                                        extracted_lists.append(list_data)
                                    # Append text before the list
                                    result_string += s[last_index:start_pos]  # noqa: F821
                                    last_index = end_pos
                            except (SyntaxError, ValueError):
                                pass
                            current_chunk = ''
                    else:
                        pass
            else:
                if c == '[':
                    in_bracket = True
                    start_pos = i
                    current_chunk = c
                    bracket_level = 0
            i += 1

        # Append the remaining text
        result_string += s[last_index:]

        if extracted_lists == []:
            extracted_lists = [[]]

        return extracted_lists, result_string

    def process_list_string(self, list_str):
        """
        Processes a list string to ensure all elements are properly quoted.

        Args:
            list_str (str): The list string to process.

        Returns:
            list: The evaluated list with properly quoted elements.
        """
        # Pattern to match unquoted words (excluding commas, brackets, and whitespace)
        pattern = r'(?<=\[|,|\s)([^\s\[\],]+)(?=,|\s|\])'
        
        def replacer(match):
            word = match.group(1)
            # Check if the word is already quoted
            if not (word.startswith('"') and word.endswith('"')) and not (word.startswith("'") and word.endswith("'")):
                # Wrap unquoted elements with double quotes
                return '"' + word + '"'
            else:
                return word
        
        # Apply the pattern to replace unquoted elements
        processed_list_str = re.sub(pattern, replacer, list_str)
        # Safely evaluate the processed list string
        return ast.literal_eval(processed_list_str)
    
    async def openai_with_retry(self, messages, tools, user_json, max_retries=1):  
        retries = 0  
        while retries < max_retries:    
            azure_openai_client = await init_openai_client()  
            raw_rag_response = await azure_openai_client.chat.completions.with_raw_response.create(  
                model=self.chat_model,  
                messages=messages,  
                tools=tools,  
                temperature=0.05,  
                user=user_json  
            )  
            rag_response = raw_rag_response.parse()  
            apim_request_id = raw_rag_response.headers.get("apim-request-id")

            try:
                answer = rag_response.choices[0].message.content
                # print("rag_str_response", rag_str_response)
                return answer, apim_request_id 
            except Exception as e:
                retries += 1  
                print(f"Attempt {retries} failed: {e}")  
                if retries >= max_retries:  
                    return rag_response.choices[0].message.content, apim_request_id

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
                    print("type_", type_)
                    if type_ == "email":
                        title = "Email subject: "+title
                        timestamp_link = None
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

        contexts, messages, citations = await self.get_prompt_message(query, top, rag_filter)
        tools = None# tools = [openai.pydantic_function_tool(AnswerCitation)]
        answer, apim_request_id = await self.openai_with_retry(messages, tools, user_json, max_retries=3)
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
