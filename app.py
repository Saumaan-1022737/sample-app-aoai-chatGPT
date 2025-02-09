import json
import os
import logging
import uuid
import asyncio
from quart import (Blueprint, Quart, jsonify, make_response, request, send_from_directory, flash, 
                   render_template, redirect, session, url_for,)
from azure.identity.aio import DefaultAzureCredential
from backend.auth.auth_utils import get_authenticated_user_details
from backend.utils import format_as_ndjson, format_stream_response
from azure.storage.blob.aio import BlobServiceClient
from backend.openai_client import init_openai_client
from backend.settings import app_settings
from backend.azure_rag import AzureSearchPromptService
from backend.resolver_group import find_resolver_group
from backend.openai_image_client import has_image, image_resolver
import asyncio
# from dotenv import load_dotenv
# load_dotenv()


async def get_blob_service_client():
    credential = DefaultAzureCredential()
    account_url = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net"
    # Create BlobServiceClient using the credential
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    return blob_service_client, credential

async def list_blobs_with_metadata(blob_service_client, credential, container_name):   
    container_client = blob_service_client.get_container_client(container_name)
    
    blob_list = []
    tasks = []
    # Define a semaphore to limit the number of concurrent operations
    semaphore = asyncio.Semaphore(15)  # Adjust the value as needed

    async for blob in container_client.list_blobs():
        if '/' not in blob.name:
            # Create a task for each blob
            task = asyncio.create_task(process_blob(blob, container_client, semaphore))
            tasks.append(task)
    
    # Gather results concurrently
    blob_list = await asyncio.gather(*tasks)
    return blob_list

async def process_blob(blob, container_client, semaphore):
    async with semaphore:
        metadata_ = {
            'name': blob.name,
            'url': None,
            'uploaded_by': None,
            'type': None,
            'uploaded_at': blob.last_modified.strftime("%d/%m/%Y %H:%M:%S")
        }
        
        blob_client = container_client.get_blob_client(blob)
        blob_properties = await blob_client.get_blob_properties()
        blob_metadata = blob_properties.metadata

        metadata_['url'] = blob_metadata.get('url_metadata')
        metadata_['uploaded_by'] = blob_metadata.get('uploaded_by')
        metadata_['type'] = blob_metadata.get('type')

        return metadata_


bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")



def create_app():
    app = Quart(__name__)
    app.secret_key = 'key'
    app.register_blueprint(bp)
    # app.config['PROVIDE_AUTOMATIC_OPTIONS'] = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    @app.before_serving
    async def init():
        app.cosmos_conversation_client = None
    
    return app


@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/file_edit", methods=["GET", "POST"])  
async def file_edit():  
    error_message = None 
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # auth_data = json.loads(base64.b64decode(authenticated_user['client_principal_b64']).decode('utf-8'))
    email_address = authenticated_user['user_name']
    session['email_address'] = email_address
    session['container_name_ui'] = os.getenv("AZURE_STORAGE_CONTAINER_NAME") 
    session['container_name'] = os.getenv("AZURE_STORAGE_CONTAINER_NAME") 
    if os.getenv("AZURE_STORAGE_CONTAINER_NAME") == "training-videos-testing":
        session['container_name_ui'] = "CAD KB"
    users = os.getenv("UPLOAD_USERS")
    if email_address in  users:
        if request.method == "POST":  
            form = await request.form       
            files = await request.files  
            file = files.get("file")  
            if file:  
                blob_service_client, credential = await get_blob_service_client()  
                try:  
                    container_name = session.get('container_name')  
                    if not container_name:  
                        error_message = "Please provide a correct container name."  
                    else:
                        metadata={'url_metadata': form['url'], 
                                  'file_name_metadata':form['file_name'],
                                  'type': form['type'], 'uploaded_by': email_address}

                        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)  
                        await blob_client.upload_blob(file.stream, overwrite=True, metadata=metadata)  
                except Exception as e:  
                    error_message = f"Error: {str(e)}"  
                finally:  
                    await blob_service_client.close()  
                    await credential.close()  

        container_name = session.get('container_name')  
        if not container_name:  
            error_message = "Please provide a container name."  
            return await render_template("file_edit.html", error_message=error_message, blobs=[])  
    
        blob_service_client, credential = await get_blob_service_client()  
        try:  
            blob_list = await list_blobs_with_metadata(blob_service_client, credential, container_name)

        except Exception as e:  
            error_message = f"Error: The specified container does not exist.\n\n{e}"  
            blob_list = []  
        finally:  
            await blob_service_client.close()  
            await credential.close()  
    
        return await render_template("file_edit.html", error_message=error_message, blobs=blob_list)
    else:
       return await render_template("un_auth.html", error_message="", blobs=[])   
  
@bp.route("/delete_file/<blob_name>", methods=["POST"])  
async def delete_file(blob_name):  
    container_name = session.get('container_name')  
    if not container_name:  
        flash("Please provide a container name.")  
        return redirect(url_for("routes.file_edit"))  
  
    blob_service_client, credential = await get_blob_service_client()  
    try:  
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)  
        await blob_client.delete_blob()  
    except Exception as e:  
        flash(f"Error: {str(e)}")  
    finally:
        await blob_service_client.close()  
        await credential.close()  
    return redirect(url_for("routes.file_edit")) 

@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)


frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and
        app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": False,
}


# Enable Microsoft Defender for Cloud Integration
MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"

async def prepare_model_args(request_body, request_headers):
    rag_filter = None
    request_messages = request_body.get("messages", [])
    if await has_image(request_messages[-1]):
        query = await image_resolver(request_body, request_headers)
        #rag_filter = "type eq 'error'"
        rag_filter = 'error'
    else:
        query = request_messages[-1]['content']
    azure_search_service = AzureSearchPromptService()
    answer = None
    actual_citations, answer, apim_request_id, user_json = await azure_search_service.rag(
                                                            query = query, 
                                                            top=3,
                                                            request_body=request_body,
                                                            request_headers=request_headers,
                                                            rag_filter= rag_filter)
    
    resolver_group = await find_resolver_group(query)
    resolver_list = ["pdmlink_admin@microsoft.com", "creo_help@microsoft.com", "surfswlic@microsoft.com", "deviceshelp@microsoft.com", "destasreredmond@microsoft.com"]
    resolver_string = ""

    if resolver_group != 'None' and not any(resolver in answer.lower() for resolver in resolver_list):
        resolver_string = f"""Always add this in response, "If the above information does not resolve your issue, please feel free to reach out for further assistance at {resolver_group}" at end"""
    
    if request_messages[-1]['role'] == 'user' and answer is not None:
        request_messages[-1]['content'] = f"""**query:** \n {query} \n\n\n **Answer from RAG:**\n {answer}"""
    messages = []
    if actual_citations != []:
        system_prompt = f"""
**Instruction for Generating and Formatting the Response**   
1. **Answer from RAG** is a correct answer to unser's query, therefore in response just use re-write **Answer from RAG** without referencing other sources or prior knowledge.  
2. Re-Write **Answer from RAG** in a step-by-step format, enhancing readability and making it more engaging without altering the original content. Structure each step clearly, using bullet points or numbered steps if appropriate.
3. if **Answer from RAG** contains this "the context provided does not provided the specific detais", "There is no answer available" or similar then do not re-write **Answer from RAG**, Just state There isn't an available answer at the moment, but I've included a few articles in the citations that may be helpful or something similar in step by step format 

{resolver_string}
"""
    else:
        system_prompt = f"""For every user query you always give this response 'There is no answer available', except Hi, hello, why am I here, query about you, greeting queries or similar.

Example 1:
Query: Hi.
Your response: Hello! How can I assist you today?

Example 2:
Query: How to play football?
Your response: There is no answer available

Example 3:
Query: What do you do?
Your response: I assist with a wide range of tasks, from **answering questions** and **providing guidance** to helping with **technical issues**, and more.

Example 4:
Query: My pc is running slow?
Your response: **There is no answer available**


Example 5:
Query: Why am I here?
Your response: You're here because you might need assistance with **technical** or **IT-related** questions, particularly those related to **design**, **tools**, or **troubleshooting** within your work environment. \n\nLet me know if there's something specific I can help you with!

Example 6:

Your response: 
I can assist you by:
1. **Answering Queries on Document Design and Tools**
2. **Technical Support**
3. **Best Practices and Workflow Guidance**
Let me know how I can specifically help with your current project or challenge!

{resolver_string}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for message in request_messages:
        if message:
            if message["role"] == "assistant" and "context" in message:
                context_obj = json.loads(message["context"])
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "context": context_obj
                    }
                )
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                )
    model_args = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 8000,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,
        "user": user_json
    }

    return model_args, actual_citations, apim_request_id

async def send_chat_request(request_body, request_headers):
    filtered_messages = []
    messages = request_body.get("messages", [])

    for message in messages:
        if message.get("role") != 'tool':
            filtered_messages.append(message)
            
    request_body['messages'] = filtered_messages
    model_args, actual_citations, apim_request_id = await prepare_model_args(request_body, request_headers)
    session['content_mapping'] = actual_citations
    try:
        azure_openai_client = await init_openai_client()
        raw_response = await azure_openai_client.chat.completions.with_raw_response.create(**model_args)
        response = raw_response.parse()
        apim_request_id = raw_response.headers.get("apim-request-id") 
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e
    return response, apim_request_id

async def stream_chat_request(request_body, request_headers):
    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})
    
    async def generate():
        async for completionChunk in response:
            yield format_stream_response(completionChunk, history_metadata, apim_request_id)
    return generate()


async def conversation_internal(request_body, request_headers):
    try:
        result = await stream_chat_request(request_body, request_headers)
        response = await make_response(format_as_ndjson(result))
        response.timeout = None
        response.mimetype = "application/json-lines"
        return response
    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500


@bp.route('/api/content-mapping', methods=['GET'])
async def get_content_mapping():
    mapping = session.get('content_mapping', [])
    return jsonify(mapping)

@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()

    result = await conversation_internal(request_json, request.headers)
    return result
 
  
@bp.route('/test', methods=['POST', 'GET'])  
def test_endpoint():
    print('123')
    return jsonify(message="Working")  


@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/ensure", methods=["GET"])
async def ensure_cosmos():
    if not app_settings.chat_history:
        return jsonify({"error": "CosmosDB is not configured"}), 404
    return jsonify({"message": "CosmosDB is configured and working"}), 200


app = create_app()
