import json
import os
import logging
import uuid
import asyncio
from quart import (Blueprint, Quart, jsonify, make_response, request, send_from_directory, flash, 
                   render_template, current_app, redirect, session, url_for,)
from azure.identity.aio import (DefaultAzureCredential)
from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.utils import format_as_ndjson, format_stream_response
from azure.storage.blob.aio import BlobServiceClient
from backend.openai_client import init_openai_client
from backend.settings import app_settings
from backend.azure_rag import AzureSearchPromptService
from backend.resolver_group import find_resolver_group
from backend.openai_image_client import has_image, image_resolver
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
    async for blob in container_client.list_blobs():
        if '/' not in blob.name: 
            metadata_ = {'name': None,
                        'url':None,
                        'uploaded_by': None,
                        'type':None}
            blob_client = container_client.get_blob_client(blob)  
            blob_properties = await blob_client.get_blob_properties()  
            blob_metadata = blob_properties.metadata
            
            metadata_['name'] = blob.name
            metadata_['uploaded_at'] = blob.last_modified.strftime("%d/%m/%Y %H:%M:%S")
            if 'url_metadata' in blob_metadata:
                metadata_['url'] = blob_metadata['url_metadata']
            if 'uploaded_by' in blob_metadata:
                metadata_['uploaded_by'] = blob_metadata['uploaded_by']
            if 'type' in blob_metadata:
                metadata_['type'] = blob_metadata['type']
                
            blob_list.append(metadata_)  
      
    return blob_list 


bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

cosmos_db_ready = asyncio.Event()


def create_app():
    app = Quart(__name__)
    app.secret_key = 'key'
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e
    
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


async def init_cosmosdb_client():
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = (
                f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            )

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred
                    
            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            cosmos_conversation_client = None
            raise e
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client


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
    resolver_string = ""

    if resolver_group != 'None':
        resolver_string = f"""Always add this in response, "If the above information does not resolve your issue, please feel free to reach out for further assistance at {resolver_group}" at end"""
    
    if request_messages[-1]['role'] == 'user' and answer is not None:
        request_messages[-1]['content'] = f"""**query:** \n {query} \n\n\n **Answer from RAG:**\n {answer}"""
    messages = []
    if actual_citations != []:
        system_prompt = f"""
**Instruction for Generating and Formatting the Response**   
1. **Answer from RAG** is a correct answer to unser's query, therefore in response just use re-write **Answer from RAG** without referencing other sources or prior knowledge.  
2. Re-Write **Answer from RAG** in step by step format.
3. if **Answer from RAG** contains this "the context provided does not provided the specific detais", "There is no answer available" or similar then do not re-write **Answer from RAG**, Just state There isn't an available answer at the moment, but I've included a few articles in the citations that may be helpful or something similar in step by step format 

{resolver_string}
"""
    else:
        system_prompt = f"""For every user query you always give this response 'There is no answer available', except Hi, hello, query about you and greeting queries

Example 1:
Query: Hi.
Your response: Hello! How can I assist you today?

Example 2:
Query: How to play football?
Your response: There is no answer available

Example 3:
Query: What do you do?
Your response: I assist with a wide range of tasks, from answering questions and providing guidance to helping with technical issues, brainstorming ideas, and more

Example 4:
Query: My pc is running slow?
Your response: There is no answer available



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


@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


## Conversation History API ##
@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = await current_app.cosmos_conversation_client.create_conversation(
                user_id=user_id, title=title
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "user":
            createdMessageValue = await current_app.cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
            if createdMessageValue == "Conversation not found":
                raise Exception(
                    "Conversation not found for the given conversation ID: "
                    + conversation_id
                    + "."
                )
        else:
            raise Exception("No user message found")

        # Submit request to Chat Completions for response
        request_body = await request.get_json()
        history_metadata["conversation_id"] = conversation_id
        request_body["history_metadata"] = history_metadata
        return await conversation_internal(request_body, request.headers)

    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/update", methods=["POST"])
async def update_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "assistant":
            if len(messages) > 1 and messages[-2].get("role", None) == "tool":
                # write the tool message first
                await current_app.cosmos_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2],
                )
            # write the assistant message
            await current_app.cosmos_conversation_client.create_message(
                uuid=messages[-1]["id"],
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
        else:
            raise Exception("No bot messages found")

        # Submit request to Chat Completions for response
        response = {"success": True}
        return jsonify(response), 200

    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/message_feedback", methods=["POST"])
async def update_message():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for message_id
    request_json = await request.get_json()
    message_id = request_json.get("message_id", None)
    message_feedback = request_json.get("message_feedback", None)
    try:
        if not message_id:
            return jsonify({"error": "message_id is required"}), 400

        if not message_feedback:
            return jsonify({"error": "message_feedback is required"}), 400

        ## update the message in cosmos
        updated_message = await current_app.cosmos_conversation_client.update_message_feedback(
            user_id, message_id, message_feedback
        )
        if updated_message:
            return (
                jsonify(
                    {
                        "message": f"Successfully updated message with feedback {message_feedback}",
                        "message_id": message_id,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."
                    }
                ),
                404,
            )

    except Exception as e:
        logging.exception("Exception in /history/message_feedback")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos first
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        ## Now delete the conversation
        deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
            user_id, conversation_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted conversation and messages",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/list", methods=["GET"])
async def list_conversations():
    await cosmos_db_ready.wait()
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversations from cosmos
    conversations = await current_app.cosmos_conversation_client.get_conversations(
        user_id, offset=offset, limit=25
    )
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    ## return the conversation ids

    return jsonify(conversations), 200


@bp.route("/history/read", methods=["POST"])
async def get_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation object and the related messages from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    # get the messages for the conversation from cosmos
    conversation_messages = await current_app.cosmos_conversation_client.get_messages(
        user_id, conversation_id
    )

    ## format the messages in the bot frontend format
    messages = [
        {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"],
            "feedback": msg.get("feedback"),
        }
        for msg in conversation_messages
    ]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200


@bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    ## update the title
    title = request_json.get("title", None)
    if not title:
        return jsonify({"error": "title is required"}), 400
    conversation["title"] = title
    updated_conversation = await current_app.cosmos_conversation_client.upsert_conversation(
        conversation
    )

    return jsonify(updated_conversation), 200


@bp.route("/history/delete_all", methods=["DELETE"])
async def delete_all_conversations():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    # get conversations for user
    try:
        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        conversations = await current_app.cosmos_conversation_client.get_conversations(
            user_id, offset=0, limit=None
        )
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404

        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
                conversation["id"], user_id
            )

            ## Now delete the conversation
            deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
                user_id, conversation["id"]
            )
        return (
            jsonify(
                {
                    "message": f"Successfully deleted conversation and messages for user {user_id}"
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/clear", methods=["POST"])
async def clear_messages():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted messages in conversation",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/ensure", methods=["GET"])
async def ensure_cosmos():
    await cosmos_db_ready.wait()
    if not app_settings.chat_history:
        return jsonify({"error": "CosmosDB is not configured"}), 404

    try:
        success, err = await current_app.cosmos_conversation_client.ensure()
        if not current_app.cosmos_conversation_client or not success:
            if err:
                return jsonify({"error": err}), 422
            return jsonify({"error": "CosmosDB is not configured or not working"}), 500

        return jsonify({"message": "CosmosDB is configured and working"}), 200
    except Exception as e:
        logging.exception("Exception in /history/ensure")
        cosmos_exception = str(e)
        if "Invalid credentials" in cosmos_exception:
            return jsonify({"error": cosmos_exception}), 401
        elif "Invalid CosmosDB database name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception} {app_settings.chat_history.database} for account {app_settings.chat_history.account}"
                    }
                ),
                422,
            )
        elif "Invalid CosmosDB container name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception}: {app_settings.chat_history.conversations_container}"
                    }
                ),
                422,
            )
        else:
            return jsonify({"error": "CosmosDB is not working"}), 500


async def generate_title(conversation_messages) -> str:
    ## make sure the messages are sorted by _ts descending
    title_prompt = "Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Do not include any other commentary or description."

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_messages
    ]
    messages.append({"role": "user", "content": title_prompt})

    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model, messages=messages, temperature=1, max_tokens=64
        )

        title = response.choices[0].message.content
        return title
    except Exception as e:
        logging.exception("Exception while generating title", e)
        return messages[-2]["content"]


app = create_app()
