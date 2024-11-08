import os  
from azure.storage.blob.aio import BlobServiceClient  
from azure.identity.aio import DefaultAzureCredential  
import json
import re
from dotenv import load_dotenv
import asyncio 
from datetime import datetime
load_dotenv()


def custom_encoder(obj):  
        if isinstance(obj, datetime):  
            return obj.isoformat()  # Convert datetime to ISO 8601 string  
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def custom_decoder(dct):
    for key, value in dct.items():
        if key == 'time_py' and isinstance(value, str):
            try:
                dct[key] = datetime.fromisoformat(value)
            except ValueError:
                pass  # If it's not a valid ISO format, leave it as is
    return dct
  
async def get_blob_service_client():  
    credential = DefaultAzureCredential()  
    account_url = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net"  
    blob_service_client = BlobServiceClient(account_url, credential=credential)  
    return blob_service_client, credential  
  
async def file_upload(data, semaphore, progress):
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    async with semaphore:
        try:  
            
            blob_service_client, credential = await get_blob_service_client()

            type_value = "email"

            subject = data['subject'].replace("RE:", "").replace("re:", "").replace("Re:", "").replace("[EXTERNAL]", "").strip()
            url = f"Subject: {re.sub(r'[^A-Za-z0-9]', ' ', subject)}" 
            file_name = re.sub(r'[^A-Za-z0-9]', ' ', subject) + ".txt"
            file_content = f"**Issue discuss in the email:** {data['issue']} \n\nEmail Subject: {subject}\n\n**"+ data['email_content']  
            
            # Metadata  
            metadata = {  
                'url_metadata': url,  
                'file_name_metadata': file_name,  
                'type': type_value,  
                'uploaded_by': "email@ms.com"  
            }  

            # Create a blob client using the file name  
            blob_client = blob_service_client.get_blob_client(container=f"{container_name}/{type_value}", blob=file_name)  

            # Upload the file content as a blob  
            await blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)
            await blob_service_client.close()  
            await credential.close() 

            progress["uploaded"] += 1
            print(f"Files uploaded: {progress['uploaded']}/{progress['total']}")

    
        except Exception as e:  
            error_message = f"Error: {str(e)}"  
            print(error_message)  
        finally:  
            await blob_service_client.close()  
            await credential.close()   
    
async def file_upload_parallel(data_list):

    max_concurrent_uploads = 10  # Adjust based on system performance
    semaphore = asyncio.Semaphore(max_concurrent_uploads)

    progress = {"uploaded": 0, "total": len(data_list)}  
    
    tasks = [
        file_upload(data, semaphore, progress)
        for data in data_list
    ]

    await asyncio.gather(*tasks)
 
  
def read_json_file(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file, object_hook=custom_decoder)  
    return data_list
# Example usage  
if __name__ == "__main__":  
     
  
    json_file_path = r"C:\Users\v-samomin\Desktop\git\sample-app-aoai-chatGPT\backend\email\processed_emails.json"
    data_list = read_json_file(json_file_path)  
    asyncio.run(file_upload_parallel(data_list))