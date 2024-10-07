import os  
from azure.storage.blob.aio import BlobServiceClient  
from azure.identity.aio import DefaultAzureCredential  
import json
import re
from dotenv import load_dotenv
load_dotenv()
  
async def get_blob_service_client():  
    credential = DefaultAzureCredential()  
    account_url = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net"  
    blob_service_client = BlobServiceClient(account_url, credential=credential)  
    return blob_service_client, credential  
  
async def file_upload(data_list):  
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")  
    blob_service_client, credential = await get_blob_service_client()  
  
    try:  
        for data in data_list:  
            url = data.get("URL")  
            # Extract type from URL  
            type_value = "creo_view" #url.split('/')[4] + "_" +  url.split('/')[6]
            # type_value = re.sub(r'[^a-zA-Z0-9]', '_', type_value)
              
            # Find the second key other than "URL"  
            for key in data:  
                if key != "URL":  
                    file_name = key + ".txt"  
                    file_content = data[key]  
                    break  
              
            # Metadata  
            metadata = {  
                'url_metadata': url,  
                'file_name_metadata': key,  
                'type': type_value,  
                'uploaded_by': "crawlers@ms.com"  
            }  
  
            # Create a blob client using the file name  
            blob_client = blob_service_client.get_blob_client(container=f"{container_name}/{type_value}", blob=file_name)  
  
            # Upload the file content as a blob  
            await blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)  
  
    except Exception as e:  
        error_message = f"Error: {str(e)}"  
        print(error_message)  
    finally:  
        await blob_service_client.close()  
        await credential.close()   
  
# Call the function with the data list  
  
def read_json_file(file_path):  
    with open(file_path, 'r') as file:  
        data_list = json.load(file)  
    return data_list  
  
# Example usage  
if __name__ == "__main__":  
    import asyncio  
  
    json_file_path = r'C:\Users\v-samomin\Downloads\extracted_data 2.json'  
    data_list = read_json_file(json_file_path)  
  
    asyncio.run(file_upload(data_list))  
