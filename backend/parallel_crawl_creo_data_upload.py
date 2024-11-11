import os  
from azure.storage.blob.aio import BlobServiceClient  
from azure.identity.aio import DefaultAzureCredential  
import json
import re
from dotenv import load_dotenv
import asyncio 
load_dotenv()
  
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
            url = data.get("URL")   
            type_value = "creo_parametric"

            if type_value == "creo_view":
                if url.split('/')[5] == "view":
                    pass
                else:
                    raise ValueError(f"{url}\nurl does not belongs to creo_view")
            elif type_value == "creo_parametric":
                if url.split('/')[5] == "creo_pma":
                    pass
                else:
                    raise ValueError(f"{url}\nurl does not belongs to creo_parametric")
                    
                
            # Find the second key other than "URL"  
            for key in data:  
                if key != "URL":  
                    file_name = re.sub(r'[^A-Za-z0-9]', ' ', key) + ".txt"  
                    file_content = f"**{re.sub(r'[^A-Za-z0-9]', ' ', key)}\n\n**"+ data[key]  
                    break  
                
            # Metadata  
            metadata = {  
                'url_metadata': url,  
                'file_name_metadata': re.sub(r'[^A-Za-z0-9]', ' ', key),  
                'type': type_value,  
                'uploaded_by': "crawlers@ms.com"  
            }  

            # Create a blob client using the file name  
            blob_client = blob_service_client.get_blob_client(container=f"{container_name}/{type_value}", blob=file_name)  

            # Upload the file content as a blob  
            await blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)
            await blob_service_client.close()  
            await credential.close()
    
        except Exception as e:  
            error_message = f"Error: {str(e)}"  
            print(error_message)  
        finally:  
            await blob_service_client.close()  
            await credential.close()

            progress["uploaded"] += 1
            print(f"Files uploaded: {progress['uploaded']}/{progress['total']}")


async def file_upload_parallel(data_list):

    max_concurrent_uploads =   15  # Adjust based on system performance
    semaphore = asyncio.Semaphore(max_concurrent_uploads)

    progress = {"uploaded": 0, "total": len(data_list)}  
    
    tasks = [
        file_upload(data, semaphore, progress)
        for data in data_list
    ]

    await asyncio.gather(*tasks)
  
# Call the function with the data list  
  
def read_json_file(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        data_list = json.load(file)  
    return data_list  
  
# Example usage  
if __name__ == "__main__":   
  
    json_file_path = r"C:\Users\v-samomin\Downloads\CREO_PMA\CREO_PMA\CREO-PMA_Data.json"
    data_list = read_json_file(json_file_path)  
    asyncio.run(file_upload_parallel(data_list))