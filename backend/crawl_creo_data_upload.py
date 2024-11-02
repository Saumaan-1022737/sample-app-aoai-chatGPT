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
    c = 0
  
    try:  
        for data in data_list:
            blob_service_client, credential = await get_blob_service_client()
            c = c+1
            print(c)
            url = data.get("URL")  
            # Extract type from URL  
            type_value = "creo_parametric"#"creo_parametric"#"creo_view" #url.split('/')[4] + "_" +  url.split('/')[6]
            # type_value = re.sub(r'[^a-zA-Z0-9]', '_', type_value)
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
  
    except Exception as e:  
        error_message = f"Error: {str(e)}"  
        print(error_message)  
    finally:  
        await blob_service_client.close()  
        await credential.close()   

def convert_to_version(version_str):
    version_num = int(version_str)
    major = version_num // 10
    minor = version_num % 10
    build = 0
    revision = 0
    return f"{major}.{minor}.{build}.{revision}"

def get_whats_new(data_list):
    whats_new = {}
    for data in data_list:
        if "whats_new" in data['URL']:
            for key in data:
                if key != "URL":
                    other_key = key
            try:
                version = data['URL'].split('/')[10]
                version_name = convert_to_version(version)
            except:
                if '/creo_pma/' in data['URL']:
                    version = data[other_key].split(" ")[2].split('\n')[0]
                    version_name = version


            if version in list(whats_new.keys()):
                whats_new[version]['text'] = whats_new[version]['text'] + f"\n- { whats_new[version]['counter']}. {other_key}"
                whats_new[version]['counter'] = whats_new[version]['counter'] + 1
            
            else:
                whats_new[version] = {'text': f"# What's New in _tool_name_placeholder_ {version_name}\n- 1. {other_key}",
                                    'counter': 1}
                
        elif ("/welcome/" in data['URL'] or "/introduction/" in data['URL']):
            if 'metadata_url' in locals():
                pass
            else:
                metadata_url = data['URL']
                if data['URL'].split('/')[8] == "creo_view":
                    metadata_type = "creo_view"
                    file_name_metadata = "whats new in Creo View"
                    file_name = "whats new in Creo View.txt"
                    tool_name = "Creo View"
                else:
                    metadata_type = "creo_parametric"
                    file_name_metadata = "whats new in Creo Parametric"
                    file_name = "whats new in Creo Parametric.txt"
                    tool_name = "Creo Parametric"

    metadata = {  
    'url_metadata': metadata_url,  
    'file_name_metadata': file_name_metadata,  
    'type': metadata_type,  
    'uploaded_by': "crawlers@ms.com"  
    } 

            
    whats_new_text = ""
    for i in whats_new:
        if whats_new_text == "":
            whats_new_text = whats_new_text + whats_new[i]['text']
        else:
            whats_new_text = whats_new_text +"\n\n\n"+whats_new[i]['text']
    whats_new_text = whats_new_text.replace("_tool_name_placeholder_", tool_name)

    whats_new_text_format = [{
        'URL': metadata_url,
        file_name_metadata: whats_new_text
    }]


    return whats_new_text_format

# # Call the function with the data list  
  
def read_json_file(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        data_list = json.load(file)  
    return data_list
  
# Example usage  
if __name__ == "__main__":  
    import asyncio  
  
    json_file_path = r"C:\Users\v-samomin\Downloads\CREO_PMA\CREO_PMA\CREO-PMA_Data.json" 
    data_list = read_json_file(json_file_path)
    data_list = get_whats_new(data_list) + data_list

    asyncio.run(file_upload(data_list))