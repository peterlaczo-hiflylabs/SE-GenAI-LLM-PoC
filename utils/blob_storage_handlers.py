from azure.storage.blob import BlobServiceClient


def connect_to_storage(account_name, key):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={key};EndpointSuffix=core.windows.net")
    return blob_service_client

def list_files_in_container(blob_service_client, container_name):
    client = blob_service_client.get_container_client(container_name)
    blob_list = client.list_blobs()
    return [blob for blob in blob_list]

def select_blob_file(blob_service_client, container_name, blob):
    client = blob_service_client.get_container_client(container_name)
    blob_file = client.get_blob_client(blob)
    return (blob_file.download_blob()).readall().decode("utf-8")

def upload_to_blob_storage(blob_service_client, container_name, blob_path, content):
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
        blob_client.upload_blob(content, overwrite=True)
        return True, None
    except Exception as e:
        return False, str(e)