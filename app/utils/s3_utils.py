import boto3
import io
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_s3_client():
    """
    Create and return an S3 client
    """
    try:
        s3_client = boto3.client('s3')
        return s3_client
    except Exception as e:
        logger.error(f"Error creating S3 client: {e}")
        return None

def upload_to_s3(file_obj, bucket, object_name, content_type=None):
    """
    Upload a file to an S3 bucket
    
    :param file_obj: File object to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name
    :param content_type: Content type of the file
    :return: True if file was uploaded, else False
    """
    s3_client = get_s3_client()
    if not s3_client:
        return False
    
    try:
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
            
        s3_client.upload_fileobj(file_obj, bucket, object_name, ExtraArgs=extra_args)
        logger.info(f"Successfully uploaded {object_name} to {bucket}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error uploading to S3: {e}")
        return False

def download_from_s3(bucket, object_name):
    """
    Download a file from an S3 bucket
    
    :param bucket: Bucket name
    :param object_name: S3 object name
    :return: File content as BytesIO object or None if error
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None
        
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket, object_name, buffer)
        buffer.seek(0)
        logger.info(f"Successfully downloaded {object_name} from {bucket}")
        return buffer
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading from S3: {e}")
        return None

def list_s3_files(bucket, prefix=''):
    """
    List files in an S3 bucket
    
    :param bucket: Bucket name
    :param prefix: Prefix to filter results
    :return: List of object keys
    """
    s3_client = get_s3_client()
    if not s3_client:
        return []
        
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        # Check if there are any contents
        if 'Contents' not in response:
            return []
            
        return [obj['Key'] for obj in response['Contents']]
    except ClientError as e:
        logger.error(f"Error listing objects in S3: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing S3 objects: {e}")
        return []

def generate_presigned_url(bucket, object_name, expiration=3600):
    """
    Generate a presigned URL for an S3 object
    
    :param bucket: Bucket name
    :param object_name: S3 object name
    :param expiration: Expiration time in seconds
    :return: Presigned URL or None if error
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None
        
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                   Params={'Bucket': bucket,
                                                          'Key': object_name},
                                                   ExpiresIn=expiration)
        return response
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating URL: {e}")
        return None