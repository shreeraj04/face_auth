# All the configurations related to SQL, paths and logs are done here.

version = "1.0.8.17"
sourceFaceImage = "//"  # source face image path to be saved
destFaceImage = "//"  # destination face image path to be saved Profile Picture path
sourceFaceImageURL = (
    "http://localhost:50021/Images/"  # service running url + folder for Source Img
)
destFaceImageURL = "http://localhost:50021/Images/DestImage/"  # servie running url + folder for dest Image
server = "demo.tetherfilabs.com\SQLExpress,14330"  # sql server name
database = "OCM"  # sql server database name
username = "sa"  # sql server username
# password = "D3vDB@#$%2020" # sql server password
is_encrypted_password = False  # True if password given is encrypted else False
password = "password"  # normal or encrypted password
cipher_key = "lPEKW0MmuZhAeH3JNPCZXOoc5sN9H1KkzRVEh7Wt6ag=@#"  # cipher key used during encryption
port = 50021  # Port on which it is listening as a standalone app (not in Gunicorn, for gunicorn you have to explicity mention in service file)
detect_liveness = True  # True - the system will detect if the image in snap_data is real or fake image. False- ignore the liveness
face_liveness_threshold = 0.75
# directory_browsing_proPic = "//"
# directory_browsing_destImage = "//"
