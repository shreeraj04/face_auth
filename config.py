# All the configurations related to SQL, paths and logs are done here.
version = "1.0.8.17"
# Below not applicable in v2
sourceFaceImage = "D:\\GIT\\TFaceAuth\\tfacerecognition\\Linux\\TFaceAuthServer\\Images\\"  # source face image path to be saved
destFaceImage = "D:\\GIT\\TFaceAuth\\tfacerecognition\\Linux\\TFaceAuthServer\\Images\\DestImage\\"  # destination face image path to be saved Profile Picture path
# Above not applicable in v2
sourceFaceImageURL = (
    "http://localhost:50021/Images/"  # service running url + folder for Source Img
)
destFaceImageURL = "http://3.12.124.74:55080/FaceImagesDst/"  # servie running url + folder for dest Image (ProfilePic)
# Below are SQL configurations
server = "demo2.tetherfilabs.com\SQLExpress,14330"  # sql server name
database = "TetherfiHome"  # sql server database name
username = "sa"  # sql server username
is_encrypted_password = False  # True if password given is encrypted else False
password = "Qwerty@123"  # normal or encrypted password
cipher_key = "lPEKW0MmuZhAeH3JNPCZXOoc5sN9H1KkzRVEh7Wt6ag=@#"  # cipher key used during encryption
# Above are SQL Configurations
port = 50021  # Port on which it is listening as a standalone app (not in Gunicorn, for gunicorn you have to explicity mention in service file)
detect_liveness = True  # True - the system will detect if the image in snap_data is real or fake image. False- ignore the liveness
face_liveness_threshold = 0.75 # Liveness Threshold
is_load_profilepic_onstartup = True # Load profile Pic from OCM DB and save to SQLite if set to True
is_profilepic_url = False # Get ProfilePic from the URL for each request