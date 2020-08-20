from aiohttp import web
import io
import base64
import math
import json
import face_recognition
import math
from PIL import ImageOps, Image, ExifTags
import re
import dateutil.parser as dparser
from PIL import Image
import os
import unicodedata
import urllib
import base64
import logging
import logging.handlers as handlers
from logging.handlers import RotatingFileHandler
import config
import sql
import threading
import datetime
import uuid
import tensorflow.keras
import dlib
import cv2
import numpy as np
from skimage import io as iosk
import sqlite3
from collections import defaultdict
import aiohttp_cors
from aiohttp_cors import setup as cors_setup, ResourceOptions
import asyncio
import collections

model = tensorflow.keras.models.load_model("keras_model.h5")
logging.basicConfig(
    handlers=[RotatingFileHandler(
        "faceauth.log", maxBytes=10485760, backupCount=10)],
    level=logging.DEBUG,
    format='"%(asctime)s -[Face Recogntion] - %(levelname)-7.7s %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Path in server where Snapshot of the video is saved
# This image is agent or customer snapshot image from video.
# For Remote Survillience, it should be agent self video image
# For EKYC process, it should be cutomer image from live video.
sourceFaceImage = config.sourceFaceImage

# Path in the server where aadhar image is save
# This image is agent profile picture or any IC image like aadhar/EP/Pan card etc.
# For Remote Survillience, it should be agent profile picture
# For EKYC process, it should be any IC image like aadhar/EP/Pan card etc uploaded by customer.
destFaceImage = config.destFaceImage
# logging.info('GPU')
# logging.info(cuda.get_num_devices());

face_auth_trasaction_details = (
    []
)  # face_found, Result, Result_reason, Score, Confidence (NOT USED)
requested_datetime = None
responded_datetime = None
image_path1 = None
image_path2 = None
originator = ""
profilePic_encodingData = defaultdict(collections.Counter)


async def decodeImage(msg):
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img


async def get_face_encoding_from_base64(base64String):
    nparr = np.frombuffer(base64.b64decode(
        base64String.encode("utf-8")), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_encodeing = face_recognition.face_encodings(
        rgb_img, None, 1, "large")
    # print(image_encodeing)
    return image_encodeing


async def allowed_file(logFileName):
    return (
        "." in logFileName
        and logFileName.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def upload_image():
    """
    This is the test method which accepts any 2 images and gives out the comparision result
    """
    default_UploadPath = sourceFaceImage
    default_AadharPath = destFaceImage
    logging.info("Upload_Image")
    # Check if a valid image file was uploaded
    if request.method == "POST":
        if "file1" not in request.files:
            logging.info("test1")
            return redirect(request.url)

        file = request.files["file1"]
        file2 = request.files["file2"]

        logging.info(str(file))
        logging.info(str(file2))

        if file.filename == "":
            logging.info("Filename not found")
            return redirect(request.url)
        requested_datetime = datetime.datetime.now()

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return DetectAndCompareFacesInImages(
                file, file2, requested_datetime, "test", "test", "", ""
            )

    # If no valid image file was uploaded, show the file upload form:
    return """
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload pics and get the comparision results</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="file" name="file2">
      <input type="submit" value="Upload">
    </form>
    """


async def DetectAndCompareFacesInImages(
    sourceFileStream,
    destFileStream,
    requested_datetime,
    agentId,
    originator_name,
    source_url,
    destination_url,
):
    """
    Method to detect the faces in the image passed and do the comparision of those
    INPUT: 2 image paths 
    OUTPUT: json data of mathched information
    """
    try:
        # image1 = iosk.imread(destFileStream)
        # sourceFaceEncodingValue = np.asarray(destFileStream)
        # destFaceEncodingValue = [np.array(destFileStream)]
        sourceFaceEncodingValue = face_recognition.face_encodings(
            face_recognition.load_image_file(sourceFileStream)
        )
        destFaceEncodingValue = face_recognition.face_encodings(
            face_recognition.load_image_file(destFileStream)
        )

        # enable the below logs if you want face encodings to be displyed in logs
        # logging.info("Image1 face encodings " + str(sourceFaceEncodingValue))
        # logging.info("Image2 face encodings " + str(destFaceEncodingValue))

        if len(sourceFaceEncodingValue) <= 0:
            logging.info(
                agentId + ":Face not found in source image " + sourceFileStream
            )
            return ProcessExceptionData(
                agentId,
                originator_name,
                "Face Not Found in Source Image",
                source_url,
                destination_url,
                requested_datetime,
                datetime.datetime.now(),
            )
        if len(destFaceEncodingValue) <= 0:
            logging.info(
                agentId + ":Face not found in destination image " + destFileStream
            )
            return ProcessExceptionData(
                agentId,
                originator_name,
                "Face Not Found in Destination Image",
                source_url,
                destination_url,
                requested_datetime,
                datetime.datetime.now(),
            )

        faceFound = False
        fpercentage = 0.0

        if len(sourceFaceEncodingValue) > 0 and len(destFaceEncodingValue) > 0:
            logging.info(
                agentId
                + ":DetectAndCompareFacesInImages : Face found in both source & destination images"
            )
            faceFound = True
            # See if the first face in the uploaded image matches the known face of Obama
            match_results = face_recognition.compare_faces(
                [sourceFaceEncodingValue[0]], destFaceEncodingValue[0]
            )
            face_distances = face_recognition.face_distance(
                [sourceFaceEncodingValue[0]], destFaceEncodingValue[0]
            )
            logging.info(
                agentId
                + ":DetectAndCompareFacesInImages : Face distance "
                + str(face_distances)
            )

            fpercentage = await GetFaceComparisionPercentage(
                face_distances, agentId, 0.6
            )
            logging.info(
                agentId
                + ":DetectAndCompareFacesInImages : Percentage of face match "
                + str(fpercentage[0])
            )

        # Return the comparisionResult as json
        comparisionResult = {
            "face_found_in_image": faceFound,
            "face_authenticated_percentage": fpercentage[0] * 100,
            "face_isreal": 1,
        }
        # image.close()
        responded_datetime = datetime.datetime.now()
        threading.Thread(
            target=sql.save_faceauth_data,
            args=(
                requested_datetime,
                responded_datetime,
                source_url,
                destination_url,
                originator_name,
                agentId,
                "True",
                "Face found in Image",
                str(fpercentage[0] * 100),
                str(face_distances),
            ),
        ).start()

        return comparisionResult
    except Exception as ex:
        logging.info(
            agentId + ":Exception in DetectAndCompareFacesInImages : " +
            str(ex)
        )

        return await ProcessExceptionData(
            agentId,
            originator_name,
            "Exception in DetectAndCompareFacesInImages",
            source_url,
            destination_url,
            requested_datetime,
            datetime.datetime.now(),
        )


async def GetFaceComparisionPercentage(
    face_distance, agentId, face_match_threshold=0.6
):
    """
    This method will give the percentage of match between the 2 image encodings with respect to the threshold
    """
    logging.info(
        agentId
        + ":GetFaceComparisionPercentage Face Distance"
        + str(face_distance)
        + ", Threshold "
        + str(face_match_threshold)
    )

    try:
        if face_distance > face_match_threshold:
            range = 1.0 - face_match_threshold
            linear_val = (1.0 - face_distance) / (range * 2.0)
            return linear_val
        else:
            range = face_match_threshold
            linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + (
                (1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2)
            )
    except Exception as ex:
        logging.info("Exception in GetFaceComparisionPercentage : " + str(ex))


async def VerifyFaceImages(self, *, loads=json.loads):
    """
    Method to verify face images requested by client app
    INPUT: snaptype (url,base64), snapdata, pptype(url,base64), ppdata, agentid, orginator
               ppType means - ProfilePIC
    OUTPUT: json data of mathched information
    """
    try:
        json_req = await self.text()
        json_req = json.loads(json_req)
        requested_datetime = datetime.datetime.now()
        defaultSourceFilePath = sourceFaceImage
        defaultDestFilePath = destFaceImage

        agent_id = json_req["agentId"]
        originator_name = json_req["originator"]
        snap_data = json_req["snapdata"]
        pp_data = json_req["ppdata"]
        snap_type = json_req["snaptype"]
        pp_type = json_req["pptype"]
        isrealface = json_req["isrealface"]

        # return web.json_response({})
        logging.info(
            agent_id
            + ":VerifyFaceImages Request: snaptype: "
            + snap_type
            + ", pptype: "
            + pp_type
            + ", agentId: "
            + agent_id
            + ", ppdata: "
            + pp_data
            + ", originator: "
            + originator_name
            + ", isRealFace: "
            + isrealface
        )

        # Check if snapdata or ppdata or agentId is empty. If empty return back with exception to the client
        if snap_data == "" or pp_data == "" or agent_id == "":
            logging.info(
                agent_id
                + ":VerifyFaceImages snapdata: "
                + snap_data
                + " ppdata: "
                + pp_data
                + " agentId: "
                + agent_id
            )
            return ProcessExceptionData(
                agent_id,
                originator_name,
                "SnapData or PpData or AgentId is empty",
                "",
                "",
                datetime.datetime.now(),
                datetime.datetime.now(),
            )

        # check if the originator has value or not. If not, assing TMAC to it
        if originator_name:
            originator = originator_name
        else:
            originator = "TMAC"

        if snap_type == "url":
            defaultSourceFilePath = defaultSourceFilePath + snap_data
            source_filename = snap_data
        else:
            # base64
            if snap_data != "":
                datetime_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                source_filename = agent_id + "_" + datetime_stamp + ".png"
                defaultSourceFilePath = defaultSourceFilePath + source_filename

                with open(defaultSourceFilePath, "wb") as fh:
                    fh.write(base64.b64decode(snap_data))

            # detect if this image is a real or fake image based on the configuration set
            if isrealface != "1":
                logging.info(
                    agent_id + ": isRealFace is 0. Check for liveness in the image"
                )
                if config.detect_liveness:
                    logging.info(
                        agent_id + " :detect Liveness is enabled in the config"
                    )
                    isReal = await detect_liveness(defaultSourceFilePath, False, False)
                    if not isReal:
                        logging.info(
                            agent_id + ": isReal is 0. Fake Image. Return to the client"
                        )
                        retData = {
                            "face_found_in_image": 0,
                            "face_authenticated_percentage": 0,
                            "face_isreal": 0,
                        }
                        return web.json_response(retData)

        if pp_type == "url":
            destination_filename = pp_data
            defaultDestFilePath = defaultDestFilePath + destination_filename
        else:
            # base64
            uuid_datetime = (
                str(uuid.uuid4())
                + "_"
                + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            )
            destination_filename = uuid_datetime + ".png"
            defaultDestFilePath = defaultDestFilePath + destination_filename
            with open(defaultDestFilePath, "wb") as fh:
                fh.write(base64.b64decode(pp_data))

        dest_image_url_path = config.destFaceImageURL
        source_image_url_path = config.sourceFaceImageURL

        return_data = await DetectAndCompareFacesInImages(
            defaultDestFilePath,
            defaultSourceFilePath,
            requested_datetime,
            agent_id,
            originator,
            dest_image_url_path,
            source_image_url_path,
        )
        logging.info(agent_id + ":VerifyFaceImages Response: " +
                     str(return_data))

        return web.json_response(return_data)

    except Exception as ex:
        logging.info(agent_id + ":Exception in VerifyFaceImages : " + str(ex))

        return await ProcessExceptionData(
            agent_id,
            originator_name,
            "Exception in VerifyFaceImages",
            "",
            "",
            datetime.datetime.now(),
            datetime.datetime.now(),
        )


async def ProcessExceptionData(
    agentId,
    originator_name,
    exception_message,
    source_url,
    destination_url,
    request_datetime,
    response_datetime,
):
    """
    This method is a generic method whenever there is an exception raised. Accepts the message and other parameters and returns back the client
    """
    try:
        logging.info(agentId + ":ProcessExceptionData")
        comparisionResult = {
            "face_found_in_image": 0,
            "face_authenticated_percentage": 0,
            "face_isreal": 0,
        }

        threading.Thread(
            target=sql.save_faceauth_data,
            args=(
                request_datetime,
                response_datetime,
                source_url,
                destination_url,
                originator_name,
                agentId,
                "False",
                exception_message,
                "0",
                "0",
            ),
        ).start()
        logging.info(
            agentId + ":ProcessExceptionData Response: " +
            str(comparisionResult)
        )
        return web.json_response(comparisionResult, status=500)
    except Exception as identifier:
        logging.info(
            agentId + ":Exception in ProcessExceptionData" + str(identifier))
        comparisionResult = {
            "face_found_in_image": 0,
            "face_authenticated_percentage": 0,
        }
        return web.json_response(comparisionResult, status=500)


async def detect_faces(image, name):
    try:
        logging.info("detect_faces " + str(name))
        # Create a face detector
        face_detector = dlib.get_frontal_face_detector()

        # Run detector and get bounding boxes of the faces on image.
        detected_faces = face_detector(image, 1)
        logging.info("Number of faces detected: {}".format(
            len(detected_faces)))
        for i, d in enumerate(detected_faces):
            logging.info(
                "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    i, d.left(), d.top(), d.right(), d.bottom()
                )
            )
            crop = image[d.top(): d.bottom(), d.left(): d.right()]
            resize = cv2.resize(crop, (200, 200))
            return resize
    except Exception as identifier:
        logging.info("Exception in detect_faces " + str(identifier))


async def detect_liveness(imagename, isReturnJson, isUrl):
    try:
        logging.info("detect_liveness" + str(imagename))
        if isUrl:
            image = iosk.imread(imagename)
        else:
            image = dlib.load_rgb_image(imagename)
        detected_faces = await detect_faces(image, imagename)
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        # image = Image.open(imagename)
        image = Image.fromarray(np.uint8(detected_faces)).convert("RGB")
        # image = detected_faces

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        logging.info(prediction)
        real = prediction[0][0]
        fake = prediction[0][1]
        if isReturnJson:
            if abs(real) >= config.face_liveness_threshold:
                retData = {"isReal": "Real Face"}
                return retData
            elif abs(fake) >= config.face_liveness_threshold:
                retData = {"isReal": "Fake Face"}
                return retData
            else:
                retData = {"isReal": "Not Sure"}
                return retData
        if abs(real) >= config.face_liveness_threshold:
            return True
        else:
            return False
        return real, fake
    except Exception as identifier:
        logging.info("Exception in detect_liveness " + str(identifier))


async def VerifyFaceAuthWithEncoding(self, *, loads=json.loads):
    """
    Method to verify face images requested by client app
    INPUT: snap_encoded_data, isrealface, snap_data_url, ppdata, agentid, orginator
    OUTPUT: json data of mathched information
    """
    try:
        requested_datetime = datetime.datetime.now()
        json_req = await self.text()
        json_req = json.loads(json_req)
        requested_datetime = datetime.datetime.now()
        defaultSourceFilePath = sourceFaceImage
        defaultDestFilePath = destFaceImage

        agent_id = json_req["agentId"]
        originator_name = json_req["originator"]
        snap_encoded_data = json_req["snapEncodedData"]
        isrealface = json_req["isRealFace"]
        snap_data_url = json_req["snapDataUrl"]

        logging.info(
            agent_id
            + ":VerifyFaceAuthWithEncoding Request: "
            + "agentId: "
            + agent_id
            + ", originator: "
            + originator_name
            + ", isRealFace: "
            + isrealface
            + ", Snapdataurl: "
            + snap_data_url
        )

        # Check if snapdata  or agentId is empty. If empty return back with exception to the client
        if snap_encoded_data == "" or agent_id == "" or snap_data_url == "":
            logging.info(
                agent_id
                + ":VerifyFaceAuthWithEncoding snapdata: "
                + snap_encoded_data
                + " agentId: "
                + agent_id
            )
            return ProcessExceptionData(
                agent_id,
                originator_name,
                "snap_encoded_data or snap_data_url or AgentId is empty",
                "",
                "",
                datetime.datetime.now(),
                datetime.datetime.now(),
            )

        if snap_data_url != "":
            # detect if this image is a real or fake image based on the configuration set
            if isrealface != "1":
                logging.info(
                    agent_id + ": isRealFace is 0. Check for liveness in the image"
                )
                if config.detect_liveness:
                    logging.info(
                        agent_id + " :detect Liveness is enabled in the config"
                    )
                    isReal = await detect_liveness(snap_data_url, False, True)
                    if not isReal:
                        logging.info(
                            agent_id + ": isReal is 0. Fake Image. Return to the client"
                        )
                        retData = {
                            "face_found_in_image": 0,
                            "face_authenticated_percentage": 0,
                            "face_isreal": 0,
                        }
                        return web.json_response(retData)

        # # check if the originator has value or not. If not, assing TMAC to it
        # if originator_name:
        #     originator = originator_name
        # else:
        #     originator = "TMAC"
        if agent_id in profilePic_encodingData.keys():
            logging.info(agent_id + " is present in the dictionary")
        else:
            logging.info(
                agent_id + " is not found in the dictionary. Update ProfilePic from OCM"
            )
            comparisionResult = {
                "face_found_in_image": 0,
                "face_authenticated_percentage": 0,
                "face_isreal": 0,
            }
            source_image_url_path = ""
            dest_image_url_path = ""
            responded_datetime = datetime.datetime.now()
            threading.Thread(
                target=sql.save_faceauth_data,
                args=(
                    requested_datetime,
                    responded_datetime,
                    source_image_url_path,
                    dest_image_url_path,
                    originator_name,
                    agent_id,
                    "False",
                    "ProfilePicEncodings not found",
                    "0",
                    "0",
                ),
            ).start()
            return web.json_response(comparisionResult)
        dest_image_url_path = profilePic_encodingData[agent_id]["url"]
        source_image_url_path = snap_data_url
        logging.info(
            agent_id + ":Destination and source Image URL path assigned")

        destFaceEncodingValue = profilePic_encodingData[agent_id]["encoding"]
        logging.info(
            agent_id + ":DestinationEncodingValue is captured from the dictionary"
        )
        if isinstance(destFaceEncodingValue, str):
            lst = json.loads(destFaceEncodingValue)
            destFaceEncodingValue = [np.array(lst)]

        sourceFaceEncodingValue = [np.array(snap_encoded_data)]
        faceFound = True
        match_results = face_recognition.compare_faces(
            [destFaceEncodingValue[0]], sourceFaceEncodingValue[0]
        )
        face_distances = face_recognition.face_distance(
            [destFaceEncodingValue[0]], sourceFaceEncodingValue[0]
        )
        logging.info(
            agent_id
            + ":VerifyFaceAuthWithEncoding : Face distance "
            + str(face_distances)
        )

        fpercentage = await GetFaceComparisionPercentage(face_distances, agent_id, 0.6)
        logging.info(
            agent_id
            + ":VerifyFaceAuthWithEncoding : Percentage of face match "
            + str(fpercentage[0])
        )

        # Return the comparisionResult as json
        comparisionResult = {
            "face_found_in_image": faceFound,
            "face_authenticated_percentage": fpercentage[0] * 100,
            "face_isreal": 1,
        }
        responded_datetime = datetime.datetime.now()
        threading.Thread(
            target=sql.save_faceauth_data,
            args=(
                requested_datetime,
                responded_datetime,
                source_image_url_path,
                dest_image_url_path,
                originator_name,
                agent_id,
                "True",
                "Face found in Image",
                str(fpercentage[0] * 100),
                str(face_distances),
            ),
        ).start()

        logging.info(
            agent_id + ":VerifyFaceAuthWithEncoding Response: " +
            str(comparisionResult)
        )

        return web.json_response(comparisionResult)

    except Exception as ex:
        logging.info(
            agent_id + ":Exception in VerifyFaceAuthWithEncoding : " + str(ex))

        return await ProcessExceptionData(
            agent_id,
            originator_name,
            "Exception in VerifyFaceAuthWithEncoding " + str(ex),
            "",
            "",
            datetime.datetime.now(),
            datetime.datetime.now(),
        )


async def AddOrUpdateProfilePic(self, *, loads=json.loads):
    """
    Method to update the encoding of profile pic of user into a pickle dict. 
    INPUT: agentId, Image URL
    OUTPUT: True/False
    """
    try:
        global profilePic_encodingData

        conn = sqlite3.connect("profilePicEncoding.db")
        cursor = conn.cursor()

        json_req = await self.text()
        json_req = json.loads(json_req)
        agent_id = json_req["agentId"]
        profilepic_url = json_req["profilepicurl"]
        logging.info(
            agent_id + ": AddOrUpdateProfilePic: Profile Pic URL: " + profilepic_url
        )
        # # # if os.path.exists(pickle_filename):
        # # #     with open(pickle_filename, 'rb') as handle:
        # # #         profilePic_encodingData = pickle.load(handle)
        # # # else:
        # # #     # make a new one
        # # #     # we use a dict for keeping track of mapping of each person with his/her face encoding
        # # #     profilePic_encodingData = defaultdict(dict)
        profilePic = iosk.imread(profilepic_url)
        profilePicEncoding = face_recognition.face_encodings(profilePic)
        # enco = json.dumps(profilePicEncoding)
        string = json.dumps(profilePicEncoding[0].tolist())

        data_to_be_inserted = (agent_id, profilepic_url, string)

        cursor.execute(
            "SELECT AGENTID FROM AGENTPROFILEPICDETAILS WHERE AGENTID = ?", (
                agent_id,)
        )
        ROW = cursor.fetchone()

        if ROW == None:
            sqlite_insert_query = """INSERT INTO AGENTPROFILEPICDETAILS
                          (AgentId, PROFILEPICURL, ENCODINGDATA) 
                           VALUES 
                          (?,?,?)"""
            count = cursor.execute(sqlite_insert_query, data_to_be_inserted)
            conn.commit()
        else:
            data_to_be_inserted = (profilepic_url, string, agent_id)
            sqlite_insert_query = """UPDATE AGENTPROFILEPICDETAILS SET PROFILEPICURL=?, ENCODINGDATA=? WHERE AGENTID=?"""
            cursor.execute(sqlite_insert_query, data_to_be_inserted)
            conn.commit()
        conn.close()
        profilePic_encodingData[agent_id]["encoding"] = profilePicEncoding
        profilePic_encodingData[agent_id]["url"] = profilepic_url

        # # # profilePic_encodingData[agent_id]['url'] = profilepic_url
        # # # profilePic_encodingData[agent_id]['face_encoding'] = profilePicEncoding

        # # # Updating the agent ID and encoding to the pickle
        # # # outfile = open(pickle_filename, 'wb')
        # # # pickle.dump(profilePic_encodingData, outfile,
        # # #             protocol=pickle.HIGHEST_PROTOCOL)
        # # # outfile.close()

        # # # infile = open(pickle_filename, 'rb')
        # # # profilePic_encodingData = pickle.load(infile)
        # # # infile.close()
        return web.json_response(True)

    except Exception as ex:
        logging.info(
            agent_id + ": Exception in AddOrUpdateProfilePic: " + str(ex))
        return web.json_response(False)


def initialize_db():
    global profilePic_encodingData
    conn = sqlite3.connect("profilePicEncoding.db")
    cursor = conn.cursor()
    cursor = cursor.execute(
        "SELECT AGENTID,PROFILEPICURL,ENCODINGDATA FROM AGENTPROFILEPICDETAILS"
    )
    rows = cursor.fetchall()
    conn.close()
    for row in rows:
        profilePic_encodingData[row[0]]["url"] = row[1]
        profilePic_encodingData[row[0]]["encoding"] = row[2]
        # print(row)
    # print(cursor)


@asyncio.coroutine
def handler(request):
    return web.Response(
        text="Hello!", headers={"X-Custom-Server-Header": "Custom data", }
    )


app = web.Application()
cors = aiohttp_cors.setup(app)
initialize_db()  # initialize sqlite db
# add the below 2 methods for allowing cors
# resource = cors.add(app.router.add_resource("/VerifyFaceAuthWithEncoding"))
# profile_resource = cors.add(app.router.add_resource("/AddOrUpdateProfilePic"))
# verifyImages_resource = cors.add(app.router.add_resource("/VerifyFaceImages"))
# route = cors.add(
#     verifyImages_resource.add_route("POST", VerifyFaceImages),
#     {
#         "*": aiohttp_cors.ResourceOptions(
#             allow_credentials=True, expose_headers="*", allow_headers="*"
#         )
#     },
# )
# route = cors.add(
#     resource.add_route("POST", VerifyFaceAuthWithEncoding),
#     {
#         "*": aiohttp_cors.ResourceOptions(
#             allow_credentials=True, expose_headers="*", allow_headers="*"
#         )
#     },
# )
# route = cors.add(
#     profile_resource.add_route("POST", AddOrUpdateProfilePic),
#     {
#         "*": aiohttp_cors.ResourceOptions(
#             allow_credentials=True, expose_headers="*", allow_headers="*"
#         )
#     },
# )

routes = [
    web.post("/VerifyFaceImages", VerifyFaceImages),
    web.post("/test", upload_image),
    # web.static('/profilepic', config.directory_browsing_proPic,
    #            show_index=True),
    # web.static('/sourcepic', config.directory_browsing_destImage,
    #            show_index=True),
    web.post("/AddOrUpdateProfilePic", AddOrUpdateProfilePic),
    web.post("/VerifyFaceAuthWithEncoding", VerifyFaceAuthWithEncoding)
]

app.router.add_routes(routes)
cors = cors_setup(
    app,
    defaults={
        "*": ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*",
        )
    },
)
for route in list(app.router.routes()):
    cors.add(route)
# Comment below 2 lines if attached to Gunicorn in Linux
if __name__ == "__main__":
    web.run_app(app)
