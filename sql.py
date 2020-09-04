import logging
import logging.handlers as handlers
from logging.handlers import RotatingFileHandler
import json
from collections import defaultdict
import collections
import sqlite3
import base64
import numpy as np
from cryptography.fernet import Fernet
import cv2
import face_recognition
import pyodbc
import config

logging.basicConfig(
    handlers=[
        RotatingFileHandler("TFaceCompare.log", maxBytes=10485760, backupCount=10)
    ],
    level=logging.DEBUG,
    format='"%(asctime)s -[TFaceCompare] - %(levelname)-7.7s %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)
if config.is_encrypted_password:
    key = config.cipher_key
    cipher_suite = Fernet(key)
    password = cipher_suite.decrypt(config.password)
    password = password.decode("utf-8")
else:
    password = config.password

server = config.server
database = config.database
username = config.username
password = config.password

profilePic_encodingData1 = defaultdict(collections.Counter)


def save_faceauth_data(
    request_date,
    response_date,
    image1_path,
    image2_path,
    originator,
    agentId,
    result,
    resultReason,
    score,
    confidence,
):
    """
    This function inserts the data to the SQL server
    which has the details of face comparasion
    INPUT: request_date,response_date,originator,
    image1_path,image2_path,result,result_reason,score,confidence
    OUTPUT:
    """
    try:
        cnxn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )

        cursor = cnxn.cursor()
        logging.info(
            agentId
            + ":save_faceauth_data: requestDate: "
            + str(request_date)
            + ", responseDate: "
            + str(response_date)
            + ", Originator: "
            + str(originator)
            + ", Image Path1: "
            + str(image1_path)
            + ", Image Path2: "
            + str(image2_path)
        )

        cursor.execute(
            """
        INSERT INTO FaceAuth_Transactions (RequestDateTime, ResponseDateTime, Originator, Image1Path, 
        Image2Path, Result, ResultReason, Score, Confidence) 
        VALUES (?,?,?,?,?,?,?,?,?)""",
            request_date,
            response_date,
            originator,
            image1_path,
            image2_path,
            result,
            resultReason,
            score,
            confidence,
        )
        cnxn.commit()
        logging.info(agentId + ":Successfully inserted data to the table")
        cursor.close()
        cnxn.close()
        return

    except Exception as ex:
        logging.info(
            agentId + ":Exception in inserting data to the sql table " + str(ex)
        )
        return


def load_profilepic():
    '''
    This method will be called based on the config set to True
    THis will load the profilepic base64 from the DB configured
    to the Sqlite DB
    '''
    try:
        cnxn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
        # if True:
        #     test_method()
        #     return
        cursor = cnxn.cursor()

        conn = sqlite3.connect("profilePicEncoding.db")
        lite_cursor = conn.cursor()

        cursor.execute("SELECT AgentId,ProfilePicture FROM AGT_Agent_Profile")
        for row in cursor.fetchall():
            if row[1] is None or row[1]== "":
                continue
            lite_cursor.execute(
                "SELECT AGENTID FROM AGENTPROFILEPICDETAILS WHERE AGENTID = ?",
                (row[0],),
            )
            profilepic_base64 = row[1].split(";")[1].split(",")[1]
            profilePicEncoding = get_face_encoding_from_base64(profilepic_base64)
            string = json.dumps(profilePicEncoding[0].tolist())
            data_to_be_inserted = (row[0], profilepic_base64, string)
            ROW = lite_cursor.fetchone()
            if ROW is None:
                sqlite_insert_query = """INSERT INTO AGENTPROFILEPICDETAILS
                            (AgentId, PROFILEPICURL, ENCODINGDATA) 
                            VALUES 
                            (?,?,?)"""
                lite_cursor.execute(sqlite_insert_query, data_to_be_inserted)
                conn.commit()
            else:
                data_to_be_inserted = (profilepic_base64, string, row[0])
                sqlite_insert_query = """UPDATE AGENTPROFILEPICDETAILS SET PROFILEPICURL=?, ENCODINGDATA=? WHERE AGENTID=?"""
                lite_cursor.execute(sqlite_insert_query, data_to_be_inserted)
                conn.commit()

        lite_cursor.close()
        conn.close()
        cursor.close()
        cnxn.close()

        return profilePic_encodingData1
    except Exception as ex:
        logging.info("Exception in loadProfilePic {}".format(ex))
        return profilePic_encodingData1


def get_face_encoding_from_base64(base64String):
    nparr = np.frombuffer(base64.b64decode(base64String.encode("utf-8")), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_encodeing = face_recognition.face_encodings(rgb_img)
    return image_encodeing

def test_method():
    conn = sqlite3.connect("profilePicEncoding.db")
    lite_cursor = conn.cursor()
    string = '[-0.08938587456941605, 0.06893651187419891, 0.018590139225125313, -0.09272828698158264, -0.047466836869716644, 0.002653779461979866, -0.08349313586950302, -0.007166721858084202, 0.21946996450424194, -0.04587170481681824, 0.1667509227991104, -0.0654381737112999, -0.21751311421394348, -0.04719454050064087, -0.010839691385626793, 0.09368138760328293, -0.18806946277618408, -0.06498026847839355, -0.029204364866018295, -0.10424026846885681, 0.06686399132013321, 0.05317986011505127, -0.01600918173789978, 0.051890864968299866, -0.0858193039894104, -0.2501823604106903, -0.10916633903980255, -0.1798149049282074, 0.05383351445198059, -0.09286559373140335, -0.03175853192806244, -0.013902044855058193, -0.14152558147907257, -0.11782997101545334, 0.060388825833797455, 0.14077474176883698, -0.03575936332345009, -0.03784899041056633, 0.15619978308677673, -0.015744678676128387, -0.07130786776542664, 0.027044879272580147, 0.06568709015846252, 0.3015233874320984, 0.1593284159898758, 0.04137902706861496, -0.053089868277311325, -0.06557597219944, 0.227407768368721, -0.2649993300437927, 0.17930465936660767, 0.1297755390405655, 0.18200835585594177, 0.1046452671289444, 0.16458721458911896, -0.20860429108142853, 0.11790688335895538, 0.20218680799007416, -0.26549795269966125, 0.07407739758491516, 0.06475543975830078, 0.03485866263508797, -0.034396909177303314, 0.04477817565202713, 0.1983388066291809, 0.1719212383031845, -0.0823221355676651, -0.21032167971134186, 0.10217731446027756, -0.12086490541696548, 0.0007242048159241676, 0.11831578612327576, -0.14716403186321259, -0.12015529721975327, -0.2982698380947113, 0.058295294642448425, 0.33584436774253845, 0.06673242896795273, -0.1766497939825058, 0.033144209533929825, -0.0407220833003521, -0.015364979393780231, 0.06507329642772675, -0.019789956510066986, -0.1610373854637146, 0.013054459355771542, -0.1234896332025528, 0.011375433765351772, 0.09954973310232162, -0.01031947135925293, -0.012469345703721046, 0.19012555480003357, 0.035587433725595474, -0.007815588265657425, 0.09378251433372498, 0.04135192930698395, -0.10963395982980728, -0.08052320033311844, -0.19584552943706512, -0.06949669867753983, 0.07537500560283661, -0.11380954831838608, -0.07116565108299255, 0.0940367579460144, -0.292865127325058, 0.12168414145708084, 0.0014007885474711657, -0.0670354813337326, -0.09598086774349213, -0.08009172976016998, -0.10996457934379578, 0.06207108497619629, 0.23005101084709167, -0.2886205315589905, 0.14512990415096283, 0.11175626516342163, 0.0716632828116417, 0.14331108331680298, 0.01819598115980625, -0.014311758801341057, -0.035418055951595306, -0.06169872730970383, -0.15684519708156586, -0.02799009159207344, 0.016726262867450714, 0.017618583515286446, -0.029520666226744652, 0.04377150535583496]'
    profilepic_base64 = 'https://s01.sgp1.cdn.digitaloceanspaces.com/article/143666-oxownzwrao-1593513533.jpg'
    for ind in range(1999):
        data_to_be_inserted = (ind + 1, profilepic_base64, string)
        sqlite_insert_query = """INSERT INTO AGENTPROFILEPICDETAILS
                            (AgentId, PROFILEPICURL, ENCODINGDATA) 
                            VALUES 
                            (?,?,?)"""
        lite_cursor.execute(sqlite_insert_query, data_to_be_inserted)
        conn.commit()
    lite_cursor.close()
    conn.close()
