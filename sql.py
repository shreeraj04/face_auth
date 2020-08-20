import pyodbc
import datetime
import logging
import logging.handlers as handlers
from logging.handlers import RotatingFileHandler
import config
import threading
import time
from cryptography.fernet import Fernet
import tfaceauthserver

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
# drivers = [item for item in pyodbc.drivers()]
# driver = drivers[-1]


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
    This function inserts the data to the SQL server which has the details of face comparasion
    INPUT: request_date,response_date,originator,image1_path,image2_path,result,result_reason,score,confidence
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
        INSERT INTO FaceAuth_Transactions (RequestDateTime, ResponseDateTime, Originator, Image1Path, Image2Path, Result, ResultReason, Score, Confidence) 
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
