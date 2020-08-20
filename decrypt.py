from cryptography.fernet import Fernet

key = input("Enter the cipher key: ")
cipher_suite = Fernet(key)
print("Entered key is: " + str(key))
text_entered = input("Please enter the encrypted password: ")
res = bytes(text_entered, "utf-8")
decoded_text = cipher_suite.decrypt(res)
print(decoded_text)
