from cryptography.fernet import Fernet

key = Fernet.generate_key()  # this is your "password"
cipher_suite = Fernet(key)
print("key " + str(key))
text_entered = input("Please enter the password: ")
print("Copy the encypted text & key to the config file")
res = bytes(text_entered, "utf-8")
encoded_text = cipher_suite.encrypt(res)
print(encoded_text)
