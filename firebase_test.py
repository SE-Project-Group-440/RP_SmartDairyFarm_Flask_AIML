import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://cattle-heat-stress-default-rtdb.asia-southeast1.firebasedatabase.app"
})

print("Firebase connected successfully!")
