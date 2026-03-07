import firebase_admin
from firebase_admin import credentials, db

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": "https://cattle-heat-stress-default-rtdb.asia-southeast1.firebasedatabase.app"
        }
    )

def get_latest_readings(cattle_id: str, limit: int = 6):
    """
    Reads last 6 records from:
    collars/Cattle1
    """
    ref = db.reference(f"collars/{cattle_id}")
    data = ref.order_by_key().limit_to_last(limit).get()

    if not data:
        return []

    rows = []
    for _, v in sorted(data.items()):
        rows.append([
            v["envTemp"],
            v["humidity"]
        ])

    return rows
