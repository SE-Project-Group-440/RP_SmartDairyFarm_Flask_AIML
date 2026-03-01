from services.chat_service import get_chat_response

def chat_controller(request: dict, base_url: str):
    query = request.get("query")   # SAFE way
    return get_chat_response(query, base_url)