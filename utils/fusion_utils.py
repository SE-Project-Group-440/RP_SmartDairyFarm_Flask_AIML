def fuse_results(result):
    if "image_prediction" in result:
        return result["image_prediction"]

    if "symptoms" in result:
        return max(result["symptoms"], key=result["symptoms"].get)

    return "Insufficient data"
