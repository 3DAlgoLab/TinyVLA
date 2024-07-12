from datetime import datetime


def generate_timestamp():
    """
    Example usage:
        formatted_timestamp = generate_timestamp()
        print("Formatted timestamp:", formatted_timestamp)
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return timestamp
