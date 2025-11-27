import time

REQUEST_LIMIT = 5          # max uploads
TIME_WINDOW = 60           # per 60 seconds

user_requests = {}

def allow_request(user_id="default"):
    now = time.time()

    if user_id not in user_requests:
        user_requests[user_id] = []

    # keep recent requests only
    user_requests[user_id] = [t for t in user_requests[user_id] if now - t < TIME_WINDOW]

    if len(user_requests[user_id]) >= REQUEST_LIMIT:
        return False

    user_requests[user_id].append(now)
    return True

