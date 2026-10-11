def circuit_breaker(user_id, new_response_text):
    # 1. Check Money
    if get_spend(user_id) > 50.00:
        block_request()

    # 2. Check Loops (The Smart Part)
    current_vector = create_embedding(new_response_text)
    last_3_vectors = get_last_3_vectors_from_redis(user_id)

    for vector in last_3_vectors:
        similarity = cosine_similarity(current_vector, vector)
        if similarity > 0.95:
            block_request("Loop Detected: Agent is repeating itself.")
            
    # 3. If Safe, Save current vector to Redis
    save_vector_to_redis(user_id, current_vector)