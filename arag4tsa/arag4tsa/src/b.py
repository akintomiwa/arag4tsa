import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_relevant_prompts(prompt_pool, data):
    data_vector = embed_data(data)
    similarities = cosine_similarity(data_vector, prompt_pool['keys'])
    top_prompts_idx = np.argsort(similarities, axis=1)[:,-5:]  # Get top 5 prompts
    return prompt_pool['values'][top_prompts_idx]

def embed_data(data):
    # Convert data to the vector form
    return model.embed(data)
