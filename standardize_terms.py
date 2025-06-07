import re

term_standardization_map = {
    'LLMs': 'LLM',
    'generative AI': 'Generative Artificial Intelligence',
    'generative artificial intelligence': 'Generative Artificial Intelligence',
    'genai': 'Generative Artificial Intelligence',
    'GenAI': 'Generative Artificial Intelligence',
    'large language models': 'LLM',
    'large language model': 'LLM',
    'llms': 'LLM',
    'l.l.m.': 'LLM',
    'AI': 'Artificial Intelligence',
    'A.I.': 'Artificial Intelligence',
    'artificial inteligence': 'Artificial Intelligence',
    'artificial-intelligence': 'Artificial Intelligence',
    'a i': 'Artificial Intelligence',
    'machine learning': 'Machine Learning',
    'machine-learning': 'Machine Learning',
    'machin learning': 'Machine Learning',
    'ml': 'Machine Learning',
    'ML': 'Machine Learning',
    'deep learning': 'Deep Learning',
    'deep-learning': 'Deep Learning',
    'dl': 'Deep Learning',
    'DL': 'Deep Learning',
    'natural language processing': 'Natural Language Processing',
    'natural-language processing': 'Natural Language Processing',
    'nlp': 'Natural Language Processing',
    'NLP': 'Natural Language Processing',
    'computer vision': 'Computer Vision',
    'CV': 'Computer Vision',
    'big data': 'Big Data',
    'data science': 'Data Science',
    'robotics': 'Robotics',
    'IoT': 'Internet of Things',
    'Internet of Things': 'Internet of Things',
    'blockchain': 'Blockchain',
    'cybersecurity': 'Cybersecurity',
    # British vs. American spellings
    'colour': 'color',
    'behaviour': 'behavior',
    'behavioural': 'behavioral',
    'organisation': 'organization',
    'organise': 'organize',
    'optimisation': 'optimization',
    'optimise': 'optimize',
    'analyse': 'analyze',
    'labour': 'labor',
    'defence': 'defense',
    'theatre': 'theater',
    'centre': 'center',
    'grey': 'gray',
    'programme': 'program',
    'catalogue': 'catalog',
    'dialogue': 'dialog',
    'licence': 'license',
    'cheque': 'check',
    'fibre': 'fiber',
    'neighbour': 'neighbor',
    'globalisation': 'globalization',
    'initialisation': 'initialization',
    'normalisation': 'normalization',
    'specialisation': 'specialization',
    'utilisation': 'utilization',
    'utilise': 'utilize',
    'artefact': 'artifact',
    'authorisation': 'authorization',
    'miniaturisation': 'miniaturization',
    'standardisation': 'standardization'
}

def standardize_text(text_input):
    if not isinstance(text_input, str):
        return text_input

    # Sort keys by length in descending order to match longer phrases first
    sorted_keys = sorted(term_standardization_map.keys(), key=len, reverse=True)

    processed_text = text_input
    for key in sorted_keys:
        # Match the key in a case-insensitive way
        # Using re.escape to handle special characters in keys
        # Using \b for word boundaries to avoid partial matches within words
        pattern = r'\b' + re.escape(key) + r'\b'
        try:
            # re.IGNORECASE for the search pattern
            # The replacement is the value from the map, which has the desired capitalization
            processed_text = re.sub(pattern, term_standardization_map[key], processed_text, flags=re.IGNORECASE)
        except re.error as e:
            # Handle potential regex errors, e.g., if a key creates an invalid pattern
            # This might be overly cautious if keys are well-defined
            print(f"Regex error for key '{key}': {e}")
            continue # Skip this key if it causes an error

    return processed_text

if __name__ == '__main__':
    # This part will be implemented in a later step
    pass
