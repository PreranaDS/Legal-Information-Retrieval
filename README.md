# Legal-Information-Retrieval
SMART INDIA HACKATHON '22(Finals)... This legal information retrieval system takes a query as input and finds a list of judgments that are similar to the query(relevant judgments), in decreasing order of similarity. It is also autocompletes the query and suggests case names as the user types. It uses tf-idf and word2vec to find similar judgments. Similarity matching is done based on cosine similarity. Heuristics such as court score, citation score, case date and page rank score(where edge weight is set to citation score) along with filters such as filtering by court, case year, etc. are used. It uses LSTM trained on court data and tf for autocomplete. It makes use of flask for UI along with speech recognition to convert the spoken query into text query for retrieval.