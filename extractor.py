from keybert import KeyBERT

def get_keyword(text):
    op = []
    kw_model = KeyBERT()
    for sentence in text:

        key = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 1), stop_words=None)
        op.append(key[0][0])
        op.append(key[1][0])
        op.append(key[2][0])

    return op


