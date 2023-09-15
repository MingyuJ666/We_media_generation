# from transformers import BertModel,BertTokenizer
# BERT_PATH = './bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# print("1111")
# print(tokenizer.tokenize('I have a good time, thank you.'))
# bert = BertModel.from_pretrained(BERT_PATH)
# print('load bert model over')

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
example_text = 'I will watch Memento tonight,huang yu xuan shi sha B'
bert_input = tokenizer(example_text,padding='max_length',
                       max_length = 30,
                       truncation=True,
                       return_tensors="pt")
# ------- bert_input ------
print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])
print(tokenizer.decode(bert_input.input_ids[0]))

