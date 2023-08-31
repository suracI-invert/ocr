from src.models.tokenizer import Tokenizer

tokenizer = Tokenizer()

test_str_list = [
    'Mấtba',
    'Mat-so',
    'ĐM_ĐQA-giao_cái_task như đb'
]

for i, s in enumerate(test_str_list):
    print(f'test string {i}: {s}')
    tokenized_input = tokenizer.encode(s, return_tensor= True)
    print('encoding: ', tokenized_input['input_ids'])
    print('length of ids: ', len(tokenized_input['input_ids']))
    print('attn_mask: ', tokenized_input['attention_masks'])
    print('decoding: ', tokenizer.decode(tokenized_input['input_ids']))
