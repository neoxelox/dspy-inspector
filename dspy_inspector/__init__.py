from .inspector import Inspector

inspect = Inspector(debug=False, tokenizer_of_model="gpt-3.5-turbo-instruct").inspect
