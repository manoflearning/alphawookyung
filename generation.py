import torch
from tokenizer import CharTokenizer
from model import GPT, GPTConfig

torch.manual_seed(1337)

# load the tokenizer
tok = CharTokenizer()

# load the model from the .pth file
model = GPT(GPTConfig(block_size=64, vocab_size=tok.vocab_size, n_layer=6, n_head=6, n_embd=384, dropout=0.1, bias=False))

model_path = 'model_weight/prototype.pth'
model.load_state_dict(torch.load(model_path))

# set the model to evaluation mode
model.eval()

# run the model in a loop
while True:
    # get user input
    print('Enter a prompt:')
    user_input = input()
    if user_input == 'quit':
        break
    print()

    # encode the user input
    context = torch.tensor(tok.encode(user_input), dtype=torch.int64).view((1, len(user_input)))

    print('Response:')
    print(f'{tok.decode(context[0].tolist())}', end='')

    max_new_tokens = 800
    for _ in range(max_new_tokens):
        # generate a response
        context = model.generate(context, max_new_tokens=1)
        # decode and print the response
        print(tok.decode(context[0].tolist())[-1], end='')

    print()
    print()
