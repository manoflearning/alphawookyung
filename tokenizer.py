# naive character-level tokenizer
# TODO: add a advanced tokenizer (sentencepiece / tiktoken / minbpe / ... )

class CharTokenizer:

    def __init__(self):
        self.chars = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'ģ', 'б', '가', '갓', '규', '근', '다', '대', '둘', '렸', '릴', '마', '문', '성', '소', '수', '스', '신', '어', '없', '열', '우', '웅', '을', '음', '의', '이', '인', '있', '잘', '정', '제', '터', '틀', '했']
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i, ch in enumerate(self.chars) }
        self.itos = { i:ch for i, ch in enumerate(self.chars) }
        self.enc = lambda s: [self.stoi[c] for c in s]
        self.dec = lambda l: ''.join([self.itos[i] for i in l])
    
    def encode(self, s):
        return self.enc(s)
    
    def decode(self, l):
        return self.dec(l)
