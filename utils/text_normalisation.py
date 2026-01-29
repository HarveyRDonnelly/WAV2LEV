import sys, re
from whisper.normalizers import EnglishTextNormalizer

mapping_dict = {r"\b401[kK]\b": "four o one k"}

class CustomNormaliser(EnglishTextNormalizer):
    def __init__(self, nemo_norm):
        super().__init__()
        self.nemo_norm = nemo_norm
        self.bracket = re.compile(r"[<\[][^>\]]*[>\]]")
        self.paren = re.compile(r"\(([^)]+?)\)")
        self.mapping = [(re.compile(p), r) for p, r in mapping_dict.items()]
        self.dot = re.compile(r'\s+\.\s+|\s+\.$|^\.\s+')

    def normalize_batch(self, texts):
        out = []
        for txt in texts:
            txt = self.bracket.sub("", txt)
            txt = self.paren.sub("", txt)
            for pat, rep in self.mapping:
                txt = pat.sub(rep, txt)
                
            if self.nemo_norm is not None:
                txt = self.nemo_norm.normalize(txt)
                
            txt = super().__call__(txt)
            txt = self.dot.sub(" ", txt)
            txt = re.sub(r'\s+', ' ', txt).strip()
            out.append(txt)
        return out

    def __call__(self, text):
        return self.normalize_batch([text])[0]
