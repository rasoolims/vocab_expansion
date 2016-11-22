import re

class ConllEntry:
    def __init__(self, id, form, lemma, pos, parent_id=None, relation=None, lang_id=None, weight= 1.0):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.norm = normalize(form)
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.lang_id = lang_id
        self.weight = weight

def read_conll(fh):
    read = 0
    tokens = []
    for line in fh:
        tok = line.strip().split()
        if not tok:
            if len(tokens) > 1:
                yield tokens
                read += 1
            tokens = []
        else:
            try:
                weight = float(tok[8])
            except:
                weight = 1
            tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[5], weight))
    if len(tokens) > 1:
        yield tokens
    print read, 'sentences read.'

def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write('\t'.join(
                    [str(entry.id), entry.form, entry.lemma, entry.pos, entry.pos, entry.lang_id, str(entry.parent_id),
                     entry.pred_relation, '_', '_']))
                fh.write('\n')
            fh.write('\n')
def conll_str(sentence):
    output = []
    for entry in sentence[1:]:
        output.append('\t'.join(
            [str(entry.id), entry.form, entry.lemma, entry.pos, entry.pos, entry.lang_id, str(entry.parent_id),
             entry.pred_relation, '_', '_']))
    return '\n'.join(output)+'\n\n'

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");

def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()