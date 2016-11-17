from dynet import *
import random,sys,os,codecs,pickle
from optparse import OptionParser
import numpy as np

class AlignmentInstance:
    def __init__(self, src_line, dst_line, a_line, src_word_dict, dst_word_dict, src_pos_dict, sen):
        self.dst_words = [dst_word_dict[w] if w in dst_word_dict else dst_word_dict['_RARE_'] for w in dst_line.strip().split()]
        a_s = a_line.strip().split()
        s_wt = src_line.strip().split()
        self.src_words = []
        self.src_tags = []
        self.orig_src_tags = []
        for wt in s_wt:
            i = wt.rfind('_')
            self.src_words.append(src_word_dict[wt[:i]] if wt[:i] in src_word_dict else src_word_dict['_RARE_'])
            self.src_tags.append(src_pos_dict[wt[i+1:]])
            self.orig_src_tags.append(wt[i+1:])

        self.alignments = dict()
        for a in a_s:
            s,t = a.strip().split('-')
            self.alignments[int(s)] = int(t)
            if int(s)>=len(self.src_words):
                print sen, a, len(self.src_words), len(self.dst_words)
                print src_line
                print dst_line
                print a_line
            assert int(s)<len(self.src_words)

class Expander:
    @staticmethod
    def parse_options():
        parser = OptionParser()
        parser.add_option('--train_src', dest='train_src', metavar='FILE', default='')
        parser.add_option('--train_dst', dest='train_dst', metavar='FILE', default='')
        parser.add_option('--train_align', dest='train_align', metavar='FILE', default='')
        parser.add_option('--dev_src', dest='dev_src', metavar='FILE')
        parser.add_option('--dev_dst', dest='dev_dst', metavar='FILE')
        parser.add_option('--dev_align', dest='dev_align', metavar='FILE')
        parser.add_option('--test', dest='conll_test', metavar='FILE', default='')
        parser.add_option('--outfile', type='string', dest='outfile', default='')
        parser.add_option('--params', dest='params', help='Parameters file', metavar='FILE', default='params.pickle')
        parser.add_option('--src_embed', dest='src_embedding', help='External source word embeddings', metavar='FILE')
        parser.add_option('--dst_embed', dest='dst_embedding', help='External target word embeddings', metavar='FILE')
        parser.add_option('--pos_embed', dest='pos_embedding', help='External source pos embeddings', metavar='FILE')
        parser.add_option('--src_freq', dest='src_freq', help='Frequency level info for source word', metavar='FILE')
        parser.add_option('--dst_freq_tag', dest='dst_freq_tag', help='Frequency level + tag info for source word', metavar='FILE')
        parser.add_option('--src2dst_dict', dest='src2dst_dict', help='Dictionary (needed for decoding) -- format src[space]dst[space]freq', metavar='FILE')
        parser.add_option('--model', dest='model', help='Load/Save model file', metavar='FILE', default='model.model')
        parser.add_option('--epochs', type='int', dest='epochs', default=5)
        parser.add_option('--batch', type='int', dest='batchsize', default=1024)
        parser.add_option('--hidden', type='int', dest='hidden_units', default=200)
        parser.add_option('--hidden2', type='int', dest='hidden2_units', default=0)
        parser.add_option('--lstmdims', type='int', dest='lstm_dims', default=200)
        parser.add_option('--neg', type='int', help='number of negative samples', dest='neg', default=10)
        parser.add_option('--outdir', type='string', dest='output', default='')
        parser.add_option("--eval", action="store_true", dest="eval_format", default=False)
        return parser.parse_args()

    def __init__(self, options):
        self.model = Model()
        self.trainer = AdamTrainer(self.model)
        self.lstm_dims = options.lstm_dims
        self.neg = options.neg
        assert self.neg > 0

        if options.src_embedding != None:
            to_save_params = []
            src_embed_fp = open(options.src_embedding, 'r')
            src_embed_fp.readline()
            self.src_embed = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in src_embed_fp}
            src_embed_fp.close()
            self.src_dim = len(self.src_embed.values()[0])
            self.src_word_dict = {word: i for i, word in enumerate(self.src_embed)}
            self.src_embed_lookup = self.model.add_lookup_parameters((len(self.src_word_dict) + 3, self.src_dim))
            self.src_embed_lookup.set_updated(False)
            for word, i in self.src_word_dict.iteritems():
                self.src_embed_lookup.init_row(i, self.src_embed[word])
            assert '_RARE_' in self.src_word_dict
            self.src_rare = self.src_word_dict['_RARE_']
            to_save_params.append(self.src_word_dict)
            print 'Loaded src word embeddings. Vector dimensions:', self.src_dim

            dst_embed_fp = open(options.dst_embedding, 'r')
            dst_embed_fp.readline()
            self.dst_embed = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in dst_embed_fp}
            dst_embed_fp.close()
            self.dst_dim = len(self.dst_embed.values()[0])
            self.dst_word_dict = {word: i for i, word in enumerate(self.dst_embed)}
            self.dst_embed_lookup = self.model.add_lookup_parameters((len(self.dst_word_dict) + 3, self.dst_dim))
            self.dst_embed_lookup.set_updated(False)
            for word, i in self.dst_word_dict.iteritems():
                self.dst_embed_lookup.init_row(i, self.dst_embed[word])
            assert '_RARE_' in self.dst_word_dict
            self.dst_rare = self.dst_word_dict['_RARE_']
            to_save_params.append(self.dst_word_dict)
            print 'Loaded dst word embeddings. Vector dimensions:', self.dst_dim

            pos_embed_fp = open(options.pos_embedding, 'r')
            pos_embed_fp.readline()
            self.pos_embed = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in pos_embed_fp}
            pos_embed_fp.close()
            self.pos_dim = len(self.pos_embed.values()[0])
            self.pos_dict = {word: i for i, word in enumerate(self.pos_embed)}
            self.pos_embed_lookup = self.model.add_lookup_parameters((len(self.pos_dict) + 3, self.pos_dim))
            self.pos_embed_lookup.set_updated(False)
            for word, i in self.pos_dict.iteritems():
                self.pos_embed_lookup.init_row(i, self.pos_embed[word])
            to_save_params.append(self.pos_dict)
            print 'Loaded pos word embeddings. Vector dimensions:', self.pos_dim

            dct = pickle.load(codecs.open(options.dst_freq_tag, 'rb'))
            self.dst_freq_tag_dict = dict()
            added = 0
            for k in dct.keys():
                for w in dct[k]:
                    if w in self.dst_word_dict:
                        if not k in self.dst_freq_tag_dict:
                            self.dst_freq_tag_dict[k] = []
                        self.dst_freq_tag_dict[k].append(self.dst_word_dict[w])
                        added+=1
            to_save_params.append(self.dst_freq_tag_dict)
            print 'loaded dst_freq_tag_dict with classes:', len(self.dst_freq_tag_dict), 'added words:', added

            self.src_freq_dict = dict()
            for ln in codecs.open(options.src_freq, 'r'):
                w,l,f = ln.split()
                if w in self.src_word_dict:
                    self.src_freq_dict[self.src_word_dict[w]] = l
            to_save_params.append(self.src_freq_dict)
            print 'loaded src_freq_dict with words:',len(self.src_freq_dict)

            inp_dim = self.src_dim+self.pos_dim
            self.builders = [LSTMBuilder(1, inp_dim, options.lstm_dims, self.model),
                             LSTMBuilder(1, inp_dim, options.lstm_dims, self.model)]
            self.hid_dim = options.hidden_units
            self.hid2_dim = options.hidden2_units
            self.H1 = self.model.add_parameters((self.hid_dim, self.dst_dim + options.lstm_dims * 2))
            self.H2 = None if options.hidden2_units==0 else self.model.add_parameters((self.hid2_dim, self.hid_dim))
            last_hid_dims = options.hidden2_units if options.hidden2_units>0 else options.hidden_units
            self.O = self.model.add_parameters((2, last_hid_dims))

            to_save_params.append(self.src_dim)
            to_save_params.append(self.dst_dim)
            to_save_params.append(self.pos_dim)
            to_save_params.append(self.lstm_dims)
            to_save_params.append(self.hid_dim)
            to_save_params.append(self.hid2_dim)

            with open(os.path.join(options.output, options.params), 'w') as paramsfp:
                pickle.dump(to_save_params, paramsfp)
            print 'params written'
        else:
            self._readParams(options.params)
            self.model.load(options.model)

            dict_fp = open(options.src2dst_dict, 'r')
            dict_fp.readline()
            self.src2dst_dict = dict()
            for line in dict_fp:
                w,t,f = line.split()
                if not w in self.src2dst_dict:
                    self.src2dst_dict[w] = set()
                self.src2dst_dict[w].add(t)

            self.rev_src_dic = ['']*len(self.src_word_dict)
            for i in self.src_word_dict.keys():
                self.rev_src_dic[self.src_word_dict[i]] = i
            self.rev_dst_dic = ['']*len(self.dst_word_dict)
            for i in self.dst_word_dict.keys():
                self.rev_dst_dic[self.dst_word_dict[i]] = i

    def _readParams(self, f):
        with open(f, 'r') as paramsfp:
            saved_params = pickle.load(paramsfp)
        self.hid2_dim = saved_params.pop()
        self.hidden_dim = saved_params.pop()
        self.lstm_dims = saved_params.pop()
        self.pos_dim = saved_params.pop()
        self.dst_dim = saved_params.pop()
        self.src_dim = saved_params.pop()
        inp_dim = self.src_dim + self.pos_dim
        self.builders = [LSTMBuilder(1, inp_dim, self.lstm_dims, self.model),
                         LSTMBuilder(1, inp_dim, self.lstm_dims, self.model)]
        self.H1 = self.model.add_parameters((self.hid_dim, self.dst_dim + self.lstm_dims * 2))
        self.H2 = None if options.hidden2_units == 0 else self.model.add_parameters((self.hid2_dim, self.hid_dim))
        last_hid_dims = self.hid2_dim if self.hid2_dim > 0 else self.hid_dim
        self.O = self.model.add_parameters((2, last_hid_dims))

        self.src_freq_dict = saved_params.pop()
        self.dst_freq_tag_dict = saved_params.pop()
        self.pos_dict = saved_params.pop()
        self.dst_word_dict = saved_params.pop()
        self.src_word_dict = saved_params.pop()

    def eval_alignment(self, a_s):
        renew_cg()
        f_init, b_init = [b.initial_state() for b in self.builders]
        src_embed = [self.src_embed_lookup[i] for i in a_s.src_words]
        tag_embed = [self.pos_embed_lookup[i] for i in a_s.src_tags]
        inputs = [concatenate([src_embed[i], tag_embed[i]]) for i in xrange(len(src_embed))]
        fw = [x.output() for x in f_init.add_inputs(inputs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(inputs))]

        H1 = parameter(self.H1)
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)

        mmr = 0
        instances = 0
        top1 = 0
        for a in a_s.alignments.keys():
            src = a_s.src_words[a]
            translation =  a_s.dst_words[a_s.alignments[a]]
            if  src == self.src_rare or not src in self.src_freq_dict:
                continue # cannot train on this

            k = self.src_freq_dict[src]+' '+a_s.orig_src_tags[a]
            if not k in self.dst_freq_tag_dict:
                continue

            tr_embed = self.dst_embed_lookup[translation]
            inp = concatenate([tr_embed, fw[a], bw[len(src_embed)-1-a]])

            if H2:
                r_t = O * rectify(H2*(rectify(H1 * inp)))
            else:
                r_t = O * (rectify(H1 * inp))
            gold_res = r_t.npvalue()[1]

            others = []
            neg_samples = random.sample(self.dst_freq_tag_dict[k], min(self.neg,len(self.dst_freq_tag_dict[k])))
            for sample in neg_samples:
                tr_embed = self.dst_embed_lookup[sample]
                inp = concatenate([tr_embed, fw[a], bw[len(src_embed) - 1 - a]])
                if H2:
                    r_t = O * rectify(H2 * (rectify(H1 * inp)))
                else:
                    r_t = O * (rectify(H1 * inp))
                others.append(r_t.npvalue()[1])

            rank = 1
            for o in others:
                if o>gold_res:
                    rank+=1
            mmr += 1.0/rank
            instances += 1
            if rank ==1:
                top1+=1

        return (mmr, instances,top1)

    def build_graph(self, a_s):
        f_init, b_init = [b.initial_state() for b in self.builders]
        src_embed = [self.src_embed_lookup[i] for i in a_s.src_words]
        tag_embed = [self.pos_embed_lookup[i] for i in a_s.src_tags]
        inputs = [concatenate([src_embed[i], tag_embed[i]]) for i in xrange(len(src_embed))]
        fw = [x.output() for x in f_init.add_inputs(inputs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(inputs))]

        H1 = parameter(self.H1)
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)

        errors = []
        for a in a_s.alignments.keys():
            src = a_s.src_words[a]
            translation =  a_s.dst_words[a_s.alignments[a]]
            if  src == self.src_rare or not src in self.src_freq_dict:
                continue # cannot train on this

            k = self.src_freq_dict[src]+' '+a_s.orig_src_tags[a]
            if not k in self.dst_freq_tag_dict:
                continue

            tr_embed = self.dst_embed_lookup[translation]
            inp = concatenate([tr_embed, fw[a], bw[len(src_embed)-1-a]])

            if H2:
                r_t = O * rectify(H2*(rectify(H1 * inp)))
            else:
                r_t = O * (rectify(H1 * inp))
            err = pickneglogsoftmax(r_t, 1)
            errors.append(err)

            neg_samples = random.sample(self.dst_freq_tag_dict[k], min(self.neg,len(self.dst_freq_tag_dict[k])))
            for sample in neg_samples:
                tr_embed = self.dst_embed_lookup[sample]
                inp = concatenate([tr_embed, fw[a], bw[len(src_embed) - 1 - a]])
                if H2:
                    r_t = O * rectify(H2 * (rectify(H1 * inp)))
                else:
                    r_t = O * (rectify(H1 * inp))
                err = pickneglogsoftmax(r_t, 0)
                errors.append(err)

        return errors

    def eval_dev(self, options):
        dr1 = codecs.open(options.dev_src, 'r')
        dr2 = codecs.open(options.dev_dst, 'r')
        da = codecs.open(options.dev_align, 'r')
        l1 = dr1.readline()
        mmr = 0
        instances = 0
        tops = 0
        while l1:
            alignment_instance = AlignmentInstance(l1, dr2.readline(), da.readline(), self.src_word_dict,
                                                   self.dst_word_dict, self.pos_dict, i)
            (v, ins,top1) = self.eval_alignment(alignment_instance)
            instances+= ins
            mmr += v
            tops+= top1
            l1 = dr1.readline()

        mmr = mmr/instances
        tops = tops/instances
        renew_cg()
        print 'mmr:',mmr,'-- tops:',tops, '-- instances:',instances

    def train(self, options):
        renew_cg()
        r1 = codecs.open(options.train_src,'r')
        r2 = codecs.open(options.train_dst,'r')
        a = codecs.open(options.train_align,'r')

        l1 = r1.readline()
        loss = 0
        instances = 0
        i = 0
        errs = []
        while l1:
            i += 1
            alignment_instance = AlignmentInstance(l1, r2.readline(), a.readline(), self.src_word_dict, self.dst_word_dict, self.pos_dict, i)
            errs += self.build_graph(alignment_instance)
            if len(errs)>options.batchsize:
                sum_errs = esum(errs)
                squared = -sum_errs  # * sum_errs
                loss += sum_errs.scalar_value()
                instances += len(alignment_instance.alignments)
                sum_errs.backward()
                self.trainer.update()
                errs = []
                if options.dev_src != None:
                    self.eval_dev(options)
                renew_cg()
            if i%1000==0:
                self.trainer.status()
                print loss / instances
                loss = 0
                instances = 0
            l1 = r1.readline()
        if len(errs) > 0:
            sum_errs = esum(errs)
            squared = -sum_errs  # * sum_errs
            loss += sum_errs.scalar_value()
            instances += len(alignment_instance.alignments)
            sum_errs.backward()
            self.trainer.update()
            self.trainer.status()
            print loss / instances
            renew_cg()

if __name__ == '__main__':
    (options, args) = Expander.parse_options()
    expander = Expander(options)
    if options.train_src!='':
        for i in xrange(options.epochs):
            print 'epoch',i
            expander.train(options)
            print 'saving current epoch'
            expander.model.save(os.path.join(options.output,options.model+'_'+str(i+1)))