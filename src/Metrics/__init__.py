import pickle
import os
from functools import lru_cache
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import sacrebleu
import bert_score
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from Metrics.bleu.bleu import Bleu
from Metrics.cider.cider import Cider
from Metrics.meteor.meteor import Meteor
from Metrics.spice.spice import Spice


stride = 512


@lru_cache(4)
def init_java():
    # os.environ["JAVA_HOME"] = ""
    # os.environ["CLASSPATH"] = ""
    # os.environ["PATH"] += ""
    # os.environ["LD_LIBARY_PATH"] = ""
    pass


@lru_cache(32)
def get_model(name):
    if name == "tfidf":
        with open("Metrics/vectorizer.pkl", "rb") as f1, open("Metrics/tfidf.pkl", "rb") as f2:
            return pickle.load(f1), pickle.load(f2)
    model = AutoModelWithLMHead.from_pretrained(name, cache_dir="cache").cuda()
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir="cache")
    return model, tokenizer


def sentence_ppl(model, encoding):
    max_length = model.config.n_positions
    nlls = []
    for i in range(0, encoding.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encoding.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encoding.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len
        nlls.append(neg_log_likelihood)
    nll = torch.stack(nlls).sum() if len(nlls) > 0 else torch.tensor(0)
    return nll


def gpt_corpus_ppl(corpus):     # [[s11,s12...],[s21,s22...]]
    model, tokenizer = get_model("gpt2")
    corpus = [s for l in corpus for s in l if s.strip()]
    corpus = [tokenizer(s, return_tensors="pt") for s in corpus]
    nll = sum([sentence_ppl(model, s) for s in corpus]) / sum([s.input_ids.size(1) for s in corpus])
    ppl = torch.exp(nll)
    return ppl


def bert_sentence_ppl(model, tokenizer, sentence):
    with torch.no_grad():
        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
        nll = model(masked_input.cuda(), labels=labels.cuda())[0]
    return nll, repeat_input.shape[0]


def bert_corpus_ppl(corpus):     # [[s11,s12...],[s21,s22...]]
    model, tokenizer = get_model("bert-base-uncased")
    corpus = [s for l in corpus for s in l if s.strip()]
    res = [bert_sentence_ppl(model, tokenizer, s) for s in corpus]
    ppl = sum(nll*length for nll, length in res) / sum(length for nll, length in res)
    # ppl = torch.exp(nll)
    return ppl


def calculate_ppl(generated):
    return {"BERT-PPL": bert_corpus_ppl(generated).item(), "GPT-PPL": gpt_corpus_ppl(generated).item()}


def tf_idf(generated, reference):
    vectorizer, tfidf = get_model("tfidf")
    reference = tfidf.transform(vectorizer.transform(reference))
    generated = tfidf.transform(vectorizer.transform(generated[0]))
    assert reference.shape[0] == generated.shape[0]
    score = cosine_similarity(reference, generated).trace()/reference.shape[0]
    return {"TF-IDF": score}


def bert_similarity(generated, reference):
    model, tokenizer = get_model("bert-base-uncased")
    cls_scores, mean_scores = [], []
    for r, g in zip(reference, generated[0]):
        r, g = tokenizer(r, return_tensors='pt'), tokenizer(g, return_tensors='pt')
        r = model(input_ids=r.input_ids[:, :512].cuda(), output_hidden_states=True).hidden_states[-1].squeeze()
        g = model(input_ids=g.input_ids[:, :512].cuda(), output_hidden_states=True).hidden_states[-1].squeeze()
        cls_scores.append(torch.cosine_similarity(r[0], g[0], dim=0))
        mean_scores.append(torch.cosine_similarity(r.mean(dim=0), g.mean(dim=0), dim=0))
    return {"BERT-CLS": torch.tensor(cls_scores).mean().item(), "BERT-AVG": torch.tensor(mean_scores).mean().item()}


def get_bert_score(generated, reference):
    P, R, F = bert_score.score(generated[0], reference, model_type="bert-base-uncased")
    return {"BERT-Score": F.mean().item()}


def get_rouge(generated, reference):
    metrics = ["rouge-1", "rouge-2", "rouge-l"]
    scores = Rouge(metrics).get_scores(generated[0], reference, avg=True, ignore_empty=True)
    return {k.upper(): scores[k]["f"] for k in metrics}


def cider(generated, reference):
    generated = generated[0]
    generated = dict([(i, [generated[i]]) for i in range(len(generated))])
    reference = dict([(i, [reference[i]]) for i in range(len(reference))])
    score, scores = Cider().compute_score(reference, generated)
    return {"CIDEr": score}


def meteor(generated, reference):
    generated = generated[0]
    generated = dict([(i, [generated[i]]) for i in range(len(generated))])
    reference = dict([(i, [reference[i]]) for i in range(len(reference))])
    score, scores = Meteor().compute_score(reference, generated)
    return {"METEOR": score}


def spice(generated, reference):
    generated = generated[0]
    all_scores, nums = [], []
    for i in range(0, len(reference), 24):
        sys = dict([(i, [s]) for i, s in enumerate(generated[i:i+24])])
        refs = dict([(i, [s]) for i, s in enumerate(reference[i:i+24])])
        score, scores = Spice().compute_score(refs, sys)
        all_scores.append(score)
        nums.append(len(refs))
    score = sum(all_scores[i]*nums[i] for i in range(len(nums))) / sum(nums)
    return {"SPICE": score}


def bleus(generated, reference):
    bleu_obj = sacrebleu.corpus_bleu(reference, generated)
    bleu_scores = {"BLEU": bleu_obj.score, "BLEU format": bleu_obj.format()}
    for n in [1, 2, 3, 4]:
        bleu_scores["BLEU-%d" % n] = bleu_obj.precisions[n-1]
    return bleu_scores


def chrf(generated, reference):
    chrf_obj = sacrebleu.corpus_chrf(reference, generated)
    return {"CHRF": chrf_obj.score}


def ter(generated, reference):
    ter_obj = sacrebleu.corpus_ter(reference, generated)
    return {"TER": ter_obj.score}


def print_format_score(scores, **prefix):
    metrics = ["BERT-PPL", "GPT-PPL", "BLEU", "CHRF", "TER", "ROUGE-L", "METEOR", "TF-IDF", "CIDEr",
               "BERT-CLS", "BERT-AVG", "BERT-Score", "BLEU-1", "BLEU-4", "ROUGE-1", "ROUGE-2", "BLEU-2", "BLEU-3"]
    prefix_keys = ["%-8s" % k for k in sorted(prefix.keys())]
    prefix_values = ["%-8.4g" % prefix[k] for k in sorted(prefix.keys())]
    print(*prefix_keys, *["%-8s" % k for k in metrics], sep="\t")
    print(*prefix_values, *["%-8.4g" % scores[k] for k in metrics], sep="\t")


def all_metrics(generated, reference, desc="", **prefix):
    init_java()
    res = {k: prefix[k] for k in prefix}
    generated = [[" " if s in ["", "."] else s for s in l] for l in zip(*generated)]
    res.update(calculate_ppl(generated))
    res.update(bleus(generated, reference))
    res.update(chrf(generated, reference))
    res.update(ter(generated, reference))
    res.update(tf_idf(generated, reference))
    res.update(bert_similarity(generated, reference))
    res.update(get_bert_score(generated, reference))
    res.update(get_rouge(generated, reference))
    res.update(cider(generated, reference))
    res.update(meteor(generated, reference))
    # res.update(spice(generated, reference))
    print(desc)
    print_format_score(res, **prefix)
    return res
