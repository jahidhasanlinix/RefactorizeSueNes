import json, os
from summ_eval.supert_metric import SupertMetric
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.blanc_metric import BlancMetric
def calc_one(hyp, refs, scorers):
    score = {}
    for scorer in scorers:
        score.update(scorer.evaluate_example(hyp, refs))

    return score

def main():
    scorers = [BlancMetric()]

    in_file = 'test.tsv'

    docs = []
    sums = []
    with open(os.path.join(in_file), 'r', encoding='utf-8') as f:
        for line in f:
            texts = line.strip().split('\t')
            article = texts[0]
            summaries = texts[:-1]
            
            docs.extend([article] * len(summaries))
            sums.extend(summaries)
    
    for scorer in scorers:
        scores = scorer.evaluate_batch(sums, docs, aggregate=False)
        scorer_name = list(scores[0].keys())[0]
        with open(os.path.join("predictions", "metric_"+scorer_name+".tsv"), "w", encoding="utf-8") as f:
            for score in scores:
                f.write(str(score[scorer_name])+"\n")
            
if __name__ == '__main__':
    main()
