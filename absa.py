from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
import spacy
import pandas as pd
import numpy as np

# Change this path to wherever your model-best folder is located
ner = spacy.load(r"model-best")


def absa(documents):
    columns = ['review_id', 'dish', 'sentiment', 'confidence']
    output = pd.DataFrame(columns=columns)

    for i, doc in enumerate(documents):
        processed = ner(doc)

        for ent in processed.ents:
            doc = doc.replace(ent.text, f"[B-ASP]{ent.text}[E-ASP]")

        sent_cls = SentimentClassifier(checkpoint='english')
        res = sent_cls.predict(text=doc, print_result=False)

        for j, dish in enumerate(res['aspect']):
            row = {'review_id': [i],
                   'dish': [dish],
                   'sentiment': [res['sentiment'][j]],
                   'confidence': [res['confidence'][j]]
                   }
            output = pd.concat([output, pd.DataFrame(row)], axis=0)

    output.set_index(np.arange(output.shape[0]), inplace=True)

    return output
