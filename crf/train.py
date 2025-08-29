from feature_extractor import FeatureExtractor
from crf import CRF_NER

if __name__ == '__main__':
    data = open('/home/rb071/Documents/named-entity-recognition-vietnamese/data/data/covid_train.txt', 'r', encoding='utf-8').read()
    sents = data.split('\n\n')
    # data = open('/home/rb071/Documents/vietner/vlsp2016_exp/data/test_sample-tab.txt', 'r', encoding='utf-8').read()
    # sents += data.split('\n\n')

    train_data = []
    for sent in sents:
        x = []
        items = sent.split('\n')
        for item in items:
            if not item:
                continue
            try:
                word, tag = item.split('\t')
                word = '_'.join(word.split())
                x.append((word, tag))
            except Exception as e:
                pass
        train_data.append(x)

    feature_extractor = FeatureExtractor()
    X_train, y_train = feature_extractor.extract(train_data)

    model = CRF_NER(
        c1=1.0,
        c2=1e-3,
        max_iterations=200,
        all_possible_transitions=True,
        verbose=True,
    )

    model.fit(X_train, y_train)
    model.save("model.crfsuite")