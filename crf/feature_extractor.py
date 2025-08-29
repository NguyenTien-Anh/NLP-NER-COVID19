from string import punctuation

class FeatureExtractor:
    def extract(self, sentences):
        X = [self.sentence2features(s) for s in sentences]
        y = [self.sentence2lables(s) for s in sentences]
        return X, y

    def sentence2features(self, s):
        return [self.word2features(s, i) for i in range(len(s))]

    def sentence2lables(self, s):
        return [row[-1] for row in s]

    def word2features(self, s, i):
        def get_word(seq, j):
            if j < 0 or j >= len(seq):
                return None
            tok = seq[j]
            return tok[0] if isinstance(tok, (list, tuple)) and len(tok) > 0 else tok

        word = get_word(s, i)
        features = {
            'bias': 1.0,
            '[0]': word,
            '[0].lower': word.lower(),
            '[0].istitle': word.istitle(),
            '[0].isdigit': word.isdigit(),
            '[0].ispunct': word in punctuation,
        }

        # prev-1
        w_m1 = get_word(s, i - 1)
        if w_m1 is not None:
            features.update({
                '[-1]': w_m1,
                '[-1].lower': w_m1.lower(),
                '[-1].istitle': w_m1.istitle(),
                '[-1].isdigit': w_m1.isdigit(),
                '[-1].ispunct': w_m1 in punctuation,
                '[-1,0]': f"{w_m1} {word}",
            })
            # prev-2
            w_m2 = get_word(s, i - 2)
            if w_m2 is not None:
                features.update({
                    '[-2]': w_m2,
                    '[-2].lower': w_m2.lower(),
                    '[-2].istitle': w_m2.istitle(),
                    '[-2].isdigit': w_m2.isdigit(),
                    '[-2].ispunct': w_m2 in punctuation,
                    '[-2,-1]': f"{w_m2} {w_m1}",
                })
        else:
            features['BOS'] = True

        # next+1
        w_p1 = get_word(s, i + 1)
        if w_p1 is not None:
            features.update({
                '[+1]': w_p1,
                '[+1].lower': w_p1.lower(),
                '[+1].istitle': w_p1.istitle(),
                '[+1].isdigit': w_p1.isdigit(),
                '[+1].ispunct': w_p1 in punctuation,
                '[0,+1]': f"{word} {w_p1}",
            })
            # next+2
            w_p2 = get_word(s, i + 2)
            if w_p2 is not None:
                features.update({
                    '[+2]': w_p2,
                    '[+2].lower': w_p2.lower(),
                    '[+2].istitle': w_p2.istitle(),
                    '[+2].isdigit': w_p2.isdigit(),
                    '[+2].ispunct': w_p2 in punctuation,
                    '[+1,+2]': f"{w_p1} {w_p2}",
                })
        else:
            features['EOS'] = True

        return features