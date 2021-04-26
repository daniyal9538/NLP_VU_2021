import spacy
import pandas as pd
import csv
from spacy import displacy
from collections import Counter


def frequencies(nlp, dataset):
    # set up counters
    num_words = 0
    num_tokens = 0
    total_word_length = 0
    types_list = []
    POS_frequencies_coarse = Counter()
    POS_frequencies_fine = Counter()

    # create Bag of Words
    # this is used for POS tagging question, but we now only have to iterate over the data once
    token_count_dict = {}

    for index, tweet in dataset['Tweet text'].iteritems():
        doc = nlp(tweet)

        for sentence in doc.sents:
            POS_tags_coarse = []
            POS_tags_fine = []

            for token in sentence:
                num_tokens += 1
                if token.text not in types_list:
                    types_list.append(token.text)

                # Let's filter out punctuation
                if not token.is_punct:
                    num_words += 1
                    total_word_length += len(token)

                    # POS tags on non punctuation tokens
                    POS_tags_coarse.append(token.pos_)
                    POS_tags_fine.append(token.tag_)

                    # create dict entry if token does not exist yet
                    if token.text not in token_count_dict:
                        token_count_dict[token.text] = {}
                        token_count_dict[token.text]['count'] = 0
                        token_count_dict[token.text]['POS_tag_fine'] = token.tag_
                        token_count_dict[token.text]['POS_tag_coarse'] = token.pos_

                    token_count_dict[token.text]['count'] += 1

            POS_frequencies_coarse.update(POS_tags_coarse)
            POS_frequencies_fine.update(POS_tags_fine)

    print('Number of tokens: ', num_tokens)
    print('Number of types: ', len(types_list))
    print('Number of words: ', num_words)

    print('Average number of words per tweet: ', num_words / len(dataset))
    print('Average word length: ', total_word_length / num_words)

    return POS_frequencies_coarse, POS_frequencies_fine, token_count_dict


def pos_tagging(POS_frequencies_coarse, POS_frequencies_fine, token_count_dict):
    word_frequency_df = pd.DataFrame.from_dict(token_count_dict,
                                               orient='index').reset_index()  # df of frequencies per word + corresponding POS tag
    common_tags = POS_frequencies_fine.most_common(10)
    print(common_tags)

    for tag in list(zip(*common_tags))[0]:
        print(tag)
        df_tag = word_frequency_df.loc[word_frequency_df['POS_tag_fine'] == tag].sort_values(by='count',
                                                                                             ascending=False)
        print(df_tag.iloc[:3])  # print three most occuring tokens
        print(df_tag.iloc[-1])  # print least occuring token


def lemmatization(dataset, nlp):
    for index, tweet in dataset['Tweet text'].iloc[:100].iteritems():
        doc = nlp(tweet)

        for sentence in doc.sents:
            for token in sentence:
                if token.text != token.lemma_:  # only print if the lemma of token is not equal to displayed token
                    print(token.text, token.lemma_, sentence)


def named_entity_recognition(dataset, nlp):
    NER_frequencies = Counter()
    for index, tweet in dataset['Tweet text'].iteritems():
        doc = nlp(tweet)

        NER_list = []
        for ent in doc.ents:
            NER_list.append(ent.label_)

        NER_frequencies.update(NER_list)

    print('Number of named entities: ', len(NER_frequencies))
    print('Number of different entity labels: ', sum(NER_frequencies.values()))

    for index, tweet in dataset['Tweet text'].iloc[:3].iteritems():
        doc = nlp(tweet)
        print(doc.text)
        for ent in doc.ents:
            print(ent.text, ent.label_)
        # displacy.render(doc, style='ent')


def main():
    nlp = spacy.load('en_core_web_sm')
    dataset = pd.read_csv(
        "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskB.txt",
        delimiter='\t',
        quoting=csv.QUOTE_NONE, error_bad_lines=False)
    POS_frequencies_coarse, POS_frequencies_fine, token_count_dict = frequencies(nlp, dataset)
    pos_tagging(POS_frequencies_coarse, POS_frequencies_fine, token_count_dict)
    lemmatization(dataset, nlp)
    named_entity_recognition(dataset, nlp)


if __name__ == '__main__':
    main()
