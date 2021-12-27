import spacy

nlp = spacy.load("en_core_web_sm")

# doc = nlp(u'I want to know the time that most people work')
# for token in doc:
#     print(token.text, token.pos_, token.dep_)
#
# doc1 = nlp(u'I am Huy. I am handsome.')
# for sentences in doc1.sents:
#     print(sentences)

doc2 = nlp(u'The Google build a Smart House factory for 6$ million')
# for entity in doc2.ents:
#     print(entity)
#     print(entity.label_)
#     print(str(spacy.explain(entity.label_)))

# for chunk in doc2.noun_chunks:
#     print(chunk)

doc3 = nlp(u'the last time i saw him is yesterday, we have seen each other for 5 times')
for token in doc3:
    print("{} {} {}".format(token.text, token.dep_, token.lemma_))
