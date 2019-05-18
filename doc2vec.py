from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():

    # Importing libraries

    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from tika import parser
    import requests
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument


    # Request PDF URLs

    # url_2012 = 'https://sso.agc.gov.sg/Act/PDPA2012?ViewType=Pdf'
    url_2012 = input('Enter the URL of the first PDF: ')
    response_2012 = requests.get(url_2012)

    # url_2014 = 'https://sso.agc.gov.sg//SL/PDPA2012-S362-2014?DocDate=20140519&ViewType=Pdf&_=20180520142357'
    url_2014 = input('Enter the URL of the second PDF: ')
    response_2014 = requests.get(url_2014)

    # Save URL content to PDF locally

    with open("sample_2012.pdf",'wb') as f: 
        f.write(response_2012.content)
        
    with open("sample_2014.pdf",'wb') as f: 
        f.write(response_2014.content) 

    # Parsing text from PDF using tika

    text_2012 = parser.from_file('sample_2012.pdf')
    text_2014 = parser.from_file('sample_2014.pdf')

    # Stopwords removal using NLTK

    stop_words = stopwords.words('english')


    # Cleaning text of PDF from the year 2014
    # Meanwhile, specific to the document
    # Must be generalized for broad usage
    # Removal of new lines
    # Removal of footers
    # Removal of headers
    # Removal of headings - text containing 'definitions' text
    # Removal of unnecessary text during initial and end pages of the PDF
    # Removal of excessively small sentences
    # Stopwords removal

    sentences_2014 = sent_tokenize(text_2014['content'])
    sentences_2014 = [sent.replace('\n', '') for sent in sentences_2014]
    sentences_2014 = sentences_2014[15:]
    sentences_2014 = [sent.replace('Informal Consolidation – version in force from 2/7/2014', '') for sent in sentences_2014]
    sentences_2014 = [sent.replace('S 362/2014', '') for sent in sentences_2014]
    sentences_2014 = [sent for sent in sentences_2014 if 'Definitions of this Part' not in sent]
    sentences_2014 = sentences_2014[:42]
    sentences_2014 = [sent for sent in sentences_2014 if len(sent) > 20]
    sentences_2014 = [' '.join(w for w in sent.split() if w not in stop_words) for sent in sentences_2014]


    # Cleaning text of PDF from the year 2012
    # Meanwhile, specific to the document
    # Must be generalized for broad usage
    # Removal of new lines
    # Removal of footers
    # Removal of headers
    # Removal of unnecessary text during initial and end pages of the PDF
    # Removal of excessively small sentences
    # Stopwords removal

    sentences_2012 = sent_tokenize(text_2012['content'])
    sentences_2012 = [sent.replace('\n', '') for sent in sentences_2012]
    sentences_2012 = sentences_2012[74:]
    sentences_2012 = [sent.replace('Informal Consolidation – version in force from 2/10/2016', '') for sent in sentences_2012]
    sentences_2012 = [sent.replace('26 OF 2012', '') for sent in sentences_2012]
    sentences_2012 = sentences_2012[:361]
    sentences_2012 = [sent for sent in sentences_2012 if len(sent) > 20]
    sentences_2012 = [' '.join(w for w in sent.split() if w not in stop_words) for sent in sentences_2012]


    # Data preparation to train Gensim Doc2Vec model

    data = sentences_2012
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    # Saving and loading Doc2Vec
    model.save("d2v.model")
    model= Doc2Vec.load("d2v.model")

    # Computing the similarity of a sample sentence from 2014 PDF and all sentences from 2012 PDF

    test_data = word_tokenize(sentences_2014[0].lower())
    v1 = model.infer_vector(test_data)

    # Returning the two most similar sentences
    
    sims = model.docvecs.most_similar([v1], topn = 2)

    return  'Sample sentence: {} ########################## Similar sentence: {} ######## {}'.format(sentences_2014[0], sentences_2012[int(sims[0][0])], sentences_2012[int(sims[1][0])])


if __name__ == '__main__':
    app.run(debug=True)