from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():


    # Importing libraries

    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from tika import parser
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.feature_extraction.text import TfidfVectorizer
    import requests

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


    # Generating TFIDF features

    sentences_2012_tfidf = TfidfVectorizer(max_features = 200).fit_transform(sentences_2012)
    sentences_2014_tfidf = TfidfVectorizer(max_features = 200).fit_transform(sentences_2014)

    # Comparing the TFIDF vectors of sentences from 2014 PDF to the sentences from 2012 PDF
    # Two sentences are similar if the cosine similarity > 0.25

    sent_2014_index = 0
    similarity_list = []
    for test_tfidf in sentences_2014_tfidf:
        cosine_similarities = linear_kernel(test_tfidf, sentences_2012_tfidf).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        for i in range(len(related_docs_indices)):
            if (cosine_similarities[related_docs_indices[i]] >= 0.25):
                similarity_list.append(tuple([sentences_2014[sent_2014_index], sentences_2012[related_docs_indices[i]]]))
        
        sent_2014_index += 1

    # Saving results to a tuple

    similarity_tuple = tuple(similarity_list)

    # Returning a sample sentence from 2014 PDF and its similar sentences from 2012 PDF

    return  'Sample sentence: {} ########################## Similar sentence: {}'.format(similarity_tuple[0][0], similarity_tuple[0][1])


if __name__ == '__main__':
    app.run(debug=True)