#import multiprocessing as mp
from sklearn.base import TransformerMixin, BaseEstimator

import json, pandas
import numpy as np

import nltk, nltk.corpus, re, spacy, datetime
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob,Word
from nltk.stem.snowball import SnowballStemmer

import en_core_web_lg

# Suppress SettingWithCopyWarning since no chained assignments
pandas.options.mode.chained_assignment = None

nlp = en_core_web_lg.load()

stoplist = stopwords.words('english')
stopset = set(stoplist)
stopstr = r'\b'+r'\b|\b'.join(stopset) + r'\b'

days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
mos = ['january','february','march','april','may','june','july','august','september','october','november','december']
daymos = days + mos
daymostr = r''
for daymo in daymos:
    if daymo == 'monday':
        daymostr = daymostr + r'\b' + daymo + r'\b|\b' + daymo[:3] + r'\b'
    elif daymo == 'thursday':
        daymostr = daymostr + r'|\b' + daymo + r'\b|\b' + daymo[:4] + r'\b|\b' + daymo[:5] + '\b'
    elif daymo in ['tuesday','september']:
        daymostr = daymostr + r'|\b' + daymo + r'\b|\b' + daymo[:3] + r'\b|\b' + daymo[:4] + '\b'
    else:
        daymostr = daymostr + r'|\b' + daymo + r'\b|\b' + daymo[:3] + r'\b'
        
#-----------------------------------------------------------------------------------------        
    
class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 purpose='clean',
                 #n_jobs=1,
                 num=False,
                 date = False,
                 url_email = False,
                 calendar = False,
                 punctuation = False,
                 capitalize = False,
                 stopword = False):
        """
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Punctuation removal
            3. Stop words removal
            4. Lemmatization
        
        n_jobs - parallel jobs to run
        """
        self.purpose = purpose
        #self.n_jobs = n_jobs
        self.num = num
        self.date = date
        self.url_email = url_email
        self.calendar = calendar
        self.punctuation = punctuation
        self.capitalize = capitalize
        self.stopword = stopword
        
    def fit(self, X, y=None):
        return self

#     def transform(self, X, *_):
#         X_copy = X.copy()

#         partitions = 1
#         cores = mp.cpu_count()
#         if self.n_jobs <= -1:
#             partitions = cores
#         elif self.n_jobs <= 0:
#             return X_copy.apply(self._preprocess_text)
#         else:
#             partitions = min(self.n_jobs, cores)

#         data_split = np.array_split(X_copy, partitions)
#         pool = mp.Pool(cores)
#         data = pd.concat(pool.map(self._preprocess_part, data_split))
#         pool.close()
#         pool.join()

#         return data

#     def _preprocess_part(self, part):
#         return part.apply(self._preprocess_text)

    def transform(self, docs_df):
        return self._preprocess_text(docs_df)
    
    def _preprocess_text(self, docs):
        docs['body'] = docs['text'].apply(lambda x: self._extract(x))

        if (self.purpose == 'sqlsp') or (self.purpose == 'clinical'):
            docs = docs.rename(columns={'body':'cleantext'})
            output = docs[['id','cleantext']]

        else:
            docs['nlp'] = self._spacify(docs['body'])
            
            if self.url_email == False:
                docs['body'] = docs['nlp'].apply(lambda x: self._url_email_remove(x))

            if self.date == False:
                docs['ents'] = docs['nlp'].apply(lambda x: x.ents if x else None)
                docs['datetexts'] = docs['ents'].apply(lambda x: self._ent_label_match(x,'DATE'))
                docs['body'] = docs.apply(self._remove_dates, axis = 1)

            if self.capitalize == False:
                docs['body'] = docs['body'].apply(lambda x: x.lower())

            if self.calendar == False:
                docs['body'] = docs['body'].apply(lambda x: re.sub(daymostr,'',x.lower()))

            # 1 or 2 char word removal - PRE-PUNCTUATION REMOVAL
            if self.purpose == 'ml':
                docs['body'] = docs['body'].apply(lambda x: re.sub(r"\b(\w\w?)\b",'',x))

            if self.punctuation == False:
                docs['body'] = docs['body'].apply(lambda x: re.sub(r'[^\w\s&]','',x))

            if self.num == False:
                docs['body'] = docs['body'].apply(lambda x: self._remove_nums(x))

            # Single char word removal - POST-PUNCTUATION REMOVAL
            if self.purpose == 'ml':
                docs['body'] = docs['body'].apply(lambda x: re.sub(r"\b(\w)\b",'',x))

            if self.stopword == False:
                docs['body'] = docs['body'].apply(lambda x: re.sub(stopstr,'',x))

            # Remove extra whitespace
            docs['body'] = docs['body'].apply(lambda x: re.sub(r" +",' ',x))

            if self.purpose == 'ml': 
                # Part of Speech tagging
                docs['POS'] = docs['body'].apply(lambda x: TextBlob(x).tags)
                # Lemmatization
                docs['lemma'] = docs['POS'].apply(lambda x: 
                                                  " ".join([Word(word).lemmatize(self._posmap(pos.lower())) for word,pos in x]))
                # Stemming
                docs['stem'] = docs['lemma'].apply(lambda x: self._stemmer(x))
                docs['cleantext'] = docs['stem'].apply(lambda x: x.strip())
            else:
                docs['cleantext'] = docs['body'].apply(lambda x: x.strip())

            output = docs[['id','cleantext']]
            
        return output
    
    def _ent_label_match(self,ents,label):
        texts = []
        if ents:
            for ent in ents:
                if ent.label_ == label:
                    texts.append(ent.text)
        return texts
        
    def _extract(self,document):
        lines = document.split('\n')
        maxindex = len(lines)
        body = ""
        for idx,rawline in enumerate(lines):
            line = rawline.strip()
            if line.startswith('About'):
                maxindex = idx
            else:
                None
            if idx <= (maxindex - 1):
                body = body + line + '\n'
            else:
                None
        return body.strip()
    
    def _spacify(self,bodies):
        spacers = []
        for pr in bodies:
            spacers.append(nlp(pr))
        return spacers

    def _url_email_remove(self,doc):
        outlist = []
        for word in doc:
            if not word.like_url:
                if not word.like_email:
                    outlist.append(word.text)
        return ' '.join(outlist)

    def _remove_dates(self,row):
        bod = row['body']
        datetxts = row['datetexts']
        outputs = bod
        if len(datetxts)>0:
            try:
                phrases = r"\b"+r"\b|\b".join([re.escape(phrase) for phrase in datetxts]) + r"\b"
                outputs = re.sub(phrases,'',bod)
            except:
                print('Possible special char in regex error')
                print(datetxts)
        return outputs

    def _remove_nums(self,body):
        no_nums = body.translate({ord(ch): None for ch in '0123456789'})
        return no_nums

    def _posmap(self,tag):
        if tag in ['jj','jjr','jjs']:
            return 'a'
        elif tag in ['nn','nns','nnp','nnps','prp','prp$','wp','wp$']:
            return 'n'
        elif tag in ['md','vb','vbz','vbp','vbd','vbn','vbg']:
            return 'v'
        elif tag in ['rb','rbr','rbs','rp','wrb']:
            return 'r'
        else:
            None

    def _stemmer(self,body):
        stemmed = ''
        for word in body.split():
            stemmed = stemmed + ' ' + SnowballStemmer("english").stem(word)
        return stemmed.strip()
        

