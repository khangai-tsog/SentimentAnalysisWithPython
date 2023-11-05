import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text
df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()
tokenizer('runners like running and thus they run')

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter('runners like running and thus they run')

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

DataCutOff = 0.8*50000
X_train = df.loc[:DataCutOff, 'review'].values
y_train = df.loc[:DataCutOff, 'sentiment'].values
X_test = df.loc[DataCutOff:, 'review'].values
y_test = df.loc[DataCutOff:, 'sentiment'].values