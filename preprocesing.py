from Data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import TextVectorization
import neattext as  neattext
data=Data()
x_data = data.getx_data()
y_data = data.gety_data()
class preprocessing:
    def cleaning_data(self,x_data):
        self.x_data=x_data
        self.x_data=self.x_data.apply(neattext.remove_urls)
        self.x_data=self.x_data.apply(neattext.remove_html_tags)
        self.x_data=self.x_data.apply(neattext.remove_hashtags)
        self.x_data=self.x_data.apply(neattext.remove_special_characters)
        return self.x_data
    def text_tokenization(self,x_data):
        self.max_fatures = 2000
        tokenizer = Tokenizer(num_words=self.max_fatures, split=' ')
        tokenizer.fit_on_texts(self.x_data.values)
        self.X_data = tokenizer.texts_to_sequences(self.x_data.values)
        self.X_pad_data= pad_sequences(self.X_data)
        print(self.X_pad_data[0],len(self.X_pad_data[0]))
        return self.X_pad_data
    def text_vextorization(self,x_data):
        self.vectorizer = TextVectorization(max_tokens=20000,
                                       output_sequence_length=180,
                                       output_mode='int')

        self.vectorizer.adapt(self.x_data.values)
        self.vectorized_text = self.vectorizer(self.x_data.values)
        print(self.vectorized_text[1],len(self.vectorized_text[1]))
        return self.vectorized_text
    def get_pad_data(self):
        return self.X_pad_data
    def get_vectorizer_data(self):
        return self.vectorized_text

