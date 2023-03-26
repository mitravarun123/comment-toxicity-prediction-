import tensorflow as tf
import gradio as gr
from Model import Model
from keras.layers import TextVectorization
from Data import Data
model=Model()
model.get_data()
model.cleaning_and_preprocessing()
model.train_test_val_partition()
model.create_model()
model.complie_model()
model.fit_model()
model.ploting_model_prefromence()
my_model=tf.keras.models.load_model("toxicity.h5")
vectorizer = TextVectorization(max_tokens=20000,
                               output_sequence_length=180,
                               output_mode='int')

data=Data()
y_data=data.gety_data()
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = my_model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(y_data):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    return text
interface = gr.Interface(fn=score_comment,
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

interface.launch(share=True)