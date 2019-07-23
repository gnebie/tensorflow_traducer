from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import json
from tensorflow import keras

def unicode_to_ascii(s):
    # TODO check the unicode normailze
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    """
    remove accents and unused elements
    """
    # TODO check if it's usefull
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    """
    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    """

    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

# Try experimenting with the size of that dataset

# ____________________________________________________start main function


# example_input_batch, example_target_batch = next(iter(dataset))
# print(example_input_batch.shape)
# print(example_target_batch.shape)


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


# sample input
# sample_hidden = encoder.initialize_hidden_state()
# sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
# print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
# print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


# exit(0)

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


# from https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


class NmtWithAttention:

    def __init__(self):
        self.charge_values = False

    def get_train_db(self, path_to_file, num_examples, train_test_ration):

        input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

        # Calculate max_length of the target tensors
        max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
        self.max_length_targ = max_length_targ
        self.max_length_inp = max_length_inp
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=train_test_ration)

        self.prepare_values(inp_lang, targ_lang, input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val)

        return input_tensor_train, target_tensor_train

    def prepare_values(self, inp_lang, targ_lang, input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val):
        self.BUFFER_SIZE = len(input_tensor_train)
        self.steps_per_epoch = len(input_tensor_train)//self.BATCH_SIZE
        self.vocab_inp_size = len(inp_lang.word_index)+1
        self.vocab_tar_size = len(targ_lang.word_index)+1

        print("\nGiven values:")
        print("\tBATCH_SIZE :  " + str(self.BATCH_SIZE))
        print("\tunits :  " + str(self.units))
        print("\tembedding_dim :  " + str(self.embedding_dim))
        print("\nAutogenerate values:")
        print("\tBUFFER_SIZE :  " + str(self.BUFFER_SIZE))
        print("\tvocab_inp_size :  " + str(self.vocab_inp_size))
        print("\tvocab_tar_size :  " + str(self.vocab_tar_size))
        print("\tsteps_per_epoch :  " + str(self.steps_per_epoch))


    def save_training(self):
        if self.charge_values == False:
            print("no model create")
            return
        save_obj = {}
        save_obj["path_to_file"] = self.path_to_file
        save_obj["BUFFER_SIZE"] = self.BUFFER_SIZE
        save_obj["BATCH_SIZE"] = self.BATCH_SIZE
        save_obj["steps_per_epoch"] = self.steps_per_epoch
        save_obj["embedding_dim"] = self.embedding_dim
        save_obj["units"] = self.units
        save_obj["vocab_inp_size"] = self.vocab_inp_size
        save_obj["vocab_tar_size"] = self.vocab_tar_size
        save_obj["max_length_targ"] = self.max_length_targ
        save_obj["max_length_inp"] = self.max_length_inp
        save_obj["inp_lang"] = self.inp_lang.to_json()
        save_obj["targ_lang"] = self.targ_lang.to_json()
        savetmp = json.dumps(save_obj)
        # print(savetmp)
        os.makedirs("config", exist_ok=True)
        saving = os.path.join("config", os.path.basename(self.path_to_file))
        with open(saving, 'w') as save_file:
            save_file.write(savetmp)
        print("saved on " + saving)
        return saving

    def get_load_file(self, training_model_file):
        load_obj = {}
        saving = os.path.join("config", os.path.basename(training_model_file))
        with open(saving, 'r') as save_file:
            load_obj = json.load(save_file)
        # print(json.dumps(load_obj))
        self.path_to_file = load_obj["path_to_file"]
        self.BUFFER_SIZE = load_obj["BUFFER_SIZE"]
        self.BATCH_SIZE = load_obj["BATCH_SIZE"]
        self.steps_per_epoch = load_obj["steps_per_epoch"]
        self.embedding_dim = load_obj["embedding_dim"]
        self.units = load_obj["units"]
        self.vocab_inp_size = load_obj["vocab_inp_size"]
        self.vocab_tar_size = load_obj["vocab_tar_size"]
        self.max_length_targ = load_obj["max_length_targ"]
        self.max_length_inp = load_obj["max_length_inp"]
        self.inp_lang = tokenizer_from_json(load_obj["inp_lang"])
        self.targ_lang = tokenizer_from_json(load_obj["targ_lang"])

    def load_elems(self, training_model_file):
        self.create_encoder_decoder()
        self.create_checkpoint_obj('./training_checkpoints', training_model_file, train=False)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()
        self.charge_values = True

    def load_training(self, training_model_file):
        self.get_load_file(training_model_file)
        self.load_elems(training_model_file)



    def improve_train(self, training_model_file, BATCH_SIZE=None, EPOCHS=None, path_to_file=None, num_examples=None, train_test_ration=0.2):
        self.get_load_file(training_model_file)
        if BATCH_SIZE != None:
            self.BATCH_SIZE = BATCH_SIZE
        if EPOCHS != None:
            self.EPOCHS = EPOCHS
        if path_to_file != None:
            self.path_to_file = path_to_file
        path_to_file = self.path_to_file

        input_tensor_train, target_tensor_train = self.get_train_db(path_to_file, num_examples, train_test_ration)
        # TODO config file
        self.create_dataset(input_tensor_train, target_tensor_train)

        # besoin d'une fonction qui verifie les entreees du dataset par rapport a l'ancien
        self.create_encoder_decoder()
        self.create_checkpoint_obj('./training_checkpoints', training_model_file)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()
        self.charge_values = True
        self.traning_epochs(self.EPOCHS)


    def train(self, path_to_file, BATCH_SIZE=64, embedding_dim=256, units=1024, EPOCHS=10, num_examples=30000, train_test_ration=0.2):
        self.path_to_file = path_to_file
        self.charge_values = True
        self.BATCH_SIZE = BATCH_SIZE
        self.embedding_dim = embedding_dim
        self.units = units
        self.EPOCHS = EPOCHS
        input_tensor_train, target_tensor_train = self.get_train_db(path_to_file, num_examples, train_test_ration)

        # TODO config file
        self.create_dataset(input_tensor_train, target_tensor_train)
        self.create_encoder_decoder()
        self.create_checkpoint_obj('./training_checkpoints', path_to_file)

        # self.steps_per_epoch = steps_per_epoch
        self.traning_epochs(self.EPOCHS)
        # restoring the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))



    def create_dataset(self, input_tensor_train, target_tensor_train):
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(self.BUFFER_SIZE)
        self.dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)

    def create_encoder_decoder(self):
        self.encoder = Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self.decoder = Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)

    def create_checkpoint_obj(self, checkpoint_dir, file_dir=None, train=True):
        """
        create the checkpoint objects
        checkpoint_dir path of the checkpoint folder used
        need to build the encoder and the decoder before
        """
        self.checkpoint_dir = checkpoint_dir
        if file_dir != None:
            self.checkpoint_dir = os.path.join(checkpoint_dir, os.path.basename(file_dir).replace(".", "_"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        if train:
            self.optimizer = tf.keras.optimizers.Adam()
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)
        else:
            self.checkpoint = tf.train.Checkpoint(
                                    encoder=self.encoder,
                                    decoder=self.decoder)


    def traning_epochs(self, epochs):
        for epoch in range(epochs):
          start = time.time()

          enc_hidden = self.encoder.initialize_hidden_state()
          total_loss = 0

          for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
            batch_loss = self.train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
          # saving (checkpoint) the model every 2 epochs
          if (epoch + 1) % 2 == 0:
            self.checkpoint.save(file_prefix = self.checkpoint_prefix)

          print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                              total_loss / self.steps_per_epoch))
          print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
      loss = 0

      with tf.GradientTape() as tape:
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']] * self.BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
          # passing enc_output to the decoder
          predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

          loss += self.loss_function(targ[:, t], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)

      batch_loss = (loss / int(targ.shape[1]))

      variables = self.encoder.trainable_variables + self.decoder.trainable_variables

      gradients = tape.gradient(loss, variables)

      self.optimizer.apply_gradients(zip(gradients, variables))

      return batch_loss

    def loss_function(self, real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = self.loss_object(real, pred)

      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask

      return tf.reduce_mean(loss_)


    def translate(self, sentence, graph=False):
        if self.charge_values == False:
            print("no model create")
            return

        result, sentence, attention_plot = self.evaluate(sentence)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

        if graph == True:
            attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
            plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    # need self.max_length_targ, self.max_length_inp
    # self.inp_lang self.units self.targ_lang
    # self.decoder
    #
    # self.encoder
    #
    def evaluate(self, sentence):
        attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))

        sentence = preprocess_sentence(sentence)

        inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=self.max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']], 0)

        for t in range(self.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.targ_lang.index_word[predicted_id] + ' '

            if self.targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot
