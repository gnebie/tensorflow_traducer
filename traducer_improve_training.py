

from traducer_nmt_with_attention import NmtWithAttention

import tensorflow as tf
import os

def main():
    path_to_file = "db/traducer.csv"
    # path_to_zip = tf.keras.utils.get_file(
    # 'fr-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fr-eng.zip',
    # extract=True)

    # path_to_file = os.path.dirname(path_to_zip)+"/fra-eng/fra.txt"
    path_to_file = "db/fra-eng/fra.txt"
    print(path_to_file)
    test = NmtWithAttention()
    test.improve_train(path_to_file, BATCH_SIZE=60, EPOCHS=4, num_examples=1000)
    test.save_training()
    # test.translate(u"j'aime les carottes")


if __name__ == '__main__':
    main()
