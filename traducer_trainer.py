

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
    test.train(path_to_file,BATCH_SIZE=20, embedding_dim=256, units=1024, EPOCHS=0, num_examples=3000000)
    test.save_training()
    # test.translate(u"j'aime les carottes")


if __name__ == '__main__':
    main()
