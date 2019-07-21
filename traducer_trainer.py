

from traducer_nmt_with_attention import NmtWithAttention

def main():
    test = NmtWithAttention()
    test.train("db/traducer.csv")
    test.save_training()
    test.translate(u"j'aime les carottes")


if __name__ == '__main__':
    main()
