
from traducer_nmt_with_attention import NmtWithAttention

def main():
    test = NmtWithAttention()
    test.load_training("traducer.csv")
    test.translate(u"j'aime les carottes")

if __name__ == '__main__':
    main()
