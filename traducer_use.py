
from traducer_nmt_with_attention import NmtWithAttention

def main():
    test = NmtWithAttention()
    test.load_training("fra.txt")
    while True:
        sentence = input()
        if (sentence == "exit()" or sentence == "quit()"):
            return
        test.translate(sentence)

if __name__ == '__main__':
    main()
