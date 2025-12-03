from pathlib import Path

INPUT_FILE = Path("hindi_final_merged.conllu")
TRAIN_FILE = Path("train.conllu")
DEV_FILE = Path("dev.conllu")
TEST_FILE = Path("test.conllu")


def read_conllu_sentences(path):
    sentences = []
    current = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if current:
                    sentences.append("".join(current))
                    current = []
            else:
                current.append(line)

        # last sentence (if file doesn't end with blank line)
        if current:
            sentences.append("".join(current))

    return sentences


def write_sentences(path, sentences):
    with path.open("w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent.strip() + "\n\n")


def main():
    print("📖 Reading file:", INPUT_FILE)
    sentences = read_conllu_sentences(INPUT_FILE)

    total = len(sentences)
    print("Total sentences:", total)

    train_size = int(total * 0.70)
    dev_size = int(total * 0.15)
    test_size = total - train_size - dev_size

    train_sents = sentences[:train_size]
    dev_sents = sentences[train_size:train_size + dev_size]
    test_sents = sentences[train_size + dev_size:]

    write_sentences(TRAIN_FILE, train_sents)
    write_sentences(DEV_FILE, dev_sents)
    write_sentences(TEST_FILE, test_sents)

    print("✅ Split completed:")
    print("Train:", len(train_sents))
    print("Dev  :", len(dev_sents))
    print("Test :", len(test_sents))


if __name__ == "__main__":
    main()
    