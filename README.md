Unsupervised En-Ko Translation Model

supervised:
perplexity: 38
accuracy: 0.32
command:
python translation_sup.py -Tts data/train_en_UNK.txt data/train_ko_UNK.txt data/test_en_UNK.txt data/test_ko_UNK.txt -m "*2en*"

unsupervised:
accuracy: 0.10
command:
python translation_unsup.py -Tts data/train_en_UNK.txt data/train_ko_UNK.txt data/test_en_UNK.txt data/test_ko_UNK.txt

overleaf: 
