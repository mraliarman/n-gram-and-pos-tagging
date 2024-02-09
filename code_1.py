import pandas as pd
from collections import Counter
from nltk import ngrams, pos_tag
from hazm import sent_tokenize, word_tokenize, stopwords_list, Normalizer
from nltk.probability import LaplaceProbDist, SimpleGoodTuringProbDist, FreqDist

# Step 0: Open the Excel file
file_name = "digikala_comment.xlsx"
data_file = pd.read_excel(file_name)
data_file['comment'] = data_file['comment'].astype(str)

# Step 1: Preprocessing
# Step 1-1: Tokenize comments into sentences
print('step 1-1')
data_file['comment_sentences'] = data_file['comment'].apply(sent_tokenize)

# Step 1-2: Remove extra spaces
print('step 1-2')
data_file['comment_sentences'] = data_file['comment_sentences'].apply(lambda sentences: [sentence.strip() for sentence in sentences])

# Step 1-3: Tokenize, remove stop words, non-alphabetic characters, HTML tags, emojis, and normalize text
print('step 1-3')
stopwords = set(stopwords_list())
normalizer = Normalizer()
data_file['tokenized_comment'] = data_file['comment_sentences'].apply(lambda sentences: [
    [normalizer.normalize(word.replace('\u200c', ' ')) for word in word_tokenize(sentence) if word.isalpha() and word not in stopwords]
    for sentence in sentences
])

# Step 1-4: Replace numbers with <NUM> and URLs with <URL>
print('step 1-4')
data_file['tokenized_comment'] = data_file['tokenized_comment'].apply(lambda sentences: [
    ['<NUM>' if word.isdigit() else '<URL>' if word.startswith(('http', 'www')) else word for word in sentence]
    for sentence in sentences
])

# Step 2: Create language models
# Step 2-1: Create n-grams
print('step 2-1')
def get_ngrams(data, n):
    ngram_list = [ngrams(sentence, n) for sentence in data]
    ngrams_flat = [ng for sent in ngram_list for ng in sent]
    return ngrams_flat

unigrams = get_ngrams(data_file['tokenized_comment'], 1)
bigrams = get_ngrams(data_file['tokenized_comment'], 2)
trigrams = get_ngrams(data_file['tokenized_comment'], 3)

# Print each n-gram count
print('Unigram Count:', len(unigrams))
print('Bigram Count:', len(bigrams))
print('Trigram Count:', len(trigrams))

# Print each n-gram
print('Sample Unigrams:', unigrams[:10])
print('Sample Bigrams:', bigrams[:10])
print('Sample Trigrams:', trigrams[:10])

# Step 2-2: Convert lists to tuples
flat_unigrams = [tuple(word) for gram in unigrams for word in gram]
flat_bigrams = [tuple(word) for gram in bigrams for word in gram]
flat_trigrams = [tuple(word) for gram in trigrams for word in gram]

freq_unigrams = FreqDist(flat_unigrams)
freq_bigrams = FreqDist(flat_bigrams)
freq_trigrams = FreqDist(flat_trigrams)

# Print top 10 most common unigrams, bigrams, and trigrams
print('Top 10 Unigrams:', freq_unigrams.most_common(10))
print('Top 10 Bigrams:', freq_bigrams.most_common(10))
print('Top 10 Trigrams:', freq_trigrams.most_common(10))

# Step 2-3: Calculate frequencies
print('step 2-3')
import math
def calculate_perplexity(model, test_data):
    N = sum(len(sentence) for sentence in test_data)
    cross_entropy = -sum(math.log2(model.prob(word)) for sentence in test_data for word in sentence) / N
    perplexity = 2 ** cross_entropy
    return perplexity

test_sentences = [
    "بوی تند ولی خوشبو داره",
    "بلوتوثش کار نمی‌کنه حالا تا بدستم رسیده باید برش گردونم",
    "بلند گوهاش بیس بالا و صدای زیادی بمی‌داره که بعد از مدتی باعث خسته شدن مغز آدم می‌شه",
    "لطفاً کالای مورد نظر رو در پیشنهاد ویژه قرار بدید"
]

for i, sentence in enumerate(test_sentences, 1):
    laplace_unigram = LaplaceProbDist(freq_unigrams, bins=len(freq_unigrams))
    perplexity_unigram = calculate_perplexity(laplace_unigram, [sentence])

    laplace_bigram = LaplaceProbDist(freq_bigrams, bins=len(freq_bigrams))
    perplexity_bigram = calculate_perplexity(laplace_bigram, [sentence])

    laplace_trigram = LaplaceProbDist(freq_trigrams, bins=len(freq_trigrams))
    perplexity_trigram = calculate_perplexity(laplace_trigram, [sentence])

    print(f"Sentence {i} - Laplace Unigram Perplexity: {perplexity_unigram}")
    print(f"Sentence {i} - Laplace Bigram Perplexity: {perplexity_bigram}")
    print(f"Sentence {i} - Laplace Trigram Perplexity: {perplexity_trigram}")
    print('-' * 50)

# Step 2-4: Word Prediction
print('step 2-4')
def predict_next_words(model, input_sequence, length=15):
    for i in range(length):
        next_word = model.generate()
        input_sequence.append(next_word)

    return input_sequence

for sentence in ["صرفە جویی در پودر ماشين", "یکی از چراغهاى وضعيت", "گوشی سامسونگ", "رنگ قرمز كفش", "یک تن ماهی خوب"]:
    print(sentence)
    print(predict_next_words(laplace_unigram, sentence.split()))
    print('one \n')
    print(predict_next_words(laplace_bigram, sentence.split()))
    print('two \n')
    print(predict_next_words(laplace_trigram, sentence.split()))
    print('three \n')


# Step 5-2: Calculate perplexity for generated sentences
print('step 5-2')
generated_sentences = [
    predict_next_words(laplace_unigram, "صرفە جویی در پودر ماشين".split()),
    predict_next_words(laplace_bigram, "یکی از چراغهاى وضعيت".split()),
    predict_next_words(laplace_trigram, "گوشی سامسونگ".split()),
    predict_next_words(laplace_unigram, "رنگ قرمز كفش".split()),
    predict_next_words(laplace_trigram, "یک تن ماهی خوب".split())
]

perplexity_generated_unigram = calculate_perplexity(laplace_unigram, generated_sentences)
perplexity_generated_bigram = calculate_perplexity(laplace_bigram, generated_sentences)
perplexity_generated_trigram = calculate_perplexity(laplace_trigram, generated_sentences)

print(f"Perplexity of Generated Unigram Sentences: {perplexity_generated_unigram}")
print(f"Perplexity of Generated Bigram Sentences: {perplexity_generated_bigram}")
print(f"Perplexity of Generated Trigram Sentences: {perplexity_generated_trigram}")

# Step 3: POS Tagging
# Step 3-1: Perform POS tagging on the preprocessed data
print('step 3-1')
data_file['pos_tags'] = data_file['tokenized_comment'].apply(lambda sentences: [pos_tag(sentence) for sentence in sentences])

with open('pos.txt', 'w', encoding='utf-8') as file:
    for sentence_tags in data_file['pos_tags']:
        for tags in sentence_tags:
            sentence = ' '.join([f"{word} --> {tag}" for word, tag in tags])
            file.write(sentence + '\n')
        file.write('\n')

# Step 3-2: Count occurrences of each POS tag
print('step 3-2')
pos_tags_flat = [tag[1] for sentence in data_file['pos_tags'].sum() for tag in sentence]
pos_tags_count = Counter(pos_tags_flat)
print(f"3-2: POS Tags Count: {pos_tags_count}")

# Step 3-3: Extract and count proper nouns (NNP)
print('step 3-3')
proper_nouns = [word[0] for sentence in data_file['pos_tags'].sum() for word in sentence if word[1] == 'NNP']
proper_nouns_count = Counter(proper_nouns)

print(f"3-3: Top 15 Proper Nouns: {proper_nouns_count.most_common(15)}")
