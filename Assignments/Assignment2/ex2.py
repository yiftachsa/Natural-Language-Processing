import re, sys, random, math, collections, nltk
from collections import Counter, defaultdict
from nltk import ngrams


# noinspection PyMethodMayBeStatic,SpellCheckingInspection
class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable. The language model should suppport the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """
        if not lm is None:
            self.lm = lm
            self.update_fields()
        else:
            self.lm = None
            self.vocab = None
            self.unigrams = None
            self.n = -1

        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        self.error_types = ["insertion", "deletion", "substitution", "transposition"]
        self.error_tables = {error_type: defaultdict(int) for error_type in self.error_types}

    def update_fields(self):
        """
        Updates fields after insertion of a new langauge model
        """
        self.update_vocab()
        self.update_n()
        self.update_char_unigrams()
        self.update_char_bigrams()

    def update_char_bigrams(self):
        """ Counts all the character-level bigrams and their apperances in the corpus"""
        self.char_bigrams = defaultdict(int)
        for key, total_count in self.lm.get_model().items():
            bi_count = {''.join(c_key): c_value for c_key, c_value in Counter(ngrams(key, 2)).items()}
            for bigram, count in bi_count.items():
                self.char_bigrams[bigram] = self.char_bigrams[bigram] + count * total_count

    def update_char_unigrams(self):
        """
        Counts all the character-level unigrams and their apperances in the corpus
        """
        self.char_unigrams = defaultdict(int)
        for key, total_count in self.lm.get_model().items():
            for char, count in Counter(key).items():
                self.char_unigrams[char] = self.char_unigrams[char] + count * total_count

    def count_unigram(self, unigram):
        """
        Returns the count of a given char unigram based on the lm dictionary

        Args:
            unigram(str): str of length 1

        Returns:
            (int) the number of occurences of the char in the lm's corpus
        """
        if unigram == '#':
            unigram = " "
        if unigram in self.char_unigrams:
            return self.char_unigrams[unigram]
        else:
            return 0

    def count_bigram(self, bigram):
        """
        Returns the count of a given char bigram based on the lm dictionary

        Args:
            bigram(str): str of length 2

        Returns:
            (int) the number of occurences of the bigram in the lm's corpus
        """
        if bigram[0] == '#':
            bigram = " " + bigram[1]
        if bigram in self.char_bigrams:
            return self.char_bigrams[bigram]
        else:
            return 0

    def get_unigram_prob(self, w, smooth=True):
        """
        Calculates the log likelihood probability of a given unigram word based on the lm dictionary

        Args:
            w (str): a unigram word
            smooth (bool): if True use Laplace smoothing

        Returns:
            (float) log likelihood probability of a given unigram
        """
        if not self.unigrams:
            # calculate unigram dict
            self.unigrams = defaultdict(int)

            for ngram, total_count in self.lm.get_model().items():
                for unigram, count in Counter(ngram.split()).items():
                    self.unigrams[unigram] = self.unigrams[unigram] + count * total_count

        if smooth:
            if w in self.unigrams:
                count = self.unigrams[w] + 1
            else:  # Laplace
                count = 1
            total = sum(self.unigrams.values()) + len(self.unigrams)
            return math.log(count / total)

    def build_model(self, text, n=3):
        """Returns a language model object built on the specified text. The language
            model should support evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """

        nt = normalize_text(text)  # lower casing, padding punctuation with white spaces
        lm = Ngram_Language_Model(n=n)
        lm.build_model(nt)
        self.lm = lm
        self.update_fields()
        return lm

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM disctionary if set)

            Args:
                ls: a language model object
        """
        if lm is not None:
            self.lm = lm
            self.update_fields()

    def is_sub_error(self, candidate, original_word):
        """
            Recieves a candidate word and an original typed word and returns True if the
            original word is the result of substition error in the candidate word

        Args:
            candidate (str): a candidate word
            original_word(str): a word with possible error

        Returns:
            (bool) True if the original word is the result of
            substition error in the candidate word, otherwise False

        """
        comparisons = [candidate[i] != original_word[i] for i in range(len(original_word))]
        return sum(comparisons) == 1

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {str: int} represents the confution matrix of the
            specific errors, where str is a string of two characters mattching the
            row and culumn "indixes" in the relevant confusion matrix and the int is the
            observed count of such an error (computed from the specified errors file).
            Examples of such string are 'xy', for deletion of a 'y'
            after an 'x', insertion of a 'y' after an 'x'  and substitution
            of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy' indicates the characters that are transposed.


            Notes:
                1. Ultimately, one can use only 'deletion' and 'insertion' and have
                    'substitution' and 'transposition' derived. Again,  we use all
                    four types explicitly in order to keep things simple.
            Args:
                errors_file (str): full path to the errors file. File format, TSV:
                                    <error>    <correct>


            Returns:
                A dictionary of confusion "matrices" by error type (dict).
        """
        typo_correct_pairs = [line.split() for line in open(errors_file).read().split("\n") if
                              len(line.split()) == 2]
        for typo, correct in typo_correct_pairs:
            if len(typo) > len(correct):
                # insertion error
                if self.get_insert_err(correct, typo) not in self.error_tables["insertion"]:
                    self.error_tables["insertion"][self.get_insert_err(correct, typo)] = 0
                self.error_tables["insertion"][self.get_insert_err(correct, typo)] += 1
            elif len(typo) < len(correct):
                # deletion error
                for error in self.get_del_err(correct, typo):
                    if error not in self.error_tables["deletion"]:
                        self.error_tables["deletion"][error] = 0
                    self.error_tables["deletion"][error] += 1
            elif self.is_sub_error(correct, typo):
                if self.get_sub_err(correct, typo) not in self.error_tables["substitution"]:
                    self.error_tables["substitution"][self.get_sub_err(correct, typo)] = 0
                self.error_tables["substitution"][self.get_sub_err(correct, typo)] += 1
            else:
                if self.get_trans_err(correct, typo) not in self.error_tables["transposition"]:
                    self.error_tables["transposition"][self.get_trans_err(correct, typo)] = 0
                self.error_tables["transposition"][self.get_trans_err(correct, typo)] += 1

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        for error_type in self.error_types:
            self.error_tables[error_type] = dict(
                Counter(self.error_tables[error_type]) + Counter(error_tables[error_type]))

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        self.lm.evaluate(text)

    def update_n(self):
        """
        Updates n - the length of the ngram used by the language model
        """
        if self.lm is not None:
            n_gram = list(self.lm.get_model().keys())[0]
            self.n = len(n_gram.split())

    def update_vocab(self):
        """
        Updates the vocabulary of the corpus used to train the language model
        """
        if self.lm is not None:
            self.vocab = []  # delete the old vocab
            for key in self.lm.get_model().keys():
                a = key.split()
                self.vocab.extend([x.lower() for x in a])
            self.vocab = set(self.vocab)

    def spell_check(self, text, alpha, scale=True):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        # tokenize text
        nt = normalize_text(text)  # lower casing, padding punctuation with white spaces
        split_text = nt.split()

        candidates_sentences_details = {}  # dictionary of the complete sentences candidates and their values
        # adding the unchanged sentence
        candidates_sentences_details[nt] = {'p_x_given_w': math.log(alpha)}

        for index, word in enumerate(split_text):
            word_correction_candidates = self.get_correction_candidates(
                w=word)  # {eror_type: [candidates]} here the inclusion of a candidate in the vocab is verified

            for error_type, candidates in word_correction_candidates.items():
                candidate_sentences = []
                p_x_given_w_list_pre_scale = []
                for candidate in candidates:
                    candidate_sentence = " ".join(split_text[:index] + [candidate] + split_text[index + 1:])
                    if candidate != word:
                        candidate_sentences.append(candidate_sentence)
                        p_x_given_w = self.noisy_channel_evaluate(error_type, candidate, word,
                                                                  smooth=True)
                        p_x_given_w_list_pre_scale.append(p_x_given_w)

                # Here, there are all the p(x|w) of the candidates (that are different from the original word).
                # Their sum should be (1-alpha)
                if scale:
                    p_x_given_w_list_post_scale = self.normalize_sum(p_x_given_w_list_pre_scale,
                                                                     target_sum=1 - alpha)
                else:
                    p_x_given_w_list_post_scale = p_x_given_w_list_pre_scale

                for i in range(len(candidate_sentences)):
                    candidates_sentences_details[candidate_sentences[i]] = {
                        'p_x_given_w': math.log(p_x_given_w_list_post_scale[i])}

        for candidate_sentence in candidates_sentences_details.keys():
            # calculating the prior or conditional probability
            if len(split_text) < self.n:
                # Use unigram-based lm along with the noisy channel model
                priors = [self.get_unigram_prob(w) for w in candidate_sentence.split()]
                prior_or_conditional = sum(priors)
            else:
                # Use n_gram-based lm along with the noisy channel model
                prior_or_conditional = self.lm.evaluate(candidate_sentence)
            candidates_sentences_details[candidate_sentence]['p_w'] = prior_or_conditional

            # calculating the noisy channel + lm probability
            candidates_sentences_details[candidate_sentence]["noisy_lm_p"] = \
                candidates_sentences_details[candidate_sentence]['p_x_given_w'] + \
                candidates_sentences_details[candidate_sentence]['p_w']
            # select top candidate
        top_candidate = max(candidates_sentences_details,
                            key=lambda key: candidates_sentences_details[key]["noisy_lm_p"])
        return top_candidate

    def get_correction_candidates(self, w):
        """
        Calculates all the possible correction candidates within edit
        distance of 1 for a given word

        Args:
            w(str): a word to calculate correction candidates for

        Returns:
            (dict) dictionary containing error types and the possible candidates
            that are the result of each error. {eror_type: [candidates]}
        """
        correction_candidates = {}
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        correction_candidates["insertion"] = [candidate for candidate in self.get_insertion_candidates(splits) if
                                              self.word_in_vocab(candidate)]
        correction_candidates["deletion"] = [candidate for candidate in self.get_deletion_candidates(splits) if
                                             self.word_in_vocab(candidate)]
        correction_candidates["substitution"] = [candidate for candidate in self.get_substitution_candidates(splits) if
                                                 self.word_in_vocab(candidate)]
        correction_candidates["transposition"] = [candidate for candidate in self.get_transposition_candidates(splits)
                                                  if self.word_in_vocab(candidate)]
        return correction_candidates

    def word_in_vocab(self, w):
        """
        Checks if a given word is in the vocabulary of the language model.

        Args:
            w(str): a single word

        Returns:
            (bool) True if the given word is in the vocabulary of the language model
        """
        if w in self.vocab:
            return True
        return False

    def get_insertion_candidates(self, splits):
        """
        generates candidate words that could be the source of the given word and an insertion error.
        i.e. x typed as xy. A character was wrongfully added.
        Therefore, a deletion of each character at a time generates candidates for the originally intent word.

        Args:
            splits(tuple): a list of all the possible 2-parts splits of a word

        Returns:
            (tuple) a set of candidate words that could be the source of the given word and an insertion error
        """
        candidates = [L + R[1:] for L, R in splits if R]
        return set(candidates)

    def get_deletion_candidates(self, splits):
        """
        generates candidate words that could be the source of the given word and a deletion error.
        i.e. xy typed as x. A character was wrongfully omitted.
        Therefore, an insertion of characters in every position of the word generates candidates for the originally intent word.

        Args:
            splits(tuple): a list of all the possible 2-parts splits of a word

        Returns:
            (tuple) a set of candidate words that could be the source of the given word and a deletion error.
        """
        candidates = [L + c + R for L, R in splits for c in self.letters]
        return set(candidates)

    def get_substitution_candidates(self, splits):
        """
        generates candidate words that could be the source of the given word and a substitution/replacement error.
        i.e. x typed as y. A character was wrongfully substituted with another.
        Therefore, replacement of characters in every position of the word generates candidates for the originally intent word.

        Args:
            splits(tuple): a list of all the possible 2-parts splits of a word

        Returns:
            (tuple) a set of candidate words that could be the source of the given word and a substitution/replacement error.
        """
        candidates = [L + c + R[1:] for L, R in splits if R for c in self.letters]
        return set(candidates)

    def get_transposition_candidates(self, splits):
        """
        generates candidate words that could be the source of the given word and a transposition error.
        i.e. xy typed as yx. 2 character was wrongfully substituted with another.
        Therefore, switching every couple of sequential characters in every position of the word generates candidates for the originally intent word.

        Args:
            splits(tuple): a list of all the possible 2-parts splits of a word

        Returns:
            (tuple) a set of candidate words that could be the source of the given word and a transposition error.
        """
        candidates = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        return set(candidates)

    def get_error_count(self, error_type, error):
        """
        Return the amount of times a given error apperas in the error matrix

        Args:
            error_type(str): The type of the error
            error(str): the error

        Returns:
            (int) The amount of times a given error apperas in the error matrix
        """
        if error in self.error_tables[error_type]:
            return self.error_tables[error_type][error]
        else:
            return 0

    def noisy_channel_evaluate(self, error_type, candidate, original_word, smooth):
        """
        Evaluates the noisy channel probability based on the given error type.

        Args:
            error_type(str): The type of the error
            candidate(str): a candidate word
            original_word(str): a word with possible error
            smooth(bool): if True use Laplace smoothing

        Returns:
            (float) The noisy channel probability based on the given error type.
        """
        if error_type == "insertion":
            p_x_given_w = self.noisy_channel_eval_insert_err(candidate, original_word, smooth=smooth)
        elif error_type == "deletion":
            p_x_given_w = self.noisy_channel_eval_del_err(candidate, original_word, smooth=smooth)
        elif error_type == "substitution":
            p_x_given_w = self.noisy_channel_eval_sub_err(candidate, original_word, smooth=smooth)
        elif error_type == "transposition":
            p_x_given_w = self.noisy_channel_eval_trans_err(candidate, original_word, smooth=smooth)
        return p_x_given_w

    def noisy_channel_eval_insert_err(self, candidate, original_word, smooth=True):
        """
        Evaluates the noisy channel probability for insertion.

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error
            smooth(bool): if True use Laplace smoothing

        Returns:
            (float) the noisy channel probability
        """
        # discover what error was made
        error = self.get_insert_err(candidate, original_word)
        # find the appropriate error in the insertion matrix
        numerator = self.get_error_count('insertion', error)
        # find the denominator
        denominator = self.count_unigram(error[0])
        if smooth:
            numerator = numerator + 1
            denominator = denominator + len(self.char_unigrams)
        return numerator / denominator

    def noisy_channel_eval_del_err(self, candidate, original_word, smooth=True):
        """
        Evaluates the noisy channel probability for deletion.

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error
            smooth(bool): if True use Laplace smoothing

        Returns:
            (float) the noisy channel probability
        """
        # discover what error was made
        errors = self.get_del_err(candidate, original_word)
        # find the appropriate error in the deletion matrix
        numerator = 0
        for error in errors:
            numerator = numerator + self.get_error_count('deletion', error)
        numerator = numerator / len(errors)  # avg possible corrections
        # find the denominator
        denominator = self.count_bigram(error)
        if smooth:
            numerator = numerator + 1
            denominator = denominator + len(self.char_bigrams)
        return numerator / denominator

    def noisy_channel_eval_sub_err(self, candidate, original_word, smooth=True):
        """
        Evaluates the noisy channel probability for substition.

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error
            smooth(bool): if True use Laplace smoothing

        Returns:
            (float) the noisy channel probability
        """
        # discover what error was made
        error = self.get_sub_err(candidate, original_word)
        # find the appropriate error in the substitution matrix
        numerator = self.get_error_count('substitution', error)
        # find the denominator
        denominator = self.count_unigram(error[1])
        if smooth:
            numerator = numerator + 1
            denominator = denominator + len(self.char_unigrams)
        return numerator / denominator

    def noisy_channel_eval_trans_err(self, candidate, original_word, smooth=True):
        """
        Evaluates the noisy channel probability for transposition.

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error
            smooth(bool): if True use Laplace smoothing

        Returns:
            (float) the noisy channel probability
        """
        # discover what error was made
        error = self.get_trans_err(candidate, original_word)
        # find the appropriate error in the transposition matrix
        numerator = self.get_error_count('transposition', error)
        # find the denominator
        denominator = self.count_bigram(error)
        if smooth:
            numerator = numerator + 1
            denominator = denominator + len(self.char_bigrams)
        return numerator / denominator

    def get_del_err(self, candidate, original_word):
        """
        Returns the deletion error
        xy typed as x. A character was wrongfully added.

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error

        Returns:
            (str) the deletion error
        """
        # original_word<candidate
        corrections = []
        if candidate[-1] != original_word[-1]:  # last char is a mistake
            corrections.append(candidate[-1] + original_word[-1])
        if candidate[-1] == candidate[-2] and len(
                original_word) > 1:  # candidate ending with 2 identical letters. exp. too
            if candidate[-2] != original_word[-2]:
                corrections.append(candidate[-2] + candidate[-1])
        if candidate[0] == candidate[1]:  # candidate beginning with 2 identical letters. exp. aab
            if len(original_word) == 1 or candidate[1] != original_word[1]:
                corrections.append('#' + candidate[0])
        for i in range(len(original_word)):  # original_word<candidate
            if original_word[i] != candidate[i]:
                if i == 0:  # if insertion occurs at first position: x given "" before
                    corrections.append('#' + candidate[i])
                    break
                elif i != 0:  # if letter not inserted at position 0
                    corrections.append(original_word[i - 1] + candidate[i])
                    break

        return corrections

    def get_insert_err(self, candidate, original_word):
        """
        Returns the insertion error
        x typed as xy. A character was wrongfully added.

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error

        Returns:
            (str) the insertion error
        """
        # original_word>candidate
        if original_word[0] != candidate[0]:  # first char is a mistake
            return '#' + original_word[0:1]
        else:
            for index in range(len(candidate)):  # middle char is a mistake
                if original_word[index] != candidate[index]:
                    return original_word[index - 1:index + 1]
            return original_word[-2:]  # last char is a mistake

    def get_sub_err(self, candidate, original_word):
        """
        Returns the substituion error

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error

        Returns:
            (str) the substituion error
        """
        for index in range(len(candidate)):
            if candidate[index] != original_word[index]:
                return candidate[index] + original_word[index]

    def get_trans_err(self, candidate, original_word):
        """
        Returns the transposition error

        Args:
            candidate(str): a candidate word
            original_word(str): a word with possible error

        Returns:
            (str) the transposition error
        """
        for index in range(len(candidate)):
            if candidate[index] != original_word[index]:
                return candidate[index:index + 2]

    def normalize_sum(self, vector, target_sum):
        """
        Normalizes the sum of a vector to the given target sum.
        intuition: [a,b], S:=sum, X:=target sum =>
        a+b=S => *X => X(a+b)=X*S => /S => X/S*(a+b)=X => a'=a*X/S, b'=b*X/S

        Args:
            vector(tuple): a vector of non-negative numbers
            target_sum(numeric): a target sum value

        Returns:
            vector(tuple): the normalized-sum vector
        """
        old_sum = sum(vector)
        return [x * target_sum / old_sum for x in vector.copy()]


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    norm_text = text.lower()
    norm_text = re.sub('([.,!?()\n])', r' \1 ', norm_text)
    norm_text = re.sub('\s{2,}', ' ', norm_text)
    return norm_text


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a language model
        from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        self.n = n
        self.model_dict = defaultdict(int)
        self.n_gram_dicts = {}
        self.n_gram_dicts_total_count = {}
        # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
        self.chars = chars

    def build_model(self, text):
        """populates the instance variable model_dict.

            Args:
                text (str): the text to construct the model from.
        """
        if not self.chars:
            split_text = text.split()
            for n in range(1, self.n + 1):
                n_gram_dict, n_grams_total_count = self.generate_n_gram_dict(n, split_text)
                self.n_gram_dicts[f"{n}"] = n_gram_dict
                self.n_gram_dicts_total_count[f"{n}"] = n_grams_total_count
                if n == self.n:
                    self.model_dict = n_gram_dict
        else:
            for n in range(1, self.n + 1):
                n_gram_dict, n_grams_total_count = self.generate_n_gram_dict_char(n, text.lower())
                self.n_gram_dicts[f"{n}"] = n_gram_dict
                self.n_gram_dicts_total_count[f"{n}"] = n_grams_total_count
                if n == self.n:
                    self.model_dict = n_gram_dict

    def generate_n_gram_dict(self, n, split_text):
        """
        generates n_gram dictionary for words using the given n, based on the given normalized split text

            Args:
                n (int): he length of the markov unit
                split_text (str): normalized split text to construct the dict from.
            Return:
                dict. The generated dictionary.
                int. n-grams total count
        """
        n_gram_dict = defaultdict(int)
        n_grams_total_count = 0
        for start_index in range(len(split_text)):
            end_index = start_index + n
            if len(split_text) < end_index:  # if there are no more n-grams
                break
            else:
                n_gram = ' '.join(split_text[start_index:end_index])
                n_gram_dict[n_gram] = n_gram_dict[n_gram] + 1
            n_grams_total_count = n_grams_total_count + 1
        return n_gram_dict, n_grams_total_count

    def generate_n_gram_dict_char(self, n, text_lower):
        """
        generates n_gram dictionary for chars using the given n, based on the given normalized split text

            Args:
                n (int): he length of the markov unit
                text_lower (str): normalized (lowered) text to construct the dict from.
            Return:
                dict. The generated dictionary.
                int. n-grams total count
        """
        n_gram_dict = defaultdict(int)
        n_grams_total_count = 0
        for start_index in range(len(text_lower)):
            end_index = start_index + n
            if len(text_lower) < end_index:  # if there are no more n-grams
                break
            else:
                n_gram = text_lower[start_index:end_index]
                n_gram_dict[n_gram] = n_gram_dict[n_gram] + 1
            n_grams_total_count = n_grams_total_count + 1
        return n_gram_dict, n_grams_total_count

    def get_model_dictionary(self):
        """Returns the dictionary class object        """
        return self.model_dict

    def get_model(self):
        """Returns the dictionary class object        """
        return self.model_dict

    def get_model_window_size(self):
        """Returning the size of the context window (the n in "n-gram")
        """
        return self.n

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted. If the length of the specified context exceeds (or equal to)
        the specified n, the method should return the a prefix of length n of the specified context.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.

        """
        if not self.chars:
            if context is None:
                # # get the word with the highest frequency
                # context = max(self.n_gram_dicts["1"], key=self.n_gram_dicts["1"].get)
                # sampling initial context
                # uni_gram_probs_dict = {k: v/self.n_gram_dicts_total_count["1"] for k, v in self.n_gram_dicts["1"].items()}
                context = random.choices(list(self.n_gram_dicts["1"].keys()), weights=self.n_gram_dicts["1"].values(),
                                         k=1)[0]

            norm_context = normalize_text(context)
            split_context = norm_context.split()

            generated_text = split_context
            while len(generated_text) < n:  # while we can generate
                # advancing the context pointer
                current_context = generated_text[-self.n + 1:]
                # Finding candidates
                current_context_str = ' '.join(current_context)
                next_word_dict = self.n_gram_dicts[f"{len(current_context) + 1}"]
                candidates = {k: v for k, v in next_word_dict.items() if
                              current_context_str in k[:len(current_context_str)]}

                if len(candidates) == 0:
                    break
                else:
                    # # choose most likely following word token
                    # most_likely_next_word = max(candidates, key=candidates.get).split()[-1]
                    generated_text.append(
                        random.choices(list(candidates.keys()), weights=candidates.values(), k=1)[0].split()[-1])
            result = ' '.join(generated_text[:n])
        else:
            if context is None:
                # # get the word with the highest frequency
                # context = max(self.n_gram_dicts["1"], key=self.n_gram_dicts["1"].get)
                # sampling initial context
                # uni_gram_probs_dict = {k: v/self.n_gram_dicts_total_count["1"] for k, v in self.n_gram_dicts["1"].items()}
                context = random.choices(list(self.n_gram_dicts["1"].keys()), weights=self.n_gram_dicts["1"].values(),
                                         k=1)[0]

            lower_context = context.lower()

            generated_text = lower_context
            while len(generated_text) < n:  # while we can generate
                # advancing the context pointer
                current_context = generated_text[-self.n + 1:]
                # Finding candidates
                next_word_dict = self.n_gram_dicts[f"{len(current_context) + 1}"]
                candidates = {k: v for k, v in next_word_dict.items() if
                              current_context in k[:len(current_context)]}

                if len(candidates) == 0:
                    break
                else:
                    # # choose most likely following word token
                    # most_likely_next_word = max(candidates, key=candidates.get).split()[-1]
                    generated_text = generated_text + \
                                     random.choices(list(candidates.keys()), weights=candidates.values(), k=1)[
                                         0]
            result = generated_text
        return result

    def evaluate(self, text):
        """Returns the log-likelihood of the specified text to be a product of the model.
           Laplace smoothing should be applied if necessary.

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        if not self.chars:

            split_text = normalize_text(text).split()

            n_grams = []
            start_index = 0
            is_smoothing_required = False
            for end_index in range(1, len(split_text) + 1):
                if end_index > len(split_text):
                    break
                if end_index - start_index > self.n:
                    start_index = start_index + 1
                n_gram_length = end_index - start_index
                n_gram = ' '.join(split_text[start_index:end_index])
                n_grams.append(n_gram)
                if n_gram not in self.n_gram_dicts[f"{n_gram_length}"]:
                    is_smoothing_required = True

            if is_smoothing_required:
                probs = [self.smooth(n_gram) for n_gram in n_grams]
            else:
                probs = [self.get_n_gram_probability(n_gram) for n_gram in n_grams]

            log_probs = [math.log(prob) for prob in probs]

            log_likelihood = sum(log_probs)
            # math.exp(log_likelihood)  # ?
        else:
            text_lower = text.lower()

            n_grams = []
            start_index = 0
            is_smoothing_required = False
            for end_index in range(1, len(text_lower) + 1):
                if end_index > len(text_lower):
                    break
                if end_index - start_index > self.n:
                    start_index = start_index + 1
                n_gram_length = end_index - start_index
                n_gram = text_lower[start_index:end_index]
                n_grams.append(n_gram)
                if n_gram not in self.n_gram_dicts[f"{n_gram_length}"]:
                    is_smoothing_required = True

            if is_smoothing_required:
                probs = [self.smooth(n_gram) for n_gram in n_grams]
            else:
                probs = [self.get_n_gram_probability(n_gram) for n_gram in n_grams]

            log_probs = [math.log(prob) for prob in probs]

            log_likelihood = sum(log_probs)
            # math.exp(log_likelihood)  # ?
        return log_likelihood

    def get_n_gram_probability(self, n_gram):
        """
        receives a normalized n gram and calculates its probability

            Args:
                n_gram (str): the ngram to have it's probability smoothed

            Returns:
                float. The ngram probability.
        """
        if not self.chars:
            split_n_gram = n_gram.split()
            count = self.n_gram_dicts[f"{len(split_n_gram)}"][n_gram]
            if len(split_n_gram) == 1:  # unigram
                total = self.n_gram_dicts_total_count[
                    f"{self.n}"]  # self.n_gram_dicts_total_count[f"{len(split_n_gram)}"]
            else:
                previous = split_n_gram[0:-1]
                total = self.n_gram_dicts[f"{len(previous)}"][' '.join(split_n_gram[0:-1])]
        else:
            count = self.n_gram_dicts[f"{len(n_gram)}"][n_gram]
            if len(n_gram) == 1:  # unigram
                total = self.n_gram_dicts_total_count[
                    f"{self.n}"]  # self.n_gram_dicts_total_count[f"{len(split_n_gram)}"]
            else:
                previous = n_gram[0:-1]
                total = self.n_gram_dicts[f"{len(previous)}"][n_gram[0:-1]]
        return count / total

    def smooth(self, ngram):
        """Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """
        if not self.chars:

            norm_ngram = normalize_text(ngram)
            split_ngram = norm_ngram.split()

            if norm_ngram in self.n_gram_dicts[f"{len(split_ngram)}"]:
                count = self.n_gram_dicts[f"{len(split_ngram)}"][norm_ngram]
            else:
                count = 0

            # Laplace
            count = count + 1

            if len(split_ngram) == 1:  # unigram
                total = self.n_gram_dicts_total_count[f"{self.n}"] + \
                        len(self.n_gram_dicts["1"])  # len(self.n_gram_dicts[f"{self.n}"])
            else:
                previous = split_ngram[0:-1]
                total = self.n_gram_dicts[f"{len(previous)}"][' '.join(split_ngram[0:-1])] + \
                        len(self.n_gram_dicts["1"])  # len(self.n_gram_dicts[f"{self.n}"])
        else:
            if ngram in self.n_gram_dicts[f"{len(ngram)}"]:
                count = self.n_gram_dicts[f"{len(ngram)}"][ngram]
            else:
                count = 0

            # Laplace
            count = count + 1

            if len(ngram) == 1:  # unigram
                total = self.n_gram_dicts_total_count[f"{self.n}"] + \
                        len(self.n_gram_dicts["1"])  # len(self.n_gram_dicts[f"{self.n}"])
            else:
                previous = ngram[0:-1]
                total = self.n_gram_dicts[f"{len(previous)}"][' '.join(ngram[0:-1])] + \
                        len(self.n_gram_dicts["1"])  # len(self.n_gram_dicts[f"{self.n}"])
        return count / total


def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Yiftach Savransky', 'id': '312141369', 'email': 'yiftachs@post.bgu.ac.il'}
