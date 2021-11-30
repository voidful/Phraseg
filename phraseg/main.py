from collections import defaultdict

from nlp2 import is_file_exist, read_files_into_lines, split_lines_by_punc, split_sentence_to_ngram_in_part, \
    split_sentence_to_array
from tqdm import tqdm


class Phraseg:

    def __init__(self, source, line_max_length=50, idf_chunk=500):
        if is_file_exist(source):
            content = read_files_into_lines(source)
        else:
            content = source.splitlines()
        self.idf_chunk = idf_chunk
        self.sentences = split_lines_by_punc(content, max_len=line_max_length)
        self.ngrams, self.idf = self._cal_ngrams_idf(self.sentences)

    def _cal_ngrams_idf(self, sentences):
        ngrams = defaultdict(int)
        idf = defaultdict(int)
        _added = defaultdict(bool)
        chunks = self.idf_chunk
        for pos, sentence in tqdm(enumerate(sentences), total=len(self.sentences)):
            if chunks != 0 and (pos + 1) % chunks == 0:
                _added = defaultdict(bool)
            part = split_sentence_to_ngram_in_part(sentence)
            for ngram in part:
                ngrams["." + ngram[0]] += 1
                for i in ngram:
                    ngrams[i] += 1
                    if not _added[i]:
                        idf[i] += 1
                        _added[i] = True
                ngrams[ngram[-1] + "."] += 1
        return ngrams, idf

    def _filter_condprob(self, sentence_array, ngrams):
        internal_feature = defaultdict(int)
        if len(sentence_array) == 1:
            return {}

        p1 = ngrams["." + sentence_array[0]]
        p2 = ngrams[sentence_array[0]]
        p = p2 / p1
        internal_feature[sentence_array[0]] = p

        last_word = sentence_array[len(sentence_array) - 1]
        for i in range(1, len(sentence_array) - 1):
            p1 = ngrams[sentence_array[i]]
            p2 = ngrams[sentence_array[i + 1]]
            p = p2 / p1 if p1 is not 0 else 0
            internal_feature[sentence_array[i]] = p

        p1 = ngrams[last_word]
        p2 = ngrams[last_word + "."]
        p = p2 / p1 if p1 is not 0 else 0
        internal_feature[sentence_array[len(sentence_array) - 1]] = p

        result = {key: value for key, value in internal_feature.items() if 0 < value < 1 < len(key)}
        return result

    def _filter_second_frequently(self, result_dict):
        words = list(result_dict.keys())
        ngrams, _ = self._cal_ngrams_idf(words)
        for word in words:
            keep = False
            for i in split_sentence_to_array(word):
                if ngrams[i] < self.ngrams[word]:
                    keep = True
            if not keep:
                result_dict.pop(word)
        return result_dict

    def _reverse_non_unique_mapping(self, d):
        dinv = defaultdict(list)
        for k, v in d.items():
            if str(v) in dinv:
                dinv[str(v)].append(k)
            else:
                dinv[str(v)] = [k]
        return dinv

    def _all_words_match_maximum_array(self, list_arr):
        for word in list_arr:
            for i in range(1, len(word)):
                for j in range(len(word) - i + 1):
                    word_part = word[j:i + j]
                    if word_part in list_arr:
                        list_arr.remove(word[j:i + j])
        return list_arr

    def maximum_match_same_value(self, a):
        result = []
        res = self._reverse_non_unique_mapping(a)
        for key, value in res.items():
            if len(value) > 1:
                for i in self._all_words_match_maximum_array(value):
                    result.append(i)
            else:
                result.append(value[0])
        return result

    def _get_all_overlap(self, words_list, sentence):
        overlap_list = []
        for word in words_list:
            # if len(word) < 2:
            #     continue
            waitinglist = [word]
            word_index = sentence.find(word)
            if word_index >= 0:
                for length in range(1, len(word)):
                    left = sentence[word_index - length:word_index + len(word) - length]
                    right = sentence[word_index + length:word_index + len(word) + length]
                    if len(word) - len(left) < 2 and len(left) > 1:
                        waitinglist.append(left)
                    if len(word) - len(right) < 2 and len(right) > 1:
                        waitinglist.append(right)
            if waitinglist not in overlap_list:
                overlap_list.append(waitinglist)
        return overlap_list

    def _get_all_superlap(self, words_list, sentence):
        superlap_list = []
        for word in words_list:
            if len(word) < 2:
                continue
            waitinglist = [word]
            word_index = sentence.find(word)
            if word_index >= 0:
                for right_index in range(1, len(sentence) - word_index):
                    right = sentence[word_index:word_index + len(word) + right_index]
                    if len(right) > 1 and " " not in right:
                        waitinglist.append(right)
                for left_index in range(1, word_index):
                    left = sentence[word_index - left_index:word_index + len(word)]
                    if len(left) > 1 and " " not in left:
                        waitinglist.append(left)
                if waitinglist not in superlap_list:
                    superlap_list.append(waitinglist)
        return superlap_list

    def _get_superlap(self, words_list, sentence, ngrams):
        superlap_list = defaultdict(list)
        for word in words_list:
            if len(word) < 2:
                continue
            superlap_list[word] = {'left': [], "right": []}
            word_index = sentence.find(word)
            if word_index >= 0:
                for right_index in range(1, len(sentence) - word_index):
                    right = sentence[word_index:word_index + len(word) + right_index]
                    if len(right) > 1 and " " not in right and word not in superlap_list[word][
                        "right"] and right != word and right in ngrams and ngrams[right] > 0:
                        superlap_list[word]["right"].append(right)
                for left_index in range(1, word_index):
                    left = sentence[word_index - left_index:word_index + len(word)]
                    if len(left) > 1 and " " not in left and word not in superlap_list[word][
                        "left"] and left != word and left in ngrams and ngrams[left] > 0:
                        superlap_list[word]["left"].append(left)
        return superlap_list

    def _remove_by_overlap(self, array, sentence, ngrmas):
        result = []
        gaol = self._get_all_overlap(array, sentence)
        for i in gaol:
            max = 0
            word = ''
            if len(i) > 1:
                for j in i:
                    if ngrmas[j] > max:
                        max = ngrmas[j]
                        word = j
            else:
                word = i[0]
            if len(word) > 0:
                result.append(word)
        return result

    def _remove_by_superlap(self, array, resarray, sentence):
        result = []
        big_list = self._get_all_superlap(array, sentence)
        for word_list in big_list:
            max = 0
            word = ''
            if len(word_list) > 1:
                for superword in word_list:
                    if resarray[superword] > max:
                        max = resarray[superword]
                        word = superword
            else:
                word = word_list[0]
            if len(word) > 0:
                result.append(word)
        return result

    def _all_words_match_maximum_array(self, list_arr):
        for word in list_arr:
            for i in range(1, len(word)):
                for j in range(len(word) - i + 1):
                    wordPart = word[j:i + j]
                    if wordPart in list_arr:
                        # waitinglist.append(wordPart)
                        list_arr.remove(word[j:i + j])
        return list_arr

    def extract(self, sent=None, merge_overlap=True, result_word_minlen=1, line_max_length=50):
        result_dict = defaultdict(int)
        sents = split_lines_by_punc([sent], max_len=line_max_length) if sent is not None else self.sentences
        iter = sents if sent is not None else tqdm(sents, total=len(self.sentences))
        for sentence in iter:
            filter_dict = defaultdict(int)
            filter_arr = []
            ngram_part = split_sentence_to_ngram_in_part(sentence)
            if len(ngram_part) > 0:
                for part in ngram_part:
                    filter_result = self._filter_condprob(part, self.ngrams)
                    for key, value in filter_result.items():
                        filter_dict[key] = self.ngrams[key]
                        filter_arr.append(key)
            if len(filter_dict) > 0:
                if merge_overlap:
                    filter_arr = self.maximum_match_same_value(filter_dict)
                    rm_sup = self._remove_by_superlap(filter_arr, filter_dict, sentence)
                    filter_arr = self._remove_by_overlap(rm_sup, sentence, self.ngrams)
                    gaol = self._all_words_match_maximum_array(filter_arr)
                    for i in gaol:
                        result_dict[i] = self.ngrams[i] / self.idf[i]
                else:
                    for key in filter_arr:
                        result_dict[key] = self.ngrams[key] / self.idf[key]
        if merge_overlap:
            result_dict = self._filter_second_frequently(result_dict)
        result_dict = {k: v for k, v in result_dict.items() if len(split_sentence_to_array(k)) > result_word_minlen}
        return result_dict
