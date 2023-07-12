import re
from nltk.corpus import wordnet


class AbbreviationReduction():
    def __init__(self):
        replacement_patterns = [
            (r'won\'t', 'will not'),
            (r'can\'t', 'can not'),
            (r'i\'m', 'i am'),
            (r'ain\'t', 'is not'),
            (r'(\w+)\'ll', '\g<1> will'),
            (r'(\w+)n\'t', '\g<1> not'),
            (r'(\w+)\'ve', '\g<1> have'),
            (r'(\w+)\'s', '\g<1> is'),
            (r'(\w+)\'re', '\g<1> are'),
            (r'(\w+)\'d', '\g<1> would')]

        self.patterns = [(re.compile(regex, re.I), repl)
                         for (regex, repl) in replacement_patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s


class RepeatReplacer():
    def __init__(self):
        self.repeat_reg = re.compile(r'(\w*)(\w)\2(\w*)', re.I)
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_reg.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


class UrlProcess():
    def __init__(self):
        self.hashtag_pattern = re.compile('.?#[a-zA-Z0-9_\.]+', re.I)
        self.at_pattern = re.compile('.?@[a-zA-Z0-9_\.]+', re.I)
        self.http_pattern = re.compile("(http|ftp|https)://[a-zA-Z0-9\./]+|www\.(\w+\.)+\S*/", re.I)

    def replace(self, text):
        text = re.sub(self.hashtag_pattern, '', text)
        text = re.sub(self.at_pattern, '', text)
        text = re.sub(self.http_pattern, '', text)
        return text


class TextProcess_0():
    def __init__(self):
        pass

    def process(self, text):
        return text


class TextProcess_1():
    def __init__(self):
        self.abbreviation_reduction = AbbreviationReduction()
        self.repeat_replacer = RepeatReplacer()

    def process(self, text):
        text = self.abbreviation_reduction.replace(text)
        text = self.repeat_replacer.replace(text)
        return text


class TextProcess_2():
    def __init__(self):
        self.url_replacer = UrlProcess()

    def process(self, text):
        text = self.url_replacer.replace(text)
        return text


class TextProcess_3():
    def __init__(self):
        self.abbreviation_reduction = AbbreviationReduction()
        self.repeat_replacer = RepeatReplacer()
        self.url_replacer = UrlProcess()

    def process(self, text):
        text = self.abbreviation_reduction.replace(text)
        text = self.repeat_replacer.replace(text)
        text = self.url_replacer.replace(text)
        return text
