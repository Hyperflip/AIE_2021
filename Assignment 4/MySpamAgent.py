class MySpamAgent:
    def __init__(self, non_spam, spam):
        # init class probabilities
        self.non_spam_prob = len(non_spam) / (len(non_spam) + len(spam))
        self.spam_prob = len(spam) / (len(non_spam) + len(spam))

        # init total word counts
        self.non_spam_total_count = 0
        self.spam_total_count = 0

        # init counts (start at 1) and calc probabilities for each word
        self.non_spam_word_count = {}
        self.spam_word_count = {}
        for text in non_spam:
            for word in text.split():
                self.non_spam_total_count += 1

                if word not in self.non_spam_word_count:
                    self.non_spam_word_count[word] = 1
                self.non_spam_word_count[word] += 1

        for text in spam:
            for word in text.split():
                self.spam_total_count += 1

                if word not in self.spam_word_count:
                    self.spam_word_count[word] = 1
                self.spam_word_count[word] += 1

        self.non_spam_word_prob = {}
        self.spam_word_prob = {}
        for word in self.non_spam_word_count:
            self.non_spam_word_prob[word] = int(self.non_spam_word_count[word]) / self.non_spam_total_count

        for word in self.spam_word_count:
            self.spam_word_prob[word] = int(self.spam_word_count[word]) / self.spam_total_count

    def predict(self, text):
        non_spam_prob = self.non_spam_prob
        spam_prob = self.spam_prob

        for word in text.split():
            if word not in self.non_spam_word_prob or word not in self.spam_word_prob:
                continue
            non_spam_prob *= self.non_spam_word_prob[word]
            spam_prob *= self.spam_word_prob[word]

        return 'spam' if spam_prob > non_spam_prob else 'non_spam'
