from MySpamAgent import MySpamAgent

non_spam = [
    'dear Philipp how are you today',
    'hello Philipp today I wanted to write you',
    'greetings are you coming into work tomorrow',
    'dear Philipp I am still at work',
    'hey Philipp do you want to go to the theatre',
    'hello what shall we eat for dinner'
]

spam = [
    'good day friend I have an offer for you congratulations',
    'congratulations you won the lottery',
    'you will not believe what I can offer you',
    'congratulations you are our lucky customer',
    'dear customer we have an exclusive offer for you',
    'good day we have a shipment of gold for you congratulations',
    'congratulations your package has been delivered',
    'congratulations you won an exclusive holiday package'
]

mySpamAgent = MySpamAgent(non_spam, spam)

X_test = 'congratulations sir I have an exclusive offer for you'

print(mySpamAgent.predict(X_test))
