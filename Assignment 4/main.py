from MySpamAgent import MySpamAgent

non_spam = [
    'Dear Philipp how are you today',
    'Hello Philipp today I wanted to write you',
    'Greetings are you coming into work tomorrow',
    'Dear Philipp I am still at work',
    'Hey Philipp do you want to go to the theatre',
    'Hello what shall we eat for dinner'
]

spam = [
    'Good day friend I have an offer for you',
    'Congratulations you won the lottery',
    'You will not believe what I can offer you',
    'Congratulations you are our lucky customer',
    'Dear customer we have an exclusive offer for you',
    'Good day we have a shipment of gold for you'
    'Your package has been delivered'
    'Congratulations you won an exclusive holiday package'
]

mySpamAgent = MySpamAgent(non_spam, spam)

X_test = 'Congratulations customer you won exclusive offer'

print(mySpamAgent.predict(X_test))
