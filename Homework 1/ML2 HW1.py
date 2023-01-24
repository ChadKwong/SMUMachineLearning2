import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import * 


### Question 2 ###

#Creating Flower Orders List
flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
               'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
               'W/R/B','W/R/B','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R',
               'R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V','W/R/V',
               'W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V',
               'W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y',
               'R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/G','W/G',
               'W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V','W/R/B/V','W/R/B/V','W/R/B/V',
               'W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y','W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y',
               'W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y','W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y',
               'W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R',
               'W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y',
               'R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M',
               'W/B/V/O','N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']	

#Question 2.1
#Creating Counter function
def myCounter(data,target='none'): 
    # Defining dictionary to store output
    count = {}

    #Creating a conditional statement for counting the items in a list
    if target=='items':
        for item in data:
            if item not in count:
                count[item] = 0
            count[item] += 1
    
    # Creating a conditional statement to count all letters if no target is specified
    if target=='none':
        #Looping through each item in the dataset
        for item in data:
            #Looping through each letter of the item
            for letter in item:
                #Creating a count of the letter if there were no other previous occurences and then adding to the total
                if letter not in count:
                    count[letter] = 0
                count[letter] += 1  
                
    # Counting target
    else:
        count[target] = 0
        for item in data:
            for letter in item:
                if letter == target :
                    count[target] += 1

    #Sorting final dictionary output using Sorted function and converting returned list of tuples into a dictionary
    sortedCount = sorted(count.items(), key=lambda x:x[1],reverse=True)
    sortedCount = dict(sortedCount)

    return sortedCount

# Question 2.1
print("The results of the built in Counter function are: \n",Counter('what a wonderful world it is indeed'),'\n')
print("The results of my counter Counter function are: \n",myCounter('what a wonderful world it is indeed'),'\n')

# Question 2.2
print('\n\nThe number of W in flower_orders is: ', myCounter(flower_orders,'W'),'\n\n')

#Question 2.3
hist = [y for x in flower_orders for y in x.split('/')]
plt.hist(hist)
plt.title('histogram of colors')
plt.show()

# Question 2.4
#Rank the pairs of colors in each order regardless of how many colors are in an order.
#Creating a list to store all of the combinations of colors within each item of flower orders
colorPairs = [''.join(pair) for colors in flower_orders for pair in combinations(colors.split('/'), 2) ]

#Using the counter function to obtain counts of each pair
pairCount = Counter(colorPairs)
print('\n\n',pairCount)

#Plotting a histogram of the pair occurrences
plt.bar(pairCount.keys(), pairCount.values())
plt.show()

# Question 2.5
#Rank the triplets of colors in each order regardless of how many colors are in an order
#Creating a list to store all of the combinations of colors within each item of flower orders
nsplits = 3
colorTrips = [''.join(pair) for colors in flower_orders for pair in combinations(colors.split('/'), nsplits) ]

#Using the counter function to obtain counts of each pair
tripCount = Counter(colorTrips)
print('\n\n',tripCount)

#Plotting a histogram of the triplet occurrences (Too messy to read)
# plt.bar(tripCount.keys(), tripCount.values())
# plt.show()

# Question 2.6
#Make a dictionary with key=”color” and values = “what other colors it is ordered with”.
#creating empty dictionary
colorPairs = {}

#looping through flower orders
for item in flower_orders:
    #looping through the colors within each order
    for color in item.split('/'):
        #creating an item containing an empty list as the value if the color is not present
        if color not in colorPairs:
            colorPairs[color] = []
        #looping through the dictionary and adding the color as a value if it is not equal to the key and doesnt already exist as a pair
        for index in colorPairs:
            if color != index:
                if color not in colorPairs[index]:
                    colorPairs[index].append(color)
            
print('\n\n',colorPairs)


# Question 2.7
#Make a graph showing the probability of having an edge between two colors based on how often they co-occur.  (a numpy square matrix)
edgeProb = {}

#Using the method for searching pairs that was implemented earlier
colorPairs = [''.join(pair) for colors in flower_orders for pair in combinations(colors.split('/'), 2) ]
counts = Counter(colorPairs)

#counting the total number of pairs to divide by
total = sum([count for pair, count in counts.items()])

#looping through each pair and calculating the probability
for pair, count in counts.items():
    edgeProb[pair] = count/total

print('\n\nThe edge probabilities are:\n\n',edgeProb)













### Question 3 ###

dead_men_tell_tales = ['Four score and seven years ago our fathers brought forth on this',
                        'continent a new nation, conceived in liberty and dedicated to the',
                        'proposition that all men are created equal. Now we are engaged in',
                        'a great civil war, testing whether that nation or any nation so',
                        'conceived and so dedicated can long endure. We are met on a great',
                        'battlefield of that war. We have come to dedicate a portion of',
                        'that field as a final resting-place for those who here gave their',
                        'lives that that nation might live. It is altogether fitting and',
                        'proper that we should do this. But in a larger sense, we cannot',
                        'dedicate, we cannot consecrate, we cannot hallow this ground.',
                        'The brave men, living and dead who struggled here have consecrated',
                        'it far above our poor power to add or detract. The world will',
                        'little note nor long remember what we say here, but it can never',
                        'forget what they did here. It is for us the living rather to be',
                        'dedicated here to the unfinished work which they who fought here',
                        'have thus far so nobly advanced. It is rather for us to be here',
                        'dedicated to the great task remaining before us--that from these',
                        'honored dead we take increased devotion to that cause for which',
                        'they gave the last full measure of devotion--that we here highly',
                        'resolve that these dead shall not have died in vain, that this',
                        'nation under God shall have a new birth of freedom, and that',
                        'government of the people, by the people, for the people shall',
                        'not perish from the earth.']


# 1. Join everything
joined = ' '.join(dead_men_tell_tales)
print('\n\n','The following is everything joined together: \n\n',joined)

# 2. Remove spaces
noSpaces = joined.replace(' ', '')
print('\n\n','The following is with no spaces: \n\n', noSpaces)

# 3. Occurrence probabilities for letters
# converting joined string into lowercase alphabetic characters only
allLetters = ''.join(letter for letter in noSpaces.casefold() if letter.isalpha())

# create a dictionary of the letters of the alphabet to store the probabilities
alphabet = list(map(chr, range(97, 123)))
letterProbabilities = {letter:0 for letter in alphabet}

# loop through alphabet dictionary and set the values to the probability of occurrence 
for letter in letterProbabilities:
    letterProbabilities[letter] = round(allLetters.count(letter)/len(allLetters),5)

print('\n\n','The occurrence probabilities are contained in the following dictionary:\n\n', letterProbabilities)

# 4. Tell me transition probabilities for every pair of letters
#Compiling pairs of letters and their counts
letterPairs = [''.join(pair) for pair in combinations_with_replacement(allLetters,2)]
letterCounts = Counter(letterPairs)
letterTotals = sum([y for x,y in letterCounts.items()])

# creating a readable dictionary of the transition probabilities between letters in the data
transitionProbs = {}
for pair, count in letterCounts.items():
    transitionProbs[pair] = count/letterTotals
print('\n\n','The following are the transition probabilities from letter to letter:\n\n', transitionProbs)


# 5. Make a 26x26 graph of 4.  in numpy
# creating an identical dictionary storing the coordinates of the probabilities in a 26 by 26 graph
transitionCoord = {}
for pair, count in letterCounts.items():
    transitionCoord[(ord(pair[0])-97,ord(pair[1])-97)] = count/letterTotals

# Mapping the coordinate dictionary to the empty matrix and filling in the values
    transitionMatrix = np.zeros((26,26))
for pair in transitionCoord:
    transitionMatrix.itemset(pair,transitionCoord[pair])
transitionMatrix = np.asmatrix(transitionMatrix)
print('\n\n','The following is a matrix representation of the transition probabilities:\n\n', transitionMatrix)

# 6. plot graph of transition probabilities from letter to letter'

plt.imshow(transitionMatrix)
plt.colorbar()
plt.show()