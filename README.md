
# Spam Classification
The Naive Bayes Model is simple since it assumed the independence of each feature. 

This model works as follows:
For this problem, we started with reading the training files and maintaining a dictionary of all the words in the dataset. Also, while forming the word dictionary, we have included only those words which satisfied the condition for isalpha() and removed those words whose length was less than 2 which would inlcude quite a few stopwords. For those words not present in either the spam_words dictionary or the notspam_words dictionary, we have set the count to be zero, which would be handled by the Laplacian smoothing while calculating the prior probalilities. The initial P(S):spam and P(NS):notspam probabilities are calculated from the total respective counts from the dataset.

For the spam and not spam training set, we created separate word dictionaries for the spam and not spam words while calculating their total count and the likelihood, i.e. P(W|Spam) and P(W|Notspam).

For the test set, we determined whether each file belonged to class spam or not spam by calculating the posterior probabilities using the Naive bayes formula. The threshold was kept at 0.5, i.e. if the probability was greater than or equal to 0.5, it was classified as Spam else it would be put under notspam. The accuracy achieved for the same is ~85%.

Done by: Aneri Shah, Dhruva Bhavsar, Hely Modi.
