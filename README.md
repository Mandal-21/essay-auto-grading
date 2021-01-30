# essay_auto_grading
Essay Auto Grading


![Project Image](project-image-url)

> This README file describe the overview of the Essay Auto Grading Project.

---

### Table of Contents


- [Description](#description)
- [Dataset](#dataset)
- [Dataset description](#dataset-description)
- [Proposed steps](#proposed-steps)
- [API References](#apireferences)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description

This project is to build a model to automatically grade short essays. The dataset used is from a
Kaggle Competition “The Hewlett Foundation: Short Answer Scoring” in hopes of discovering
new tools to support schools and teachers.

Most of the assessments exclude written responses in favour of multiple-choice questions,
which are less able to assess students’ critical reasoning and writing skills. Tests that require
“constructed responses” (i.e., written answers) are useful tools, but they typically are hand
scored, commanding considerable time and expense.

So, because of those costs, standardized examinations have increasingly been limited to using
“bubble tests” that deny us opportunities to challenge our students with more sophisticated
measures of ability. Recent developments in innovative software to evaluate student written
responses and other response types are promising.

There are ten essay data sets. Each of the data sets was generated from a single prompt. Selected
responses have an average length of 50 words per response. All responses were written by
students primarily in Grade 10. All responses were hand graded and were double-scored. The
variability is intended to test the limits of the developed scoring engine's capabilities.


[Back To The Top](#essay_auto_grading)

#### Dataset

There are ten essay data sets. Each of the data sets was generated from a single prompt. Selected responses
have an average length of 50 words per response. All responses were written by students primarily in Grade
10. All responses were hand graded and were double-scored. The variability is intended to test the limits of
the developed scoring engine's capabilities.

Kaggle provided a training data of approximately 17,000 essays with two scores graded by two different
people. The test data consists of approximately 5,000 essays upon two different benchmarks (Bag of Words,
Length)

[Back To The Top](#essay_auto_grading)

#### Dataset description	

### Training Dataset:
	There are 10 datasets. Each dataset represents an Essay set. All responses were written by
	students primarily in Grade 10. All responses were hand graded and were double-scored.
	
Training data is provided in tsv form (Tab Separated Value)
	
	I. File Name: train.tsv
		Size: 17207 X 5
		
	II. File Name: train_rel_2.tsv (Some of the records are removed because of
	transcription error)
	Size: 17043 X 5

Features:
	1. Id: A unique identifier for each individual student essay.
	2. EssaySet: 1-10, an id for each set of essays.
	3. Score1: The human rater's score for the answer. This is the final score for the answer
	and the score that you are trying to predict.
	4. Score2: A second human rater's score for the answer. This is provided as a measure
	of reliability, but had no bearing on the score the essay received.
	5. EssayText: The ascii text of a student's response.


Test Dataset:

	I. File Name: test1.tsv
	
	Size: 5224 X 3
	
Features:

		1. Id:
		
		2. EssaySet:
		
		3. EssayText:
		
		
	II. File Name: test1_soln.csv (Dataset having SCORE for the test data test1.csv)
	
	Size: 5224 X 4
	
Features:

	1. Id:
	
	2. EssaySet:
	
	3. EssayWeight:
	
	4. EssayScore:
	
	III. File Name: test2.tsv (Dataset without having SCORE)
	
					Size: 5100 X 3
Features:

	1. Id:
	
	2. EssaySet:
	
	3. EssayText:
		
Here is the description of the steps followed in this project.

## I. Load the Dataset

#### Training Dataset:

Both training data files ‘train.tsv’ and ‘train_rel_2.tsv’ are downloaded. Some of the records
are removed because of transcription error in ‘train_rel_2.tsv’. As the dataset size is not large
enough, these two training dataset are combined. To show the variability, ‘train.tsv’ is taken with
grader1 score and ‘train_rel_2.tsv’ is taken with grader2 score.

#### Test Dataset:
The test dataset can be pulled from the file ‘test.tsv’ which is having the features Id, EssaySet
and EssayText. To check the performance the model created, the score for the test dataset can be
taken from other test dataset ‘test1_soln.csv’.

## II. About the Dataset

The size of the training dataset now is 34250 x 4. The features are as follows:

1. Id: A unique identifier for each individual student essay.

2. EssaySet: 1-10, an id for each set of essays.

3. Score: The human rater's score for the answer. This is the final score for the answer and the score that needs to be predicted.

4. EssayText: The ascii text of a student's response.

The predictors (Essay Set, Essay Text) and target feature (Score) needed are separated
from both train and test dataset in order to ease the creation of tensors and feature lists

## III. Data Preparation

#### 3.1 Tokenization:

	Tokenize the essay text by setting the number of top-most frequent words as 500. Fit it on both train and test dataset.
	
#### 3.2 Vectorization of tokens:

This will vectorize the text corpus ie., Essay Text by converting each text into sequence of integers.

#### 3.3 Creation of tensors of data:

As we proposed to use LSTM, the input data should be of the form tensor. So, both train and test essay text data are converted to tensor form. (word index sequence vector of same length)
Also, the essay scores are converted to categorical form.

#### 3.4 Stop word removal and lemmatization:

Stop words are removed.
Lemmatization is done to find the root word.
Word count and sentence count are done for the essay text.
A feature set is created with (word count, sentence count, essay set).

#### 3.5 Arranging train and test dataset

Predictors (Tensor data of word sequence vector) and target (categorical score) are kept separately.
Also, the feature set created in the previous step is stacked together.

## IV. Model Building:

#### 4.1 Instantiate Keras Layer:

Define Input layer with the shape of 500 (max response length considered).

#### 4.2 Building an Embedding Layer:

	Embeddings are used to convert the encoded data i.e the matrix of indices into a form compatible with LSTM.

	Embedding Layer Parameters:

	input_dim = Size of the vocabulary (No. of top most frequent words considered)

	output_dim = Size of the vector space in which words will be embedded.

	(Embedding vector length is set as 32)

	input_length = Length of input sequences. (Max response length)

#### 4.3 Building LSTM Network:

LSTM: Long Short-Term Memory Layer:

Parameters:

units: Dimensionality of the inner cell in LSTM.

return_sequences: it is set as True to return the full sequence and not the last output.

#### 4.4 Merging of LSTM with Features:

Create a Feature Input Layer and merge with Output of LSTM.
Dropout is used with 20%

#### 4.5 Building Dense Layer:

Create a Dense Layer with L2 regularization for weight and bias.
With Exponential Linear Unit Activation Function

#### 4.6 Building Final Model:

Build the final model with inputs as input layer and features.

#### 4.7 Model Compilation and Fitting:

	‘Adam’ optimizer is used with learning rate 0.01.
	The metric used is ‘accuracy’
	The loss measure used is ‘categorical cross entropy loss’.
	No. of epochs=5, Batch Size = 32, Validation Split is 10%

#### 4.8 Model Prediction:

Built model is used for the prediction of test dataset.
The output is in categorical form which then can be translated into score form.

#### 4.9 Model Evaluation:

Accuracy of the test dataset is calculated and it is 62.73%

#### Measure for Performance Evaluation is Quadratic Weighted Kappa.

Score predictions are evaluated based on objective criteria, and specifically using the quadratic weighted kappa error metric, which measures the agreement between two graders.

This metric typically varies from 0 (only random agreement between graders) to (complete agreement between graders). In the event that there is less agreement between the graders than expected by chance, this metric may go below 0.

The quadratic weighted kappa is calculated between the automated scores for the responses and the resolved score for human graders on each set of responses.

The quadratic weighted kappa is computed between the automated score predicted by the proposed model and the score given by the human grader. It is 0.6062

Since, it is more than 0.5 and nearing 1, there is an agreement between the ratings.




[Back To The Top](#essay_auto_grading)

---

## Proposed Steps

#### Text pre-processing:

	Text substitution (&, /, deg, C, F…..)
	Conversion to lower case
	Removal of special characters
	Correction of misspelled words
	Dictionary words, special words (w.r.t. essay set), common words (after removal of
	special characters.)
	Word and Sentence tokenization
	
#### Feature extraction:

	Stemming / Lemmatization
	Bag of Words (counts of n-grams), TF-IDF, Word2Vec
	Choose relevant words and bigrams
	Creation of tensors of data (essay text and labels)
	Creation of Feature sets
	
#### Predictive modelling:

	Can try with Bayes Network, Random Forest, Gradient Boosting
	LSTM: (Embedding Layer, LSTM, Dense Layer)
	
#### Performance Evaluation:

	Calculation of Quadratic Weighted Kappa
	
	This error metric measures the agreement between two graders. It typically varies from 0 (only random
	agreement between graders) to 1 (complete agreement between graders). In the event that there is less
	agreement between the graders than expected by chance, this metric may go below 0.

[Back To The Top](#essay_auto_grading)


#### API Reference

## Train offline -> Make model available as a service -> Predict online

1.	Building a machine-learning model for essay auto grading.

2.	Then create an API for the model, using Flask, 

3.	The Python micro framework for building web applications.

After training the model, it is desirable to have a way to persist the model for future use without having to retrain. To achieve this, we add the following lines to save our model as a .pkl file for the later use.

And we can load and use saved model later:

The above process called “PERSIST Model in a standard format”, that is, models are persisted in a certain format specific to the language in development.

And the model will be served in a micro-service that expose endpoints to receive requests from client.



```html
    <p>dummy code</p>
```
[Back To The Top](#essay_auto_grading)

---


---

## References
[Back To The Top](#essay_auto_grading)

---

## License

MIT License

Copyright (c) [2017] [James Q Quick]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[Back To The Top](#essay_auto_grading)

---

## Author Info

- Twitter - []()
- Website - []()

[Back To The Top](#essay_auto_grading)
