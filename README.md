# TUW-NLP2024

This project uses Python 3.12 and [Poetry](https://python-poetry.org/) for dependency management.

## Setup Instructions

1. **Install Python 3.12** (if not already installed) using [pyenv](https://github.com/pyenv/pyenv):
   ```bash
   pyenv install 3.12
   pyenv local 3.12
   ```

2. **Install Poetry**:
   ```bash
   pip install pipx
   python -m pipx ensurepath
   pipx install poetry
   ```

3. **Add New Dependencies**:
   To add a new library, run:
   ```bash
   poetry add <library>
   ```

4. **Install Project Dependencies**:
   To install all required dependencies, run:
   ```bash
   poetry env use 3.12
   poetry install --no-root
   ```

5. **Install Jupyter** (if not already installed):
   If Jupyter is not installed, you can add it with:
   ```bash
   poetry add jupyter
   ```

6. **Activate the Poetry Virtual Environment**:
   To start working in the virtual environment, type:
   ```bash
   poetry shell
   ```

7. **Run the Jupyter Notebook**:
   Navigate to the notebooks directory and start Jupyter:
   ```bash
   cd notebooks
   jupyter notebook
   ```

Now you're all set to work on the project!

## Milestone 1 Report
All of the code, analysis, and evaluations done as part of milestone 1 can be found [here](/notebooks/milestone1.ipynb)

## Milestone 2 Report
The focus of this report was to implement multiple baseline solutions for our text classification task 

**Extraction of Narratives from Online News - Narrative Classification** and evaluate their performance. This report is split into two parts. The first part will be more in-depth and focus on Traditional Machine learning methods and their performance on this dataset. The second part will describe two deep learning baselines and evaluate their performance.

### Traditional machine learning methods
In this part, we will evaluate the performance of Random Forest, which usually achieves SOTA performance on Tabular datasets and a [Multinomial Naive Bayes Classifier](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html) which are suitable for classification with discrete features such as word counts. 

For the training data, we use the data files we prepared in Milestone 1 and we encode them using The bag of words method with a feature size of 1024. 

The training and test data have been split in a ratio of 8:2 with a random seed. The training and test are the same for deep learning and traditional ML methods.

Since traditional ML methods are normally faster and easier to train we have built 4 models for each of the currently available languages (two for sub narratives and two for narratives).

For the quantitative analysis, we have decided to focus on the macro f1, precision, and recall score.

An overview of the model performance can be seen below:

| language   | level        | model_name             |   f1_macro |   recal_macro |   precision_macro |
|:-----------|:-------------|:-----------------------|-----------:|--------------:|------------------:|
| PT         | SUBNARATIVES | MultinomialNB          | 0.122312   |    0.212674   |         0.0927896 |
| PT         | SUBNARATIVES | RandomForestClassifier | 0.0203125  |    0.015377   |         0.0416667 |
| PT         | NARATIVES    | MultinomialNB          | 0.325244   |    0.574495   |         0.237612  |
| PT         | NARATIVES    | RandomForestClassifier | 0.143697   |    0.116208   |         0.210498  |
| HI         | SUBNARATIVES | MultinomialNB          | 0.055754   |    0.0589038  |         0.0673611 |
| HI         | SUBNARATIVES | RandomForestClassifier | 0.00677083 |    0.00409226 |         0.0208333 |
| HI         | NARATIVES    | MultinomialNB          | 0.284848   |    0.275325   |         0.338636  |
| HI         | NARATIVES    | RandomForestClassifier | 0.0598485  |    0.0422078  |         0.121212  |
| EN         | SUBNARATIVES | MultinomialNB          | 0.0806928  |    0.122447   |         0.0911334 |
| EN         | SUBNARATIVES | RandomForestClassifier | 0.0167824  |    0.0136846  |         0.026864  |
| EN         | NARATIVES    | MultinomialNB          | 0.281192   |    0.468887   |         0.255638  |
| EN         | NARATIVES    | RandomForestClassifier | 0.0569264  |    0.0475814  |         0.121212  |
| BG         | SUBNARATIVES | MultinomialNB          | 0.0971549  |    0.171875   |         0.0771104 |
| BG         | SUBNARATIVES | RandomForestClassifier | 0          |    0          |         0         |
| BG         | NARATIVES    | MultinomialNB          | 0.274399   |    0.447452   |         0.218578  |
| BG         | NARATIVES    | RandomForestClassifier | 0.052381   |    0.0368687  |         0.113636  |

This data can be further aggregated to get a better understanding of the model's performance.

#### Aggregation by model

|Model|   f1_macro mean |   f1_macro std  |  recal_macro mean  |  recal_macro std  |   precision_macro mean |  precision_macro std |
|-|-----------------------:|----------------------:|--------------------------:|-------------------------:|------------------------------:|-----------------------------:|
|MultinomialNB|              0.1902    |             0.110775  |                 0.291507  |                0.184909  |                     0.172357  |                    0.10282   |
|RandomForrest|              0.0445899 |             0.0463668 |                 0.0345024 |                0.0374664 |                     0.0819903 |                    0.0714341 |

If we satisfy the data by model, we can see that surprisingly the best-performing model is the Multinomial Naive Bayes and not the RandomForrest model, which normally achieves SOTA performance on tabular data

#### Aggregation by language

|Language|   f1_macro mean|   f1_macro std |   recal_macro mean |   recal_macro std |   precision_macro mean |   precision_macro', 'std |
|-|-----------------------:|----------------------:|--------------------------:|-------------------------:|------------------------------:|-----------------------------:|
|PT|               0.152891 |              0.126889 |                 0.229688  |                 0.243576 |                      0.145642 |                    0.0935754 |
|EN|               0.108898 |              0.117852 |                 0.16315   |                 0.208828 |                      0.123712 |                    0.096353  |
|BG|               0.105984 |              0.11909  |                 0.164049  |                 0.202867 |                      0.102331 |                    0.0908275 |
|HI|               0.101805 |              0.124388 |                 0.0951321 |                 0.122299 |                      0.137011 |                    0.140536  |

If we aggregate the data by language, we notice some unexpected results. Before the analysis, we assumed that English would have the best results since English is the most widely supported language in NLP, but it seems that the accuracy for the Portuges data is much better. It could be because the PT dataset is more balanced, or maybe it contains a lot more 'Other' labels which are normally the most prevalent in the dataset and easier to classify.

#### Which Narrative labels are the easiest to predict
To score used to calcučate how good label predictions are is obtained by the following formula:
$$
score = \frac{1}{n}\sum_{i=0}^{n}abs(I - p)
$$
where $I$ is an indicator varia equal to 1 or 0 and $p$ is the prediction probability of the model.



| label                                                  |   mean probability error |   Number of training datapoints |   Number of test datapoints |
|:-------------------------------------------------------|-------------------------:|--------------------------------:|----------------------------:|
| Other                                                  |                 0.712814 |                              80 |                          17 |
| URW: Blaming the war on others rather than the invader |                 0.765873 |                              15 |                           3 |
| CC: Criticism of climate policies                      |                 0.783331 |                               7 |                           3 |
| URW: Discrediting the West, Diplomacy                  |                 0.785071 |                              24 |                          11 |
| URW: Speculating war outcomes                          |                 0.818252 |                              11 |                           4 |
| CC: Criticism of institutions and authorities          |                 0.824094 |                              16 |                           3 |
| URW: Russia is the Victim                              |                 0.834108 |                               7 |                           5 |
| URW: Amplifying war-related fears                      |                 0.849082 |                              21 |                           5 |
| URW: Overpraising the West                             |                 0.849702 |                               8 |                           1 |
| CC: Controversy about green technologies               |                 0.857633 |                               4 |                           1 |
| CC: Criticism of climate movement                      |                 0.859328 |                              11 |                           1 |
| CC: Hidden plots by secret schemes of powerful groups  |                 0.864364 |                               6 |                           0 |
| URW: Discrediting Ukraine                              |                 0.873203 |                              13 |                           6 |
| URW: Praise of Russia                                  |                 0.89781  |                               7 |                           3 |
| CC: Questioning the measurements and science           |                 0.916215 |                               3 |                           1 |
| CC: Downplaying climate change                         |                 0.924982 |                               1 |                           1 |
| URW: Negative Consequences for the West                |                 0.936809 |                               6 |                           1 |
| CC: Green policies are geopolitical instruments        |                 0.946143 |                               1 |                           0 |
| URW: Hidden plots by secret schemes of powerful groups |                 0.949995 |                               6 |                           2 |
| URW: Distrust towards Media                            |                 0.974954 |                               7 |                           2 |
| CC: Climate change is beneficial                       |                 0.975    |                               0 |                           1 |
| CC: Amplifying Climate Fears                           |                 1        |                               0 |                           0 |

Based on the results we can see that we struggle with the prediction of Narrative Labels that have a small amount of training examples. Based on the data it seems like the prediction score is only related to how many training data points we have.

All of the code related to the quantitative analysis, building of models, and the required data transformations can be found [here](/notebooks/milestone2-traditional_ml_methods.ipynb)

#### Qualitative analysis

IVAN ADDS HIS ANALYSIS/notebook link here


### Deep learning methods


### Division of work
- Tibor Cus (12325298) -building traditionam ML models, quantitatev analysis, report
