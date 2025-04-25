# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classification model that predicts whether an individual's income exceeds $50,000/year based on U.S. Census data. It uses a Random Forest Classifier trained using Scikit-learn. The model was trained using a pipeline that includes preprocessing categorical and continuous features and supports inference through a RESTful API.

## Intended Use

The model is intended to support exploratory data analysis or decision-making processes in contexts such as income estimation, marketing segmentation, or donation prediction by nonprofits. It is not suitable for making high-stakes financial decisions or determining eligibility for critical services.

## Training Data

The training data comes from the UCI Machine Learning Repository Census Income dataset. It contains demographic information such as age, workclass, education, marital status, occupation, relationship, race, sex, and native country. The target variable is income, categorized as <=50K or >50K.

## Evaluation Data

The model was evaluated on a separate test split of the dataset held out during training. Additional evaluations were performed on slices of the data across categorical feature values to assess performance disparities.

## Metrics

Precision: 0.7419
Recall: 0.6384
F1: 0.6863

## Ethical Considerations

The model may reflect biases present in the original dataset. Categorical features such as sex, race, and native country can contribute to disparate model performance across groups.

This model is for educational and experimental purposes and should not be deployed in real-world decision-making systems without further validation and bias auditing.



## Caveats and Recommendations

The model's predictions are only as good as the input data. Data quality and representation across groups can significantly impact performance. Users should analyze performance on individual slices of the data before applying the model broadly. For use in production, retraining the model with updated information is recommended.