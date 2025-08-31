### Spam Detection Model


### Project Description

This project demonstrates the development of a machine learning model to accurately classify SMS messages as either **"ham"** (legitimate) or **"spam"**. Using a public dataset of over 5,000 messages, the primary challenge involved building a highly performant model despite a significant class imbalance. The project showcases a full data science workflow, including data preprocessing, feature engineering, and rigorous performance evaluation.

-----

### Key Features

  * **Data Preprocessing & NLP:** Performed essential text cleaning tasks, including tokenization, stop word removal, and stemming, to prepare raw message data for modeling.
  * **Feature Engineering:** Utilized the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization technique to convert text data into a numerical format suitable for machine learning algorithms.
  * **Model Building:** Trained and fine-tuned a **Logistic Regression** classifier, a robust and interpretable model for text classification.
  * **Performance Evaluation:** Analyzed model performance using key metrics like **accuracy**, **precision**, and **recall** to understand its strengths and weaknesses with an imbalanced dataset.
  * **Final Model Performance:** Achieved an accuracy of over **96%**, with high precision for spam detection and a perfect recall for ham messages.

-----

### Final Results

The model achieved an impressive accuracy score of **96%**. While the overall performance is high, a detailed classification report revealed key insights into its strengths and a minor weakness:

  * **Ham Classification (Legitimate Messages):** The model demonstrated exceptional performance, achieving a **recall of 1.00**. This means it successfully identified every single legitimate message in the test set, preventing any from being incorrectly filtered as spam.
  * **Spam Classification:** The model achieved a **precision of 0.98**, indicating that when it flags a message as spam, it is highly likely to be correct. However, with a **recall of 0.70**, it missed approximately 30% of the actual spam messages. This trade-off prioritizes not incorrectly filtering legitimate messages, a crucial feature for a spam filter.

-----

### Data

The dataset used for this project is the "SMS Spam Detection Dataset" dataset, publicly available on Kaggle.

  * **Dataset Link:** [https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset](https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset)

-----

### Project Technologies

  * **Programming Language:** Python
  * **Libraries:** `pandas`, `scikit-learn`, `matplotlib`, `nltk`, `numpy`
  * **Environment:** Jupyter Notebook

-----

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your-GitHub-Repo-Link]
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd [Your-Project-Folder]
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

-----

### How to Use

1.  **Download the dataset** from the link provided above and place the `spam_sms.csv` file in a `data/` folder within your project directory.
2.  **Open the Jupyter Notebook:**
    ```
    jupyter notebook
    ```
3.  **Run the cells:** Open the `spam_detection_model.ipynb` notebook and run each cell sequentially. The notebook is structured to guide you through the data processing, model training, and evaluation steps.
-----

### Project Structure

```
Spam-Detection-Model/
├── data/
│   └── spam_sms.csv
├── spam_detection_model.ipynb
├── requirements.txt
└── README.md
```