# KAS

KAS(Keen About Skin) is a product for identifying a client's skin type, gender and flaws, so we can recommend the best treatments. We take a current image, to understand what the
current skin looks like and make predictions with our 3 deeplearning models mentioned below in the table.

 Number  | Name | Predictions | Link 
--- | --- | --- |--- 
1 | Gender Type | Male or Female | [Gender type recogntion](https://github.com/maithili-31/Project-For-Microsoft-Engage-Intern-2022/blob/main/Gender_Recognition.ipynb)
2 | Skin Type | Dry or Oily | [Skin type recognition](https://github.com/maithili-31/Project-For-Microsoft-Engage-Intern-2022/blob/main/Skin_Type_Recognition.ipynb)
3 | Skin disease type | Acne or Normal | [Skin disease type recognition](https://github.com/maithili-31/Project-For-Microsoft-Engage-Intern-2022/blob/main/Skin_Disease.ipynb)

## Topics Included

1. Introduction
2. Installation Guide
3. Overview
4. Predictions
5. Conclusion

### 1. Introduction

Your skin takes a lot of wear and tear over the years. Maybe you have sun damage, redness, or uneven skin texture. Or maybe you’re having acne and have a dry or oily skin not knowing what to do with it to improve. This product makes it easy for you to take further steps. It helps analyse your skin type, skin disease and then predict your gender. With all these predictions available it then recommends the best products available according to the conditions.

### 2. Installation Guide
*The system requires tensorflow to load deeplearning models, tensorflow-gpu errors while running parallely with flask make sure you don't have cudaNN setup or this error will be raised*

```elem
1. git clone https://github.com/maithili-31/Facial-Recognition
2. pip install -r requirements.txt
3. Setup your venv path
4. python app.py (or simply run `app.py` file)
```
The website will automatically start loading in your default browser

### 3. Overview
Once the website is loaded you can click on the camera button to get your face recognized. Once you put a good smile on your face click the predict button present below the camera capture screen. 

This will take a second to make predictions. Once the predictions are made you will be redirected to an analysis page which will give you a detailed analysis of your skin and also give you recommendations according to your skin report.

### 4. Predictions

Below are some of the predictions from the deeplearning model trained.

![IMG](https://github.com/maithili-31/Project-For-Microsoft-Engage-Intern-2022/blob/main/assets/gender_preds.png?raw=true)

![IMG](https://github.com/maithili-31/Project-For-Microsoft-Engage-Intern-2022/blob/main/assets/skin_type_preds.png?raw=true)

### 5. Conclusion
The problem of skin disease is addressed and solved using a skin analysis tool(KAS).
It also enhances Skin Health.
Important Functional Features :
1. Gender Detection.
2. Skin Type Detection.
3. Skin Diseaese Detection.
   ie. Normal or Acne.

4.Recommends Skin care products according to Skin Type and Gender.

ie.a. Male Oily skin.
   b. Male Dry skin.
   c. Female Oily skin.
   d. Female Dry skin.
   
5.Customized suggestions of food suppliments according to Skin type and Skin disease.

6.Helps choosing products suitable for the skin type and provides direct links to buy them.

7.Suggests DO'S and DON'TS to avoid skin disease and provides Home Remedies for Healthy Skin.

