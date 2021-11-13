# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:12:56 2021

@author: François-Xavier
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:49:23 2021

@author: François-Xavier
"""

from flask import Flask,render_template,url_for,request
import re
from bs4 import BeautifulSoup 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer  
import joblib

# sauvegarde des modeles dans le repertoire modele
log_regres_t = joblib.load("LogisticRegression.joblib")
random_forest_t = joblib.load("RandomForest.joblib")
vecto_t = joblib.load("Vectoriser.joblib")
multi_bi_t = joblib.load("MultiLabelisator.joblib")



def text_cleaning_stemming_quest(teste_brute) :
    # suppression des  balises HTML
    text_html_off = BeautifulSoup(teste_brute).get_text() 
    
    # prise en compte des versiosn 
    #text_version =  re.sub(r'\b\d+\b', " ",  text_html_off)   # peut etre enlever cette histoire de version
    # suprresion des nombres
    text_nonum = re.sub(r'\d+', '',  text_html_off)
    # tokénisation
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    text_token =tokenizer.tokenize(text_nonum)
    
    # minisculisation
    text_miniscule  = [word.lower() for word in  text_token]    
    
    # suppression des séparateurs 
    stoppeurs = set(stopwords.words("english")) 
    text_stopper_off = [word for word in text_miniscule if word not in stoppeurs]
    
    # sterminisation
    ps = PorterStemmer()
    text_stem = [ps.stem(word) for word in text_stopper_off]
    #( " ".join(text_stem))
    return text_stem 



def text_to_numeric_transform(raw_text) :
    vecto_t = joblib.load("Vectoriser.joblib")
    stem_text = text_cleaning_stemming_quest(raw_text)
    num_text = vecto_t.transform(stem_text)
    x_text = num_text.todense()
    return x_text



def tag_predictor(model,text_num) :
    pred_model = model.predict(text_num)
    tag_list = multi_bi_t.inverse_transform(pred_model)
    return tag_list



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home_tag.html')

@app.route('/predict',methods=['POST'])


# log_regres_t = joblib.load("LogisticRegression.joblib")
# random_forest_t = joblib.load("RandomForest.joblib")
# vecto_t = joblib.load("Vectoriser.joblib")
# multi_bi_t = joblib.load("MultiLabelisator.joblib")



def predict() :
    
        # log_regres_t = joblib.load("LogisticRegression.joblib")
        # random_forest_t = joblib.load("RandomForest.joblib")
        # multi_bi_t = joblib.load("MultiLabelisator.joblib")
       #
        #return (list_tag_1,list_tag_2)
        
        if request.method == 'POST':
            question_input = request.form['message']
            x_text = text_to_numeric_transform(question_input)
            tag_list_rf = tag_predictor(random_forest_t,x_text)
            tag_list_logit = tag_predictor(log_regres_t,x_text)
            list_precis = list()
            list_recal = list()
            list_tag = list()
            list_tag_1 = list()
            list_tag_2 = list()
            for tag_t in tag_list_logit :
                list_precis.extend(tag_t)
            for tag_t in tag_list_rf :
                list_recal.extend(tag_t)
            for tag_t in list_precis :
                if tag_t in list_recal :
                    list_tag_1.append(tag_t)
                    list_recal.remove(tag_t)       
                else :
                    list_tag_2.append(tag_t)
            list_tag_2.extend(list_recal)
            list_tag_1 = list(set(list_tag_1))
            list_tag_2 = list(set(list_tag_2))
            tag_p = list_tag_1 
            tag_c = list_tag_2
            
            

        return render_template('result_tag.html',tag_precis = tag_p,tag_correles = tag_c)







if __name__ == '__main__':
    app.run(debug=True)

