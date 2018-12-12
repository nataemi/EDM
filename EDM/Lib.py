import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn import tree
import pydotplus
import re
import os
import xml.etree.cElementTree as et
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def get_best_tree_depth(x,y):
    depth_score = []

    for i in range(1, 17):
        clf = tree.DecisionTreeClassifier(max_depth=i,class_weight='balanced')
        # 7-fold cross validation
        scores = cross_val_score(estimator=clf, X=x, y=y, cv=7, n_jobs=4)
        depth_score.append(scores.mean())

    #wykres doboru glebokosci drzewa
    #x = list(range(1, 17))
    #plt.plot(x,depth_score)
    #plt.xticks(x)
    #plt.ylabel('miara skuteczności')
    #plt.xlabel('głębokość drzewa')
    #plt.title('Dobór optymalnej głębokości drzewa decyzyjnego')
    #plt.show()

    depth = np.argmax(depth_score) + 1
    accuracy = depth_score[depth]

    return depth,accuracy,np.array(depth_score).std()*2

def xml_to_pd(xml_file,isForPred):
    parsed_xml = et.parse(xml_file)
    dfcols = ['index', 'school', 'sex', 'age','adress','traveltime','failures','schoolsup',
              'activities','absences','finalGrade','subject','recentAvgGradeScore']
    df_xml = pd.DataFrame(columns=dfcols)

    for node in parsed_xml.getroot():
        index = node.find('index')
        school = node.find('school')
        sex = node.find('sex')
        age = node.find('age')
        adress = node.find('adress')
        traveltime = node.find('traveltime')
        failures= node.find('failures')
        schoolsup = node.find('schoolsup')
        activities = node.find('activities')
        absences = node.find('absences')
        finalGrade = node.find('finalGrade')
        subject = node.find('subject')
        recentAvgGradeScore = node.find('recentAvgGradeScore')


        df_xml = df_xml.append(
            pd.Series([getvalueofnode(index), getvalueofnode(school), getvalueofnode(sex),
                       getvalueofnode(age),getvalueofnode(adress),getvalueofnode(traveltime),
                       getvalueofnode(failures),getvalueofnode(schoolsup),
                       getvalueofnode(activities),getvalueofnode(absences),
                       getvalueofnode(finalGrade),getvalueofnode(subject),
                       getvalueofnode(recentAvgGradeScore)], index=dfcols),
            ignore_index=True)

        if(isForPred):
            df_xml = df_xml.drop(['finalGrade'], axis=1)

    return df_xml

def getvalueofnode(node):
    return node.text if node is not None else None

def to_xml(df, filename=None, mode='w'):
    xml = []


    def row_to_xml(row):
        xml.append('<student>')
        for i, col_name in enumerate(row.index):
            xml.append('  <{0}>{1}</{0}>'.format(col_name, row.iloc[i]))
        xml.append('</student>')
        return '\n'.join(xml)

    res = '\n'.join(df.apply(row_to_xml, axis=1))

    xml.insert(0,'<data>')
    xml.append('</data>')

    str = '\n'.join(xml)
    if filename is None:
        return str
    with open(filename, mode) as f:
        f.write(str)
