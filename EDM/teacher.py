from EDM.Lib import *

def create_grades_tree(xml_file_with_train_data):  # dodaj tu opcjonalne ustawienia dla drzewa decyzyjnego
    print("Creating decision tree...")

    data = xml_to_pd(xml_file_with_train_data,False)

    data = data.drop(['index'], axis=1)
    Y = data['finalGrade'].values  # zmien ta nazwe na final_grade
    df1 = data.drop(['finalGrade'], axis=1)

    x = df1.values
    max_depth, accuracy, std = get_best_tree_depth(x, Y)
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                 class_weight='balanced')
    print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy, std * 2))

    clf.fit(x, Y)

    return clf, accuracy, std


def predict(clf, xml_file_with_predict_data,filename=None):
    print("Predicting in process...")

    data_to_predict = xml_to_pd(xml_file_with_predict_data,True)
    df_with_predicted = data_to_predict
    data_to_predict = data_to_predict.drop(['index'], axis=1)
    predicted = clf.predict(data_to_predict)

    df_with_predicted['predictedFinalGrade'] = predicted
    to_xml(df_with_predicted,filename)
    return df_with_predicted


def create_fail_tree(xml_file_with_train_data):  # dodaj tu opcjonalne ustawienia dla drzewa decyzyjnego
    print("Creating decision tree...")

    data = xml_to_pd(xml_file_with_train_data,False)

    data = data.drop(['index'], axis=1)

    finalGrade_map = {'1': 0,
                      '2': 1,
                      '3': 1,
                      '4': 1,
                      '5': 1,
                      '6': 1}
    data.finalGrade = data.finalGrade.map(finalGrade_map)
    Y = data['finalGrade'].values
    df1 = data.drop(['finalGrade'], axis=1)

    x = df1.values
    max_depth, accuracy, std = get_best_tree_depth(x, Y)
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                 class_weight='balanced')
    print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy, std * 2))

    clf.fit(x, Y)

    return clf, accuracy, std


def visualize_grades_tree(clf, tree_name):
    print("Visualizing decision trees...")

    dot_data = tree.export_graphviz(clf,
                                    feature_names=['school', 'sex', 'age', 'address', 'traveltime', 'failures',
                                                   'schoolsup', 'activities', 'absences', 'Subject',
                                                   'RecentAvgGradeScore'],
                                    out_file=None,
                                    filled=True,
                                    rounded=True,
                                    class_names=['1', '2', '3', '4', '5', '6'])

    dot_data = re.sub("gini.+n", "", str(dot_data))
    dot_data = re.sub("class", "ocena", str(dot_data))

    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('brown3','darksalmon','wheat1','darkolivegreen2','darkolivegreen3','mediumseagreen')
    nodes = graph.get_node_list()
    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            values = clf.tree_.value[int(node.get_name())][0]
            node.set_fillcolor(colors[np.argmax(values)])
        else:
            node.set_fillcolor(colors[-1])

    graph.write_png(tree_name + '.png')


def visualize_fail_tree(clf, tree_name):
    print("Visualizing decision trees...")

    dot_data = tree.export_graphviz(clf,
                                    feature_names=['school', 'sex', 'age', 'address', 'traveltime', 'failures',
                                                   'schoolsup', 'activities', 'absences', 'Subject',
                                                   'RecentAvgGradeScore'],
                                    out_file=None,
                                    filled=True,
                                    rounded=True,
                                    class_names=['brak',' tak'])

    dot_data = re.sub("gini.+n", "", str(dot_data))
    dot_data = re.sub("class", "promocja", str(dot_data))
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('brown3', 'mediumseagreen')
    nodes = graph.get_node_list()

    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            values = clf.tree_.value[int(node.get_name())][0]
            node.set_fillcolor(colors[np.argmax(values)])
        else:
            node.set_fillcolor(colors[-1])

    graph.write_png(tree_name + '.png')




def statistics(df_with_predicted, fileName):
    print("Creating statistics...")
    df_with_predicted['predictedFinalGrade'] = pd.to_numeric(df_with_predicted['predictedFinalGrade'])
    with open(fileName, "w") as text_file:
        text_file.write(df_with_predicted.predictedFinalGrade.describe().to_string())


