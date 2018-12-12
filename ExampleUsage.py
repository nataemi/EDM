import EDM.teacher as tch
import EDM.student as st

clf,ac,std = tch.create_grades_tree('ExampleStudentData/Learning.xml')
df_pred = tch.predict(clf,'ExampleStudentData/Predicting.xml',"predicted")
clf2,ac,std = tch.create_fail_tree('ExampleStudentData/Learning.xml')
tch.visualize_fail_tree(clf2,"nowe")
tch.visualize_grades_tree(clf,"nowe2")
tch.statistics(df_pred,"stats")

clf,ac,std = st.create_grades_tree('ExampleStudentData/Learning.xml')
st.predict(clf,'ExampleStudentData/Predicting.xml',"predicted")
clf2,ac,std = st.create_fail_tree('ExampleStudentData/Learning.xml')
st.visualize_fail_trees(clf2,'ExampleStudentData/Predicting.xml','/Users/natalia/Desktop/StudentsFail')
st.visualize_grades_trees(clf,'ExampleStudentData/Predicting.xml','/Users/natalia/Desktop/StudentsGrades')


