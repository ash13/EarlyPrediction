
import pandas as pd
import numpy as np
import time


data=pd.read_csv('skill_builders_16_17.csv')
print("original data lenth:",len(data.index))


unique_Actions=list(data['action_name'])
unique_Actions=set(unique_Actions)
print("unique 1",unique_Actions)

data=data[~data['action_name'].isin(['start','comment','end','work','resume'])]

data_orig=data[~data['action_name'].isin(['start','comment','end','work','resume'])]
unique_Actions2=list(data['action_name'])
print(set(unique_Actions2))

start = time.time()

data['next_assignment_wheel_spin']=-1
data['next_assignment_stopout']=-1
data['used_bottom_out_hint']= 0
data['previous_3_states'] = [('null', 'null', 'null')]* len(data)

new_df=pd.DataFrame()
#Creating the feature 1 and 2 : Next_assignment_wheel_spin and Next_assignment_stopout

for studentID, student in data.groupby('user_id'): 

    student = student.sort_values(by=['action_time']) 
    student['next_assignment_wheel_spin'] = student['assignment_wheel_spin'].shift(-1)
    student['next_assignment_wheel_spin'] = student['next_assignment_wheel_spin'].fillna(-1)

    student['next_assignment_stopout'] = student['assignment_stopout'].shift(-1)
    student['next_assignment_stopout'] = student['next_assignment_stopout'].fillna(-1)

    for problemID, problem in student.groupby('problem_id'): 

        bottom_out_index = problem.index[problem['problem_bottom_hint'] == 1].tolist()
        if (len(bottom_out_index) > 0):              
            bottom_out_index = min(bottom_out_index)
            problem.loc[bottom_out_index:, 'used_bottom_out_hint'] = 1
        
        if (len(problem) > 1):    
            print (problemID)
        
        one_state_ago = problem['action_name'].shift(1)
        one_state_ago = one_state_ago.fillna('null')
        
        two_state_ago = problem['action_name'].shift(2)
        two_state_ago = two_state_ago.fillna('null')
        
        three_state_ago = problem['action_name'].shift(3)
        three_state_ago = three_state_ago.fillna('null')

        problem['previous_3_states'] = list(zip(three_state_ago, two_state_ago, one_state_ago))
        new_df=new_df.append(problem)
        

end = time.time()

print("took: ", end - start )


new_df1 = new_df

print("New Df1")
print(len(data.index), len(new_df1.index))
print(new_df1.columns)

#Creating feature 3 one hot encoding of correct
#print(new_df1['correct'])
mapping = {1:'correct', 0:'incorrect',np.NaN:'non-attempt'}
new_df1=new_df1.replace({'correct':mapping})

new_df1=pd.get_dummies(new_df1,columns=['correct'])
#print(new_df1[0:5])

print(new_df1.columns)

new_df1.to_csv('features_created_no_penultimate_hint.csv')

#Creating feature used_penultimate_hint

data=new_df1
new_df3=pd.DataFrame()
users=list(set(data['user_id']))
for i in users:
    user_df=data.loc[data['user_id']==i]
    #print(user_df['problem_id'])
    all_problems=np.unique(user_df["problem_id"].values)
    all_problems=list(all_problems)
    problem_hints = user_df[['problem_id', 'problem_hint_count','problem_total_hints']]
    problem_hints = problem_hints.drop_duplicates()
    problem_hints = problem_hints.dropna()
    problem_hints['hints_not_used']=problem_hints['problem_total_hints']-problem_hints['problem_hint_count']
    problem_ids=list(problem_hints.loc[problem_hints['hints_not_used']==1]['problem_id'])
    hint_counts=list(problem_hints.loc[problem_hints['hints_not_used']==1]['problem_total_hints'])
    
    user_df = user_df.reset_index(drop=True)
    
    #print(problem_ids)
    for j in range(0,len(all_problems)):
        #print("All problems:",all_problems[j])
        problem_df = user_df.loc[user_df['problem_id'] == all_problems[j]]
        if(all_problems[j] in problem_ids):
            problem_df=user_df.loc[user_df['problem_id']==all_problems[j]]
            problem_df['used_penultimate_hint'] = 0
            p=problem_ids.index(all_problems[j])
            count=hint_counts[p]
            
            curr_hint_count=0
            for index in range(0, len(problem_df.index)):
                if (problem_df.at[index, 'action_name'] == 'hint'):
                    curr_hint_count=curr_hint_count+1

                if (curr_hint_count>=(count-1)):
                    problem_df.at[index, 'used_penultimate_hint']=1

            new_df3=new_df3.append(problem_df)
        else:
            problem_df['used_penultimate_hint'] = 0
            new_df3 = new_df3.append(problem_df)

print("New df 3")
print(len(data_orig.index),len(new_df3.index))
print(new_df3.columns)
print(len(data_orig.index),len(new_df3.index))
new_df3.to_csv('features_created.csv')
print("DONE creating new file")


