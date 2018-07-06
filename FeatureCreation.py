import pandas as pd
import numpy as np


data=pd.read_csv('Filtered_skill_builders_16_17.csv')

print("original data lenth:",len(data.index))

unique_Actions=list(data['action_name'])
unique_Actions=set(unique_Actions)
print("unique 1",unique_Actions)

print(len(data.index))
data=data.loc[~data['action_name'].isin(['start','comment','end','work','resume'])]
print(len(data.index))
#data.to_csv('Filtered_skill_builders_16_17.csv')


#data=data.loc[data['action_name'] not in ('start','comment','end','work','resume')]
data=data.loc[~data['action_name'].isin(['start','comment','end','work','resume'])]
data_orig=data.loc[~data['action_name'].isin(['start','comment','end','work','resume'])]
unique_Actions2=list(data['action_name'])
print(set(unique_Actions2))


print(data.columns)
print(data['action_time'])
#next_assignment_wheel_spin,next_assignment_stopout,one hot encoding of correct,bottom out hint cumaltive,used_penultimate_hint,previous_3_states

data['next_assignment_wheel_spin']=-1
users=list(set(data['user_id']))
#print(users)
new_df=pd.DataFrame()
#Creating the feature 1 : Next_assignment_wheel_spin
data=data[0:1000]
for i in users:
    user_df=data.loc[data['user_id']==i]
    user_df=user_df.sort_values(by=['action_time'])
    assignments=user_df[['assignment_id','assignment_wheel_spin']]
    #print(type(assignments))
    #print(len(assignments.index))
    assignments=assignments.drop_duplicates()
    assignments_ids=list(assignments['assignment_id'])
    assignment_wheel_spin=list(assignments['assignment_wheel_spin'])
    #print(assignments_ids)
    #print(assignment_wheel_spin)
    user_df = user_df.reset_index(drop=True)
    for index in range(0, len(user_df.index)):
        current_assignment=user_df.at[index, 'assignment_id']
        #print("Current assignment:",current_assignment)
        pos=assignments_ids.index(current_assignment)
        #print("pos=",pos)
        if(pos==(len(assignments_ids)-1)):
            #print("last assignment")
            user_df.at[index, 'next_assignment_wheel_spin']=-1
        else:
            #print(assignments_ids[pos+1])
            user_df.at[index, 'next_assignment_wheel_spin'] = assignment_wheel_spin[pos+1]
    new_df=new_df.append(user_df)
    #print("Done")
    #print(user_df['next_assignment_wheel_spin'])

#print("New Df")
#print(len(data.index),len(new_df.index))
#print(new_df.columns)


#print("NEXT ASSIGNMENT STOPOUT")
data=new_df
data['next_assignment_stopout']=-1
#Creating the feature
new_df1=pd.DataFrame()
for i in users:
    user_df=data.loc[data['user_id']==i]
    user_df = user_df.sort_values(by=['action_time'])
    assignments=user_df[['assignment_id','assignment_stopout']]
    #print(type(assignments))
    #print(len(assignments.index))
    assignments=assignments.drop_duplicates()
    assignments_ids=list(assignments['assignment_id'])
    assignment_wheel_spin=list(assignments['assignment_stopout'])
    #print(assignments_ids)
    #print(assignment_wheel_spin)
    user_df = user_df.reset_index(drop=True)

    for index in range(0, len(user_df.index)):
        current_assignment=user_df.at[index, 'assignment_id']
        #print("Current assignment:",current_assignment)
        pos=assignments_ids.index(current_assignment)
        #print("pos=",pos)
        if(pos==(len(assignments_ids)-1)):
            #print("last assignment")
            user_df.at[index, 'next_assignment_stopout']=-1
        else:
            #print(assignments_ids[pos+1])
            user_df.at[index, 'next_assignment_stopout'] = assignment_wheel_spin[pos+1]
    #print("User df:",len(user_df.index))
    new_df1=new_df1.append(user_df)
    #print("New df1 append:",len(new_df1.index))
    #print("Done")
    #print(user_df['next_assignment_stopout'])

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

#Creating bottom out hint cumaltive

#for each student grab the problem ids and for each problem id get all the rows and check if the row is for bottom out hint and update all the
#rows after it
data=new_df1
new_df2=pd.DataFrame()
for i in users:
    user_df=data.loc[data['user_id']==i]
    user_df = user_df.sort_values(by=['action_time'])
    #print(user_df['problem_id'])
    problem_ids=np.unique(user_df["problem_id"].values)
    #print("Problem_ids=",problem_ids)

    for j in range(0,len(problem_ids)):
        bottom_out_hint=0
        #print("problem:",problem_ids[j])
        problem_df=user_df.loc[user_df['problem_id']==problem_ids[j]]
        #print(problem_df['problem_id'])
        problem_df = problem_df.reset_index(drop=True)
        problem_df['used_bottom_out_hint']=0

        for index in range(0, len(problem_df.index)):
            if(bottom_out_hint==0):
                #print("Problem bottom hint:",)
                if(problem_df.at[index, 'problem_bottom_hint'] == 1):
                    problem_df.at[index, 'used_bottom_out_hint'] = 1
                    bottom_out_hint=1
            else:
                problem_df.at[index, 'used_bottom_out_hint'] = 1

        new_df2=new_df2.append(problem_df)
        #print(problem_df['used_bottom_out_hint'])

print("New df 2")
print(len(data_orig.index),len(new_df2.index))
print(new_df2.columns)



#Creating feature used_penultimate_hint

data=new_df2
new_df3=pd.DataFrame()
for i in users:
    user_df=data.loc[data['user_id']==i]
    user_df = user_df.sort_values(by=['action_time'])
    #print(user_df['problem_id'])
    all_problems=np.unique(user_df["problem_id"].values)
    all_problems=list(all_problems)
    problem_hints = user_df[['problem_id', 'problem_hint_count','problem_total_hints']]
    problem_hints = problem_hints.drop_duplicates()
    problem_hints = problem_hints.dropna()
    problem_hints['hints_not_used']=problem_hints['problem_total_hints']-problem_hints['problem_hint_count']
    problem_ids=list(problem_hints.loc[problem_hints['hints_not_used']==1]['problem_id'])
    hint_counts=list(problem_hints.loc[problem_hints['hints_not_used']==1]['problem_total_hints'])

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



data=new_df3
new_df4=pd.DataFrame()
#Create previous 3 actions
for i in users:
    user_df = data.loc[data['user_id'] == i]
    user_df = user_df.sort_values(by=['action_time'])
    user_df['previous_3_states'] = str(np.NaN)
    user_df = user_df.reset_index(drop=True)
    for index in range(0,len(user_df.index)):
        #print("index=",index)

        if(index==0):
            user_df.at[index, 'previous_3_states'] = ('null', 'null', 'null')

        elif(index==1):
            user_df.at[index, 'previous_3_states'] = ('null', 'null',user_df.at[0, 'action_name'])

        elif(index==2):
            user_df.at[index, 'previous_3_states'] = ('null', user_df.at[index-2, 'action_name'],user_df.at[index - 1, 'action_name'])
        else:
            user_df.at[index,'previous_3_states']=(user_df.at[index-3,'action_name'],user_df.at[index-2,'action_name'],user_df.at[index-1,'action_name'])


    new_df4=new_df4.append(user_df)

print("FINAL DF")
print(new_df4.columns)
print("New df 4")
print(len(data_orig.index),len(new_df4.index))
new_df4.to_csv('features_created1.csv')
#new_df1.to_csv('features_created1.csv')
print("DONE creating new file")

