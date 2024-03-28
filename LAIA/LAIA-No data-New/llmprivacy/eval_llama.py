import json
import pdb
predict = []
with open('predicted_synthethic_llama7b.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        predict.append(data)
age,age_all = 0,0
gender,gender_all = 0,0
education,education_all = 0,0
married,married_all = 0,0
occupation,occupation_all = 0,0
location,location_all = 0,0
pobp,pobp_all = 0,0
income,income_all = 0,0

for elem in predict:
    for key in elem["reviews"]["synth"].keys():
        if key == "age":
            continue
        elif key == "gender":
            tg = elem["reviews"]["synth"]["gender"]["estimate"].strip().lower()
            pre = elem["predictions"]["/root/dataln0/anda/models/Llama-2-7b-chat-hf"]["full_answer"].strip().lower().split("guess")[-1]
            if tg in pre:
                gender = gender + 1
                gender_all = gender_all + 1
            else:
                gender_all = gender_all + 1
        elif key == "education":
            continue
        elif key == "married":
            tg = elem["reviews"]["synth"]["married"]["estimate"].strip().lower()
            pre = elem["predictions"]["/root/dataln0/anda/models/Llama-2-7b-chat-hf"]["full_answer"].strip().lower().split("guess")[-1]
            if tg in pre:
                married = married + 1
                married_all = married_all + 1
            else:
                married_all = married_all + 1
        elif key == "occupation":
            tg = elem["reviews"]["synth"]["occupation"]["estimate"].strip().lower()
            pre = elem["predictions"]["/root/dataln0/anda/models/Llama-2-7b-chat-hf"]["full_answer"].strip().lower()
            if tg in pre:
                occupation = occupation + 1
                occupation_all = occupation_all + 1
            else:
                occupation_all = occupation_all + 1
        elif key == "location":
            tg = elem["reviews"]["synth"]["location"]["estimate"].strip().lower()
            pre = elem["predictions"]["/root/dataln0/anda/models/Llama-2-7b-chat-hf"]["full_answer"].strip().lower()
            if tg in pre:
                location = location + 1
                location_all = location_all + 1
            else:
                location_all = location_all + 1
        elif key == "pobp":
            tg = elem["reviews"]["synth"]["pobp"]["estimate"].strip().lower().split(', ')[1]
            pre = elem["predictions"]["/root/dataln0/anda/models/Llama-2-7b-chat-hf"]["full_answer"].strip().lower().split("guess")[1]
            if tg in pre:
                pobp = pobp + 1
                pobp_all = pobp_all + 1
            else:
                pobp_all = pobp_all + 1
        elif key == "income":
            tg = elem["reviews"]["synth"]["income"]["estimate"].strip().lower()
            pre = elem["predictions"]["/root/dataln0/anda/models/Llama-2-7b-chat-hf"]["full_answer"].strip().lower().split("guess")[-1]
            if tg in pre:
                income = income + 1
                income_all = income_all + 1
            else:
                income_all = income_all + 1
        else:
            continue
print(".............income...................:",income,income_all,income/income_all)
print(".............gender...................:",gender,gender_all,gender/gender_all)
print(".............married...................:",married,married_all,married/married_all)
print(".............location...................:",location,location_all,location/location_all)
print(".............pobp...................:",pobp,pobp_all,pobp/pobp_all)
print(".............occupation...................:",occupation,occupation_all,occupation/occupation_all)




