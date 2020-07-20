consecutive_times_to_accept_face_rec_prediction = 3

subjects = ["", "Esteban", "Majo", "Noe", "Pau", "Melissa", "Maria Emilia", "Tony Stark", "Thor", "Ant Man",
            "Vanessa", "Hulk", "Captain America", "Black Widow", "Captain Marvel", "Maria Hill", "Scarlet Witch",
            "Ana Paula"]

face_recognition_effectiveness_dic = {}

for counter, subject in enumerate(subjects):
    face_recognition_effectiveness_dic[counter] = 0

it1 = [1, 2, 3]
it2 = [2, 5, 6]
it3 = [0]
it4 = [2, 3, 4]
it5 = [2, 7, 9]
it6 = [1, 2, 7]
it7 = [2, 5, 7]
it8 = [2, 6, 5]
it9 = [2, 4, 5]

iterations = [it1, it2, it3, it4, it5, it6, it7, it8, it9]

effectiveness_last_list = []
for iteration in iterations:  # for frame..
    reset_effectiveness_counter_elements = list(set(effectiveness_last_list) - set(iteration))
    for element in reset_effectiveness_counter_elements:
        face_recognition_effectiveness_dic[element] = 0

    for subject in iteration:
        face_recognition_effectiveness_dic[subject] += 1
        if face_recognition_effectiveness_dic[subject] >= consecutive_times_to_accept_face_rec_prediction:
            print(subject)
    effectiveness_last_list = iteration


