import torch
q = 15
v = torch.zeros(2, 2, 10)
l_sub = torch.zeros(15, 10)
print(v.shape)

l = torch.zeros(2, 10)

loss = torch.zeros(v.shape[1])

for uber_i in range(v.shape[1]):
    loss_uber_i = 0
    denominator = 0
    for j in range(v.shape[0]):
        numerator = v[j, uber_i] @ l[uber_i].T


        bat_shit_denominator = 0

        for i in range(v.shape[1]):
            image_vec_list = v[j,i]
            bat_shit_denominator += torch.exp(image_vec_list @ l[i].T)

 
        q_shit_denominator = 0
        for i in range(q):
            image_vec_list = v[j,i]
            q_shit_denominator += torch.exp(image_vec_list @ l_sub[i].T)

        denominator = torch.log(bat_shit_denominator + q_shit_denominator)
        loss_uber_i += numerator - denominator
    loss[uber_i] = loss_uber_i
