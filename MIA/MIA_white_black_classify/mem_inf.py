import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
np.set_printoptions(threshold=np.inf)

from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, roc_auc_score
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class attack_for_whitebox():
    def __init__(self,ATTACK_SETS,eval_dataset_train_m, eval_dataset_train_nm,eval_dataset_shadow_m, eval_dataset_shadow_nm,trainer,trainer_shadow,attack_model, num_labels,tasks,texthide):
        self.class_num = num_labels
        self.tasks = tasks
        self.trainer = trainer
        self.trainer_shadow = trainer_shadow
        self.texthide = texthide

        self.ATTACK_SETS = ATTACK_SETS
        
        self.eval_dataset_train_m = eval_dataset_train_m
        self.eval_dataset_train_nm = eval_dataset_train_nm
        self.eval_dataset_shadow_m = eval_dataset_shadow_m
        self.eval_dataset_shadow_nm = eval_dataset_shadow_nm

        self.attack_model = attack_model.cuda()
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        #self.target_criterion = nn.CrossEntropyLoss(reduction='none')
        self.attack_criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

        self.attack_train_data = None
        self.attack_test_data = None
    def prepare_dataset_train(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            if self.texthide:
                for eval_dataset_shadow_m, task in zip(self.eval_dataset_shadow_m, self.tasks):
                    outputs_n, losses_n, gradients_n, labels_n = self.trainer_shadow.evaluate(eval_dataset=eval_dataset_shadow_m,black_shadow='True',attack_type="mia_white_gradient",texthide_test=True)

                for eval_dataset_shadow_nm, task in zip(self.eval_dataset_shadow_nm, self.tasks):
                    outputs_nm, losses_nm, gradients_nm, labels_nm = self.trainer_shadow.evaluate(eval_dataset=eval_dataset_shadow_nm,black_shadow='True',attack_type="mia_white_gradient",texthide_test=True)
            
            else:
                for eval_dataset_shadow_m, task in zip(self.eval_dataset_shadow_m, self.tasks):
                    outputs_n, losses_n, gradients_n, labels_n = self.trainer_shadow.evaluate(eval_dataset=eval_dataset_shadow_m,black_shadow='True',attack_type="mia_white_gradient")

                for eval_dataset_shadow_nm, task in zip(self.eval_dataset_shadow_nm, self.tasks):
                    outputs_nm, losses_nm, gradients_nm, labels_nm = self.trainer_shadow.evaluate(eval_dataset=eval_dataset_shadow_nm,black_shadow='True',attack_type="mia_white_gradient")
            for i in range(len(outputs_n)):
                members = torch.ones(len(outputs_n[i]))
                members1 = torch.zeros(len(outputs_nm[i]))
                member = torch.cat((members,members1),dim=0)
                losses = torch.cat((losses_n[i],losses_nm[i]),dim=0)
                output = torch.cat((outputs_n[i],outputs_nm[i]),dim=0)
                gradients = torch.cat((gradients_n[i],gradients_nm[i]),dim=0)
                labels = torch.cat((labels_n[i],labels_nm[i]),dim=0)
                idx = torch.randperm(len(output))
                if len(idx)!=len(losses) or len(idx)!=len(gradients) or len(idx)!=len(labels) or len(idx)!=len(member):
                    print(".......................train error...............")
                output=output[idx,:]
                losses=losses[idx,:]
                gradients=gradients[idx,:]
                labels=labels[idx,:]
                member=member[idx]
                pickle.dump((output, losses, gradients, labels ,member), f)
        print("Finished Saving Train Dataset")
    
    def prepare_dataset_test(self):
        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            if self.texthide:
                for eval_dataset_train_m, task in zip(self.eval_dataset_train_m, self.tasks):
                    outputs_n1, losses_n1, gradients_n1, labels_n1 = self.trainer.evaluate(eval_dataset=eval_dataset_train_m,black_shadow='True',attack_type="mia_white_gradient",texthide_test=True)

                for eval_dataset_train_nm, task in zip(self.eval_dataset_train_nm, self.tasks):
                    outputs_nm1, losses_nm1, gradients_nm1, labels_nm1 = self.trainer.evaluate(eval_dataset=eval_dataset_train_nm,black_shadow='True',attack_type="mia_white_gradient",texthide_test=True)
            else:
                for eval_dataset_train_m, task in zip(self.eval_dataset_train_m, self.tasks):
                    outputs_n1, losses_n1, gradients_n1, labels_n1 = self.trainer.evaluate(eval_dataset=eval_dataset_train_m,black_shadow='True',attack_type="mia_white_gradient")

                for eval_dataset_train_nm, task in zip(self.eval_dataset_train_nm, self.tasks):
                    outputs_nm1, losses_nm1, gradients_nm1, labels_nm1 = self.trainer.evaluate(eval_dataset=eval_dataset_train_nm,black_shadow='True',attack_type="mia_white_gradient")
            for i in range(len(outputs_n1)):
                members2 = torch.ones(len(outputs_n1[i]))
                members3 = torch.zeros(len(outputs_nm1[i]))
                member1 = torch.cat((members2,members3),dim=0)
                losses1 = torch.cat((losses_n1[i],losses_nm1[i]),dim=0)
                output1 = torch.cat((outputs_n1[i],outputs_nm1[i]),dim=0)
                gradients1 = torch.cat((gradients_n1[i],gradients_nm1[i]),dim=0)
                labels1 = torch.cat((labels_n1[i],labels_nm1[i]),dim=0)
                idx = torch.randperm(len(output1))
                if len(idx)!=len(losses1) or len(idx)!=len(gradients1) or len(idx)!=len(labels1) or len(idx)!=len(member1):
                    print(".......................test error...............")
                output1=output1[idx,:]
                losses1=losses1[idx,:]
                gradients1=gradients1[idx,:]
                labels1=labels1[idx,:]
                member1=member1[idx]
                pickle.dump((output1, losses1, gradients1, labels1 ,member1), f)
        print("Finished Saving Test Dataset")

    def train(self, epoch, result_path,num,select):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []
        mem_loss = 0
        nomem_loss = 0
        #print("....................epoch...............:",epoch)
        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, loss, gradient, label, members = pickle.load(f)
                    output, loss, gradient, label, members = output.cuda(), loss.cuda(), gradient.cuda(), label.cuda(), members.cuda()
                    members1 = members.to(dtype=torch.int64)
                    '''
                    if num == 0:
                        for i in range(len(loss)):
                            if members[i] == 0:
                                nomem_loss += loss[i]
                            else:
                                mem_loss += loss[i]
                    '''
                    results = self.attack_model(output, loss, gradient, label,select)
                    #print(".......................train...............")
                    #results = F.softmax(results, dim=1)
                    losses = self.attack_criterion(results, members1)
                    #print(losses)
                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members1.size(0)
                    #print(predicted)
                    #print(members1)
                    correct += predicted.eq(members1).sum().item()
                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break
        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result


    def test(self, epoch, result_path,num,select):
        self.attack_model.eval()
        batch_idx = 1
        test_loss = 0
        correct = 0
        total = 0
        mem_loss = 0
        nomem_loss = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, loss, gradient, label, members = pickle.load(f)
                        output, loss, gradient, label, members = output.cuda(), loss.cuda(), gradient.cuda(), label.cuda(), members.cuda()
                        members1 = members.to(dtype=torch.int64)
                        '''
                        if num == 0:
                            for i in range(len(loss)):
                                if members[i] == 0:
                                    nomem_loss += loss[i]
                                else:
                                    mem_loss += loss[i]
                        '''
                        results = self.attack_model(output, loss, gradient, label,select)
                        #results = F.softmax(results, dim=1)
                        losses = self.attack_criterion(results, members1)
                        #print("................test..............")
                        #print(losses)
                        test_loss+=losses.item()
                        _, predicted = results.max(1)
                        #print(predicted)
                        #print(members1)
                        total += members1.size(0)
                        correct += predicted.eq(members1).sum().item()

                        results = F.softmax(results, dim=1)
                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break
        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)


            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total,1.*test_loss/batch_idx))


        return final_result

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)
