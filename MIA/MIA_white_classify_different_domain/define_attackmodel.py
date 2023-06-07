import torch
import torch.nn as nn
import torch.nn.functional as F
class WhiteBoxAttackModel(nn.Module):
    def __init__(self, class_num, total,select):
        super(WhiteBoxAttackModel, self).__init__()
        if select == "white_have_grad":
            self.Output_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(class_num, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

            self.Loss_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

            self.Gradient_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Conv2d(1, 1, kernel_size=5, padding=2),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(total, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

            self.Label_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(class_num, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.Encoder_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            
        elif select == "white_only_grad":
            self.Gradient_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Conv2d(1, 1, kernel_size=5, padding=2),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(total, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.Encoder_Component = nn.Sequential(
                nn.ReLU(),
                nn.Linear(64, 2),
            )
        else:
            self.Output_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(class_num, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

            self.Loss_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.Label_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(class_num, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.Encoder_Component = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(192, 192),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )


    def forward(self, output, loss, gradient, label,select):               

        if select == "white_only_grad":
            Gradient_Component_result = self.Gradient_Component(gradient)
            final_result = self.Encoder_Component(Gradient_Component_result)
        elif select == "black_no_grad":
            Output_Component_result = self.Output_Component(output)
            Loss_Component_result = self.Loss_Component(loss)
            Label_Component_result = self.Label_Component(label)
            final_inputs = torch.cat((Output_Component_result, Loss_Component_result,Label_Component_result), 1)
            final_result = self.Encoder_Component(final_inputs)
        else:
            Output_Component_result = self.Output_Component(output)
            Loss_Component_result = self.Loss_Component(loss)
            Label_Component_result = self.Label_Component(label)
            Gradient_Component_result = self.Gradient_Component(gradient)
            final_inputs = torch.cat((Output_Component_result, Loss_Component_result,Gradient_Component_result,Label_Component_result), 1)
            final_result = self.Encoder_Component(final_inputs)
        
 

        return final_result
