import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision # for data
import ndf # for Neural Decision Forest Model

# hyper-parameters
batch_size = 128
feat_dropout = 0.5
n_tree = 5
tree_depth = 3
tree_feature_rate = 0.5
lr = 0.001
epochs = 10
report_every = 10
shallow = False

# change gpuid to use GPU
cuda = 0 
gpuid = -1
n_class = 10

# return normalized dataset divided into two sets
def prepare_db():
    train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ]))

    eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ]))
    return {'train':train_dataset,'eval':eval_dataset}        

def prepare_model():
    feat_layer = ndf.FeatureLayer(feat_dropout, shallow)
    forest = ndf.Forest(n_tree=n_tree,tree_depth=tree_depth,n_in_feature=feat_layer.get_out_feature_size(),
                        tree_feature_rate=tree_feature_rate,n_class=n_class)
    model = ndf.NeuralDecisionForest(feat_layer,forest)

    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model

def prepare_optim(model):
    params = [ p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

def train(model,optim,db):

    for epoch in range(1, epochs + 1):

        train_loss = float(0) # change

        # Update \Theta and \Pi
        model.train()
        train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optim.zero_grad()
            output = model(data)
            loss = F.nll_loss(torch.log(output),target)
            loss.backward()
            #torch.nn.utils.clip_grad_norm([ p for p in model.parameters() if p.requires_grad],
            #                              max_norm=5)
            optim.step()
            if batch_idx % report_every == 0:
                train_loss = loss.item() # change
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # Eval
        model.eval()
        test_loss = 0
        correct = 0
        test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=batch_size, shuffle=True)
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                data= Variable(data)
            target = Variable(target)
            output = model(data)
            test_loss += F.nll_loss(torch.log(output), target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        accuracy = float(correct) / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))

        # for plotting (training)   
        with open("plot_data_train_norm.csv", 'a+') as file:
            file.write(str(epoch) + " " +  str(train_loss) + "\n")

        # for plotting (testing)
        with open("plot_data_eval_norm.csv", 'a+') as file:
            file.write(str(epoch) + " " + str(test_loss) + " " + str(accuracy) + "\n")





def main():

    # GPU
    cuda = gpuid>=0
    if gpuid>=0:
        torch.cuda.set_device(gpuid)
    else:
        print("WARNING: RUN WITHOUT GPU")

    db = prepare_db()
    model = prepare_model()
    optim = prepare_optim(model)
    train(model,optim,db)


if __name__ == '__main__':
    main()