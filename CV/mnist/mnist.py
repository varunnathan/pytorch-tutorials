import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os, logging, argparse, sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1d = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2d = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        logging.debug('shape of x: {}\n'.format(str(x.shape)))
        logging.debug('conv1d + max_pool2d + relu\n')
        x = F.relu(F.max_pool2d(self.conv1d(x), 2))
        logging.debug('shape of x: {}\n'.format(str(x.shape)))
        logging.debug('conv2d + conv2_drop + max_pool2d + relu\n')
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2d(x)), 2))
        logging.debug('shape of x: {}\n'.format(str(x.shape)))
        logging.debug('change the shape of x to have the last dim as 320\n')
        x = x.view(-1, 320)
        logging.debug('shape of x: {}\n'.format(str(x.shape)))
        logging.debug('fc1 + relu\n')
        x = F.relu(self.fc1(x))
        logging.debug('shape of x: {}\n'.format(str(x.shape)))
        logging.debug('functional dropout\n')
        x = F.dropout(x, training=self.training)
        logging.debug('shape of x: {}\n'.format(str(x.shape)))
        logging.debug('fc2\n')
        x = self.fc2(x)
        logging.debug('shape of x: {}\n'.format(str(x.shape)))
        return F.log_softmax(x, dim=1)
    
    
def model_fn(model_dir, model_name='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = os.path.join(model_dir, model_name)
    model = nn.DataParallel(Net())
    with open(fn, 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)
    
    
def save_model(model, model_dir, model_name='model.pth'):
    logger.info("saving the model\n")
    fn = os.path.join(model_dir, model_name)
    torch.save(model.cpu().state_dict(), fn)
    
    
def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    dataset = datasets.MNIST(training_dir, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]), download=True)
    train_sampler = dist.DistributedSampler(dataset) if is_distributed else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                      sampler=train_sampler, **kwargs)


def _get_test_data_loader(batch_size, training_dir, **kwargs):
    dataset = datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]), download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
        
def _average_gradients(model):
    # gradient averaging
    size = torch.distributed.get_world_size()
    for param in model.parameters():
        torch.distributed.all_reduce(param.grad.data,
                                     op=torch.distributed.reduce_op.SUM)
        param.grad.data /= size
        
        
def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logging.info('is_distributed: {}'.format(str(is_distributed)))
    use_cuda = args.num_gpus > 0
    logger.info('number of gpus: {}'.format(str(args.num_gpus)))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_distributed:
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['HOST_RANK'] = str(host_rank)
        torch.distributed.init_process_group(backend=args.backend, rank=host_rank,
                                             world_size=world_size)
        logger.info('Initialized the distributed environment\n')
        
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir,
                                          is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir,
                                        **kwargs)
    
    logger.debug('Processes {}/{} ({:.0f}%) of train data'.format(
        len(train_loader.sampler), len(train_loader.dataset),
        100.*len(train_loader.sampler) / len(train_loader.dataset)))
    
    logger.debug('Processess {}/{} ({:.0f}%) of test data'.format(
        len(test_loader.sampler), len(test_loader.dataset),
        100.*len(test_loader.sampler) / len(test_loader.dataset)))
    
    model = Net().to(device)
    
    if is_distributed and use_cuda:
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = nn.DataParallel(model)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            
            if is_distributed and (not use_cuda):
                # multi-machine cpu case
                _average_gradients(model)
                
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        
        test(model, test_loader, device)
    
    save_model(model, args.model_dir)
    
    
def test(model, loader, device):
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            logger.debug('target shape: {}'.format(str(target.shape)))
            logger.debug('pred shape: {}'.format(str(pred.shape)))
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        msg = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        logger.info(msg.format(test_loss, correct, len(loader.dataset), accuracy))
        
        
def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=[])
    parser.add_argument('--current-host', type=str, default='')
    parser.add_argument('--model-dir', type=str, default='../model/')
    parser.add_argument('--data-dir', type=str, default='../data/')
    parser.add_argument('--num-gpus', type=int, default=0)
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    train(args)
    
        
        