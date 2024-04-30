from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from tensorboardX import SummaryWriter

from src.utils.dataset import ProstateCancerDataset, GraphDataset
from src.utils.metrics import quadratic_weighted_kappa
from src.utils.helper import collate, train, evaluate
from src.utils.option import Options
from src.models.graph_transformer import Classifier



args = Options().parse()

torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True

if not os.path.isdir(args.model_path): 
    os.mkdir(args.model_path)

if not os.path.isdir(args.log_path): 
    os.mkdir(args.log_path)


# --------- DATASET & DATALOADER ----------
print('Preparing datasets and dataloaders...')

# Train data.
ids_train, labels_train = ProstateCancerDataset(args.train_images, args.train_csv).get_data()
dataset_train = GraphDataset(args.train_graphs, ids_train, labels_train)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, num_workers=10, 
                              collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
total_train_num = len(dataloader_train) * args.batch_size

# Validation data.
ids_val, labels_val = ProstateCancerDataset(args.val_images, args.val_csv).get_data()
dataset_val = GraphDataset(args.val_graphs, ids_val, labels_val)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, num_workers=10,
                            collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
total_val_num = len(dataloader_val) * args.batch_size
        

# --------- MODEL ----------
print('Building model...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Classifier(args.n_class)
model = nn.DataParallel(model)

if args.resume:
    print('Load model {}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,100], gamma=args.gamma)

writer = SummaryWriter(log_dir=args.log_path + args.task_name)
log_file_path = os.path.join(args.log_path,
                             '{}epoch_{}batch_{}lr_{}decay_{}gamma.pth'.format(args.n_epoch,
                                                                               args.batch_size,
                                                                               args.lr,
                                                                               args.weight_decay,
                                                                               args.gamma))
f_log = open(log_file_path, 'w')


# ---------- TRAINING ----------
print('Training model...')

best_qwk = 0.0
train_scores = []
val_scores = []
for epoch in range(args.n_epoch):
    print('-----------------------------------------------------------------------')
    print('epoch [%i/%i]' % (epoch+1, args.n_epoch))

    model.train()
    train_loss = 0.
    total = 0.
    current_lr = optimizer.param_groups[0]['lr']

    print('learning rate: %.7f, previous best score: %.4f' % (current_lr, best_qwk))

    for i, batch in enumerate(dataloader_train):
        preds, labels, loss = train(batch, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        qwk_train = quadratic_weighted_kappa(labels, preds, args.n_class)

        train_loss += loss
        total += len(labels)
        
    train_scores.append(qwk_train)
    print("[%d/%d] train loss: %.3f, train score: %.3f" % (total_train_num, total_train_num, train_loss / total, qwk_train))

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            total = 0.
            
            for i, batch in enumerate(dataloader_val):
                preds, labels, _ = evaluate(batch, model, args.graphcam)
    
                qwk_val = quadratic_weighted_kappa(labels, preds, args.n_class)

                total += len(labels)
                  
            val_scores.append(qwk_val)
            print('[%d/%d] val score: %.3f' % (total_val_num, total_val_num, qwk_val))
            

            if qwk_val > best_qwk: 
                best_qwk = qwk_val
                model_save_path = os.path.join(args.model_path, 
                                               '{}epoch_{}batch_{}lr_{}decay_{}gamma.pth'.format(args.n_epoch,
                                                                                                 args.batch_size,
                                                                                                 args.lr,
                                                                                                 args.weight_decay,
                                                                                                 args.gamma))
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                torch.save(model.state_dict(), model_save_path)

            log = ''
            log = log + 'epoch [{}/{}] ------ train_score = {:.4f}, val_score = {:.4f}'.format(epoch+1,
                                                                                               args.n_epoch,
                                                                                               qwk_train,
                                                                                               qwk_val) + "\n"
            f_log.write(log)
            f_log.flush()
            writer.add_scalars('score', {'train': qwk_train, 'eval': qwk_val}, epoch+1)
            
f_log.close()


# ---------- VISUALIZATION ----------
# Plot train and val scores vs epochs.
epochs = list(range(1, args.n_epoch + 1))

plt.figure(figsize=(50, 5))
plt.plot(epochs, train_scores, label='Train')
plt.plot(epochs, val_scores, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('QWK Score')
plt.legend()
plt.savefig('score_plot.png')