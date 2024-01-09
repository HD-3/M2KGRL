from trainer import Trainer
from tester import Tester
from dataset import Dataset
import argparse
import time
import SimplE
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=1000, type=int, help="number of epochs")# 1000
    parser.add_argument('-lr', default=0.3, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="WN18", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=1415, type=int, help="batch size") # 1415
    parser.add_argument('-save_each', default=1,type=int, help="validate every k epochs")# 50
    parser.add_argument('-retrain_text_layer', default=False, type=bool, help="Retrain the PV-DM model.")
    # parser.add_argument('-imagepath', default="datasets/FB15K/FB15K_ImageData.h5", type=str, help="vgg for data")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset)
    autoencoder = SimplE.AutoEncoder(dataset.ent2id,dataset,retrain_text_layer=args.retrain_text_layer,hidden_dimension=args.emb_dim)

    print("~~~~ Training ~~~~")

    #trainer = Trainer(dataset, args,autoencoder)
    #trainer.train()

    print("~~~~ Select best epoch on validation set ~~~~")
    
    epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
    dataset = Dataset(args.dataset)
    
    best_hit10 = -1.0
    best_epoch = "0"
    for epoch in epochs2test:
    #for epoch in ("1200","1300","1400","1500","1600"):
        start = time.time()
        print(epoch)
        model_path = "models/" + args.dataset + "/" + epoch + ".chkpnt"
        tester = Tester(dataset, model_path, "valid")
        hit10 = tester.test()
        if hit10 > best_hit10:
            best_hit10 = hit10
            best_epoch = epoch
        print(time.time() - start)

    print("Best epoch: " + best_epoch)
    
    # best_epoch = "250"
    print("~~~~ Testing on the best epoch ~~~~")
    best_model_path = "models/" + args.dataset +"/"+ best_epoch + ".chkpnt"
    tester = Tester(dataset, best_model_path, "test")
    tester.test()
