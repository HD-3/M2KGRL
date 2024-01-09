import numpy as np
import random
import torch
import math
import h5py
import pickle
from torchvision import transforms

class Dataset:
    def __init__(self, ds_name):
        self.name = ds_name
        self.dir = "datasets/" + ds_name + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.data = {spl: self.read(self.dir + spl + ".txt") for spl in ["train", "valid", "test"]}
        self.imageid = self.readimageid(ds_name)
        self.text = self.readtext(ds_name)
        # self.image = self.readimage(self.dir + self.name + "_ImageData.h5")
        self.batch_index = 0
       
    def read(self, file_path):
        with open(file_path, "r",encoding='utf-8') as f:
            lines = f.readlines()
        
        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids(line.strip().split()))
        return triples

    def readimageid(self, ds_name):
        imageid = {}
        if ds_name =="FB15K-IMG":
            imgpath = self.dir + "fb_vgg16_top_5_pagerank_embeddings_normalized.pkl"
            F = open(imgpath, "rb")
            content = pickle.load(F, encoding="bytes")
            for entity in self.ent2id:
                index = self.ent2id[entity]
                vgg_feat = torch.tensor(content[entity])
                imageid[index] = vgg_feat
        elif ds_name =="YAGO3-10":
            # m = torch.nn.Linear(4096, 200)
            imgpath = self.dir + "YAGO15K_ImageData.h5"
            path = self.dir + "YAGO15K_ImageIndex.txt"
            h5f = h5py.File(imgpath, 'r')
            with open(path, 'r',encoding='utf-8') as f:
                for line in f:
                    entity, imageindex = line[:-1].split("\t")
                    if entity in self.ent2id:
                        entity = self.ent2id[entity]
                        vgg_feat = h5f[imageindex]
                        vgg_feat = (torch.Tensor(vgg_feat[0])).unsqueeze(0)
                        # vgg_feat = m(vgg_feat)
                        imageid[entity] = vgg_feat
            for i in range(len(self.ent2id)):
                if (i > 0) & (i not in imageid):
                    vgg_feat = torch.rand([1, 4096])
                    # vgg_feat = m(vgg_feat)
                    imageid[i] = vgg_feat
        elif ds_name =="FB15K":
            # m = torch.nn.Linear(4096, 200)
            imgpath = self.dir + "FB15K_ImageData.h5"
            path = self.dir + "FB15K_ImageIndex.txt"
            h5f = h5py.File(imgpath, 'r')
            with open(path, 'r') as f:
                for line in f:
                    entity, imageindex = line[:-1].split("\t")
                    if entity in self.ent2id:
                        entity = self.ent2id[entity]
                        vgg_feat = h5f[imageindex]
                        vgg_feat = (torch.Tensor(vgg_feat[0])).unsqueeze(0)
                        # vgg_feat = m(vgg_feat)
                        imageid[entity] = vgg_feat
            for i in range(len(self.ent2id)):
                if (i > 0) & (i not in imageid):
                    vgg_feat = torch.rand([1, 4096])
                    # vgg_feat = m(vgg_feat)
                    imageid[i] = vgg_feat
        elif ds_name =="FB15K-237":
            # m = torch.nn.Linear(4096, 200)
            imgpath = self.dir + "FB15K_ImageData.h5"
            path = self.dir + "FB15K_ImageIndex.txt"
            h5f = h5py.File(imgpath, 'r')
            with open(path, 'r') as f:
                for line in f:
                    entity, imageindex = line[:-1].split("\t")
                    if entity in self.ent2id:
                        entity = self.ent2id[entity]
                        vgg_feat = h5f[imageindex]
                        vgg_feat = (torch.Tensor(vgg_feat[0])).unsqueeze(0)
                        # vgg_feat = m(vgg_feat)
                        imageid[entity] = vgg_feat
            for i in range(len(self.ent2id)):
                if (i > 0) & (i not in imageid):
                    vgg_feat = torch.rand([1, 4096])
                    # vgg_feat = m(vgg_feat)
                    imageid[i] = vgg_feat
        elif ds_name =="WN18":
            imgpath = self.dir + "embeddings_vgg_19_avg.pkl"
            F = open(imgpath, "rb")
            content = pickle.load(F, encoding="bytes")

            entity_new = dict()
            for e in content.keys():
                e_new = e.decode('utf-8')
                entity_new[e_new[1:]] = content[e]

            for entity in self.ent2id:
                if entity in entity_new.keys():
                    index = self.ent2id[entity]
                    vgg_feat = torch.tensor(entity_new[entity])
                    imageid[index] = vgg_feat
            print(len(imageid))

        return imageid

    def readtext(self, ds_name):
        entity2glossary = dict()
        if ds_name == "YAGO3-10":
            textpath = self.dir + "YAGO15K_description.txt"
            with open(textpath, "r") as glossf:  # TODO: add datapath
                for line in glossf:
            # print(line)
                    entity, glossary = line.split("\t")
                    entity2glossary[entity] = glossary

            entity2description = list()
            for entity, index in self.ent2id.items():
                entity2description.append(entity2glossary[entity])

        elif ds_name == "FB15K":
            textpath = self.dir + "FB15K_description.txt"
            with open(textpath, "r",encoding='utf-8') as glossf:  # TODO: add datapath
                for line in glossf:
                    # print(line)
                    entity, glossary = line.split("\t")
                    entity2glossary[entity] = glossary

            entity2description = list()
            for entity, index in self.ent2id.items():
                if entity in entity2glossary.keys():
                    entity2description.append(entity2glossary[entity])

        elif ds_name == "FB15K-237":
            textpath = self.dir + "FB15K_description.txt"
            with open(textpath, "r",encoding='utf-8') as glossf:  # TODO: add datapath
                for line in glossf:
                    # print(line)
                    entity, glossary = line.split("\t")
                    entity2glossary[entity] = glossary

            entity2description = list()
            for entity, index in self.ent2id.items():
                if entity in entity2glossary.keys():
                    entity2description.append(entity2glossary[entity])
        elif ds_name == "WN18":
            textpath = self.dir + "entity2text.txt"
            with open(textpath, "r",encoding='utf-8') as glossf:  # TODO: add datapath
                for line in glossf:
                    # print(line)
                    entity, glossary = line.split("\t")
                    entity2glossary[entity] = glossary

            entity2description = list()
            for entity, index in self.ent2id.items():
                if entity in entity2glossary.keys():
                    entity2description.append(entity2glossary[entity])

        return entity2description

    def num_ent(self):
        return len(self.ent2id)
    
    def num_rel(self):
        return len(self.rel2id)
                     
    def triple2ids(self, triple):
        return [self.get_ent_id(triple[0]), self.get_rel_id(triple[1]), self.get_ent_id(triple[2])]
                     
    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]
            
    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]
                     
    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent
                     
    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]):
            batch = self.data["train"][self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            self.batch_index = 0
        return np.append(batch, np.ones((len(batch), 1)), axis=1).astype("int") #appending the +1 label
                     
    def generate_neg(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if random.random() < 0.5:
                neg_batch[i][0] = self.rand_ent_except(neg_batch[i][0]) #flipping head
            else:
                neg_batch[i][2] = self.rand_ent_except(neg_batch[i][2]) #flipping tail
        neg_batch[:,-1] = -1
        return neg_batch

    def next_batch(self, batch_size, neg_ratio, device):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch = self.generate_neg(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        np.random.shuffle(batch)
        heads  = torch.tensor(batch[:,0]).long().to(device)
        rels   = torch.tensor(batch[:,1]).long().to(device)
        tails  = torch.tensor(batch[:,2]).long().to(device)
        labels = torch.tensor(batch[:,3]).float().to(device)
        return heads, rels, tails, labels
    
    def was_last_batch(self):
        return (self.batch_index == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))


