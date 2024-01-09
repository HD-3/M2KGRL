import torch
import torch.nn as nn
import math
import numpy as np
import h5py
import pickle
import gensim
import dataset
import transformers



class IMG_Encoder(nn.Module):
    def __init__(self, dim, imageid):
        super(IMG_Encoder, self).__init__()
        self.imageid = imageid
        self.activation = nn.Sigmoid()
        self.entity_count = len(imageid)
        self.dim = dim

        self.embedding_dim = 4096
        self.criterion = nn.MSELoss(reduction='mean')
        self.raw_embedding = nn.Embedding(self.entity_count, self.dim)

        self.visual_embedding = self._init_embedding(imageid)

        self.encoder = nn.Sequential(
            torch.nn.Linear(self.embedding_dim, 1024),
            self.activation
        )

        self.encoder2 = nn.Sequential(
            torch.nn.Linear(1024, self.dim),
            self.activation
        )

        self.decoder2 = nn.Sequential(
            torch.nn.Linear(self.dim, 1024),
            self.activation
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(1024, self.embedding_dim),
            self.activation
        )

    def _init_embedding(self, imageid):
        self.ent_embeddings = nn.Embedding(self.entity_count, self.embedding_dim)
        for param in self.ent_embeddings.parameters():
            param.requires_grad = False
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        weights = torch.empty(self.entity_count, self.embedding_dim)
        embed = 0
        for index, value in imageid.items():
            try:
                embed = embed + 1
                em = value
                weights[index] = em
            except:
                # print(index, entity)
                weights[index] = self.ent_embeddings(torch.LongTensor([index])).clone().detach()
                continue
        print(embed)
        entities_emb = nn.Embedding.from_pretrained(weights)

        return entities_emb

    def forward(self, entity_id):
        v1 = self.visual_embedding(entity_id)
        v2 = self.encoder(v1)

        v2_ = self.encoder2(v2)
        v3_ = self.decoder2(v2_)

        v3 = self.decoder(v3_)
        loss = self.criterion(v1, v3)
        return v2_, loss


def pvdm(entity2id,text,retrain=True, vector_size=200, min_count=2, epochs=40):
    if not retrain:
        with open("embedding_weights/textembed_" + str(len(entity2id)) + "_" + str(vector_size) + "_" + str(
                min_count) + "_" + str(epochs) + ".pkl", "rb") as emf:
            inferred_vector_list = pickle.load(emf)
        return inferred_vector_list

    # entity2glossary = dict()  # 用于存储实体和对应的描述文本
    # with open("datasets/name/name + "_decsription.txt", "r") as glossf:  # TODO: add datapath
    #     for line in glossf:
    #         # print(line)
    #         entity, glossary = line.split("\t")
    #         entity2glossary[entity] = glossary

    entity2description = text # 用于存储所有实体的描述文本
    # Was Doing training on whole dataset, should not do it, should be done only on training dataset
    # for entity, index in entity2id.items():
    #     entity2description.append(entity2glossary[entity])

    def read_corpus(tokens_only=False):  # 是一个生成器，用于生成训练语料
        for i, v in enumerate(entity2description):
            tokens = gensim.utils.simple_preprocess(v)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    train_corpus = list(read_corpus())

    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    inferred_vector_list = list()

    for doc_id in range(
            len(train_corpus)):  # train_corpus is already sorted in entity2id order, so will be the saved vectors
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        # print(inferred_vector)     # inferred_vector is of size embedding_dim
        inferred_vector_list.append(inferred_vector)

    with open("embedding_weights/textembed_" + str(len(entity2id)) + "_" + str(vector_size) + "_" + str(
            min_count) + "_" + str(epochs) + ".pkl", "wb+") as emf:
        pickle.dump(inferred_vector_list, emf)

    return inferred_vector_list

def bert(entity2id, text, retrain=True, vector_size=768):
    if not retrain:
        with open("embedding_weights/bertembed_" + str(len(entity2id)) + "_" + str(vector_size) + ".pkl", "rb") as emf:
            bert_embeddings = pickle.load(emf)
        return bert_embeddings

    # Load a pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'


    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name)

    entity2description = text  # Use your list of text descriptions here

    # Tokenize and convert the text to BERT embeddings
    bert_embeddings = []
    for description in entity2description:
        inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pooled_output = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        bert_embeddings.append(pooled_output)

    with open("embedding_weights/bertembed_" + str(len(entity2id)) + "_" + str(vector_size) + ".pkl", "wb+") as emf:
        pickle.dump(bert_embeddings, emf)

    return bert_embeddings



class AutoEncoder(torch.nn.Module):
    # nn.Embedding(num_embeddings=self.entity_count + 1 !!, embedding_dim=self.dim, padding_idx=self.entity_count !!)
    def __init__(self, entity2id, dataset, text_embedding_dim=768, visual_embedding_dim=4096,
                 hidden_text_dim=50, hidden_visual_dim=512, hidden_dimension=50, activation='sigmoid',
                 retrain_text_layer=False):
        """
        In the constructor we instantiate the modules and assign them as
        member variables.
        """
        super(AutoEncoder, self).__init__()

        self.entity2id = entity2id
        self.activation = nn.Sigmoid()
        self.entity_count = len(entity2id)
        self.dim = hidden_dimension
        self.text_embedding_dim = text_embedding_dim
        self.visual_embedding_dim = visual_embedding_dim
        self.criterion = nn.MSELoss(reduction='mean')  # L2 loss

        self.retrain_text_layer = retrain_text_layer

        # self.imageid = dataset.imageid
        # self.dir = dataset.name
        self.dataset = dataset

        # output from following two layers are v_1's
        # input layer
        ## self.text_embedding = nn.Embedding(No_of_entities, text_embedding_dim)
        ## self.visual_embedding = nn.Embedding(No_of_entities, visual_embedding_dim)
        self.text_embedding = self._init_text_emb()
        self.visual_embedding = self._init_visual_emb(self.dataset.imageid)

        # hidden layer 1
        self.encoder_text_linear1 = nn.Sequential(
            torch.nn.Linear(text_embedding_dim, hidden_text_dim),
            self.activation
        )
        self.encoder_visual_linear1 = nn.Sequential(
            torch.nn.Linear(visual_embedding_dim, hidden_visual_dim),
            self.activation
        )

        # hidden layer 2
        # hidden_dimension is same dimension as the relation embedding dim, which is just dim in TransE model
        self.encoder_combined_linear = nn.Sequential(
            torch.nn.Linear(hidden_text_dim + hidden_visual_dim, hidden_dimension),
            self.activation
        )

        # hidden layer 3
        # each shares the same dimension/ have the same output dimension with the corresponding hidden
        # layer (hidden layer 1) in the encoder part
        self.decoder_text_linear1 = nn.Sequential(
            torch.nn.Linear(hidden_dimension, hidden_text_dim),
            self.activation
        )
        self.decoder_visual_linear1 = nn.Sequential(
            torch.nn.Linear(hidden_dimension, hidden_visual_dim),
            self.activation
        )

        # output layer
        # the output layer and input layer are of the same dimension for each modality
        # output from following two layers are v_5's
        self.decoder_text_linear2 = nn.Sequential(
            torch.nn.Linear(hidden_text_dim, text_embedding_dim),
            self.activation
        )
        self.decoder_visual_linear2 = nn.Sequential(
            torch.nn.Linear(hidden_visual_dim, visual_embedding_dim),
            self.activation
        )

    def _init_text_emb(self):
        #inferred_vector_list = pvdm(self.entity2id, self.dataset.text,retrain=self.retrain_text_layer)  # should just train on the ones in training set

        inferred_vector_list = bert(self.entity2id, self.dataset.text,retrain=self.retrain_text_layer)

        weights = np.zeros((self.entity_count,
                            self.text_embedding_dim))
        emb_num = 0# +1 to account for padding/OOKB, initialized to 0 each one
        for index in range(len(inferred_vector_list)):
            weights[index] = inferred_vector_list[index]
            emb_num = emb_num + 1
        weights = torch.from_numpy(weights)
        # print(weights.shape)
        text_emb = nn.Embedding.from_pretrained(weights)
        del inferred_vector_list
        print(emb_num)
        return text_emb

    # DONE: TODO: initialize visual embedding layers
    # def _init_visual_emb(self):
    #     uniform_range = 6 / np.sqrt(self.dim)
    #     weights = torch.empty(self.entity_count + 1, self.visual_embedding_dim)
    #     nn.init.uniform_(weights, -uniform_range, uniform_range)
    #     # np.random.uniform(low=-uniform_range, high=uniform_range, size=(self.entity_count + 1, self.visual_embedding_dim))
    #     no_embed = 0
    #     for index, entity in enumerate(self.entity2id):
    #         # print(index, entity)
    #         try:
    #             # print("../data/Dataset/" + entity + "/avg_embedding.pkl")
    #             with open("../data/Dataset/" + entity + "/avg_embedding.pkl", "rb") as visef:
    #                 em = pickle.load(visef)
    #                 weights[index] = em
    #         except:
    #             # print(index, entity)
    #             no_embed = no_embed + 1
    #             continue
    #     print(no_embed)
    #     # weights = torch.from_numpy(weights)
    #     entities_emb = nn.Embedding.from_pretrained(weights, padding_idx=self.entity_count)
    #     # nn.Embedding(num_embeddings=self.entity_count + 1,
    #     #                            embedding_dim=self.visual_embedding_dim,
    #     #                            padding_idx=self.entity_count)
    #     # uniform_range = 6 / np.sqrt(self.dim)         # Equn 16 in the Xavier initialization paper from TransE paper
    #     # entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
    #     return entities_emb
    def _init_visual_emb(self, imageid):
        self.ent_embeddings = nn.Embedding(self.entity_count, self.visual_embedding_dim)
        for param in self.ent_embeddings.parameters():
            param.requires_grad = False
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        weights = torch.empty(self.entity_count, self.visual_embedding_dim)
        embed = 0
        for index, value in imageid.items():
            try:
                embed = embed + 1
                em = value
                weights[index] = em
            except:
                # print(index, entity)
                weights[index] = self.ent_embeddings(torch.LongTensor([index])).clone().detach()
                continue
        print(embed)
        entities_emb = nn.Embedding.from_pretrained(weights)

        del imageid

        return entities_emb

    # can pass entity_id or entity_name(in that case has to do a lookup in forward)
    def forward(self, entity_id_tensors):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # For batch training a group of entity_id will be passed
        v1_t = self.text_embedding(entity_id_tensors).float()
        v1_i = self.visual_embedding(entity_id_tensors)

        v2_t = self.encoder_text_linear1(v1_t)
        v2_i = self.encoder_visual_linear1(v1_i)

        v3 = self.encoder_combined_linear(torch.cat((v2_t, v2_i), 1))
        ### print(v3.size())
        ### print(torch.cat((v2_t, v2_i), 1).size())

        v4_t = self.decoder_text_linear1(v3)
        v4_i = self.decoder_visual_linear1(v3)

        v5_t = self.decoder_text_linear2(v4_t)
        v5_i = self.decoder_visual_linear2(v4_i)

        ### print(v5_t.size())
        ### print(v5_i.size())

        # should happen for each entity call, either positive or negative sample, does not matter, so has to be done here
        # can use the following only when loss has mean/sum as 'reduction' defined
        recon_error = self.criterion(v1_t, v5_t) + self.criterion(v1_i, v5_i)
        ## or, the following
        ## a = torch.cat((v1_t, v1_i), dim=1)
        ## b = torch.cat((v5_t, v5_i), dim=1)
        ## recon_error = self.criterion(a, b)
        ## print(recon_error)

        return v3, recon_error

class SimplE(nn.Module):
    def __init__(self, num_ent, num_rel,autoencoder, emb_dim, device):
        super(SimplE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.autoencoder = autoencoder.to(device)

        # self.imageid = imageid
        # self.a = 0.3
        # self.img_dim = 4096

        # self.imagepath = imagepath

        self.ent_h_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        #self.ent_t_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs     = nn.Embedding(num_rel, emb_dim).to(device)
        self.rel_inv_embs = nn.Embedding(num_rel, emb_dim).to(device)
        # self.ent_h_image  = nn.Embedding(num_ent, emb_dim).to(device)
        # self.ent_t_image  = nn.Embedding(num_ent, emb_dim).to(device)
        # self.linear = nn.Linear(4096, num_ent).to(device)
        # self.headlinear = nn.Linear(4096, emb_dim).to(device)
        # self.taillinear = nn.Linear(4096, emb_dim).to(device)
        # self.image_embs = IMG_Encoder(dim = self.emb_dim, imageid=self.imageid).to(device)

        # self.image_embs = self._init_embedding(imageid).to(device)
        # self.imglinear = nn.Sequential(
        #     torch.nn.Linear(self.img_dim, emb_dim),
        #     nn.Sigmoid()
        # ).to(device)

        # self.imglinear1 = nn.Sequential(
            # torch.nn.Linear(512, emb_dim),
            # nn.Sigmoid()
        # ).to(device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        # nn.init.uniform_(self.headlinear.weight.data,-sqrt_size, sqrt_size)
        # nn.init.uniform_(self.taillinear.weight.data,-sqrt_size, sqrt_size)
        # weights = np.zeros((self.num_ent + 1, emb_dim))
        # for index in range(len(imageid)):
            # weights[index] = imageid[index].detach().numpy()
        # weights = torch.from_numpy(weights)
        # print(weights.shape)
        # self.ent_h_image = nn.Embedding.from_pretrained(weights)
        # self.ent_t_image = nn.Embedding.from_pretrained(weights)
        # nn.init.uniform_(self.imglinear.weight, -sqrt_size,sqrt_size)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        #nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    # def _init_embedding(self, imageid):
    #     self.ent_embeddings = nn.Embedding(self.num_ent, self.img_dim)
    #     for param in self.ent_embeddings.parameters():
    #         param.requires_grad = False
    #     nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
    #     weights = torch.empty(self.num_ent, self.img_dim)
    #     embed = 0
    #     for index, value in imageid.items():
    #         try:
    #             embed = embed + 1
    #             em = value
    #             weights[index] = em
    #         except:
    #             # print(index, entity)
    #             weights[index] = self.ent_embeddings(torch.LongTensor([index])).clone().detach()
    #             continue
    #     print(embed)
    #     entities_emb = nn.Embedding.from_pretrained(weights)
    #
    #     return entities_emb

    def l2_loss(self):
        simloss = ((torch.norm(self.ent_h_embs.weight, p=2) ** 2) + \
                   (torch.norm(self.rel_embs.weight, p=2) ** 2) + \
                   (torch.norm(self.rel_inv_embs.weight, p=2) ** 2)) / 2
        # simloss = ((torch.norm(self.ent_h_embs.weight, p=2) ** 2) +\
        #            (torch.norm(self.rel_embs.weight, p=2) ** 2) + \
        #            (torch.norm(self.rel_inv_embs.weight, p=2) ** 2) + \
        #            (torch.norm(self.ent_t_embs.weight, p=2) ** 2)) / 2
                   # (torch.norm(self.imglinear1[0].weight.data, p=2) ** 2) + \
                   # (torch.norm(self.imglinear1[0].bias.data, p=2) ** 2) )/2
        # (torch.norm(self.ent_t_embs.weight, p=2) ** 2) +
        # (torch.norm(self.imglinear[0].weight.data, p=2) ** 2) + \
        # (torch.norm(self.imglinear[0].bias.data, p=2) ** 2))/2

        # h_image, loss_h = self.image_embs(h)
        # t_image, loss_t = self.image_embs(t)
        # loss = loss_h+loss_t
        # regul = ((torch.norm(self.image_embs.encoder[0].weight.data, p=2) ** 2) + (torch.norm(self.image_embs.encoder2[0].weight.data, p=2) ** 2) +(torch.norm(self.image_embs.decoder2[0].weight.data, p=2) ** 2) +(torch.norm(self.image_embs.decoder[0].weight.data, p=2) ** 2) +(torch.norm(self.image_embs.encoder[0].bias.data, p=2) ** 2) + (torch.norm(self.image_embs.encoder2[0].bias.data, p=2) ** 2) +(torch.norm(self.image_embs.decoder2[0].bias.data, p=2) ** 2) +(torch.norm(self.image_embs.decoder[0].bias.data, p=2) ** 2) ) / 2
        return simloss

    def forward(self, heads, rels, tails):


        self.autoencoder.encoder_combined_linear[0].weight.data.div_(
            self.autoencoder.encoder_combined_linear[0].weight.data.norm(p=2, dim=1, keepdim=True))

        #a=0.05
        hh_embs = self.ent_h_embs(heads)
        #ht_embs = self.ent_h_embs(tails)
        #th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_h_embs(tails)
        #tt_embs = self.ent_h_embs(tails)
        h_unite,h_recon_error = self.autoencoder(heads)
        t_unite,t_recon_error = self.autoencoder(tails)

        #weight_s = 0.8
        #weight_m = 0.5

        #hh_embs = hh_embs * weight_s + h_unite * weight_m
        #th_embs = th_embs * weight_s + h_unite * weight_m
        #tt_embs = tt_embs * weight_s + t_unite * weight_m
        #ht_embs = ht_embs * weight_s + t_unite * weight_m



        # h_image = self.image_embs(heads)
        # t_image = self.image_embs(tails)
        # h_image = self.imglinear(h_image)
        # t_image = self.imglinear(t_image)


        # h_image = self.imglinear1(h_image)
        # t_image = self.imglinear1(t_image)


        # ht_embs = h_image
        # th_embs = t_image


        # hh_embs = (1 - a) * hh_embs + a * h_image
        # ht_embs = (1 - a) * ht_embs + a * h_image
        # th_embs = (1 - a) * th_embs + a * t_image
        # tt_embs = (1 - a) * tt_embs + a * t_image

        r_embs = self.rel_embs(rels)
        r_inv_embs = self.rel_inv_embs(rels)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(t_unite * r_inv_embs * h_unite, dim=1)
        score = (scores1+scores2)/2
        # loss =(loss_h+loss_t)/2

        return torch.clamp(score, -20, 20) ,h_recon_error+t_recon_error



