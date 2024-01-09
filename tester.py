import torch
from dataset import Dataset
import numpy as np
from measure import Measure
from os import listdir
from os.path import isfile, join

class Tester:
    def __init__(self, dataset, model_path, valid_or_test):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location = self.device)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

        self.hit1 = {"raw": 0.0, "fil": 0.0}
        self.hit3 = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr = {"raw": 0.0, "fil": 0.0}
        self.mr = {"raw": 0.0, "fil": 0.0}
    def update(self, rank, raw_or_fil):
        if rank == 1:
            self.hit1[raw_or_fil] += 1.0
        if rank <= 3:
            self.hit3[raw_or_fil] += 1.0
        if rank <= 10:
            self.hit10[raw_or_fil] += 1.0

        self.mr[raw_or_fil] += rank
        self.mrr[raw_or_fil] += (1.0 / rank)

    def normalize(self, num_facts):
        for raw_or_fil in ["raw", "fil"]:
            self.hit1[raw_or_fil] /= (2 * num_facts)
            self.hit3[raw_or_fil] /= (2 * num_facts)
            self.hit10[raw_or_fil] /= (2 * num_facts)
            self.mr[raw_or_fil] /= (2 * num_facts)
            self.mrr[raw_or_fil] /= (2 * num_facts)

    def print_(self):
        for raw_or_fil in ["raw", "fil"]:
            print(raw_or_fil.title() + " setting:")
            print("\tHit@1 =", self.hit1[raw_or_fil])
            print("\tHit@3 =", self.hit3[raw_or_fil])
            print("\tHit@10 =", self.hit10[raw_or_fil])
            print("\tMR =", self.mr[raw_or_fil])
            print("\tMRR =", self.mrr[raw_or_fil])
            print("")

    def get_rank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, head_or_tail):
        head, rel, tail = fact
        if head_or_tail == "head":
            return [(i, rel, tail) for i in range(self.dataset.num_ent())]
        elif head_or_tail == "tail":
            return [(head, rel, i) for i in range(self.dataset.num_ent())]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)

        return self.shred_facts(result)

    # def replace_and_shred(self, fact, raw_or_fil, head_or_tail):
    #     ret_facts = []
    #     head, rel, tail = fact
    #     for i in range(self.dataset.num_ent()):
    #         if head_or_tail == "head" and i != head:
    #             ret_facts.append((i, rel, tail))
    #         if head_or_tail == "tail" and i != tail:
    #             ret_facts.append((head, rel, i))

    #     if raw_or_fil == "raw":
    #         ret_facts = [tuple(fact)] + ret_facts
    #     elif raw_or_fil == "fil":
    #         ret_facts = [tuple(fact)] + list(set(ret_facts) - self.all_facts_as_set_of_tuples)

    #     return self.shred_facts(ret_facts)

    # def test(self):
    #     settings = ["raw", "fil"] if self.valid_or_test == "test" else ["fil"]
    #
    #     for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
    #         for head_or_tail in ["head", "tail"]:
    #             queries = self.create_queries(fact, head_or_tail)
    #             for raw_or_fil in settings:
    #                 h, r, t = self.add_fact_and_shred(fact, queries, raw_or_fil)
    #                 sim_scores,recon_error = self.model(h, r, t)
    #                 sim_scores.cpu().data.numpy()
    #                 rank = self.get_rank(sim_scores)
    #                 self.measure.update(rank, raw_or_fil)
    #
    #     self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
    #     self.measure.print_()
    #     return self.measure.hit10["fil"]


    def test(self, batch_size=1000):
        settings = ["raw", "fil"] if self.valid_or_test == "test" else ["fil"]

        # Split the data into batches
        for batch_data in self.batch_data_generator(self.dataset.data[self.valid_or_test], batch_size):
            for i, fact in enumerate(batch_data):
                for head_or_tail in ["head", "tail"]:
                    queries = self.create_queries(fact, head_or_tail)
                    for raw_or_fil in settings:
                        # Split queries into batches
                        for j in range(0, len(queries), batch_size):
                            batch_queries = queries[j:j + batch_size]
                            h, r, t = self.add_fact_and_shred(fact, batch_queries, raw_or_fil)
                            sim_scores, recon_error = self.model(h, r, t)
                            sim_scores.cpu().data.numpy()
                            rank = self.get_rank(sim_scores)
                            self.measure.update(rank, raw_or_fil)

        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()
        return self.measure.hit10["fil"]

    def test(self, batch_size=1000):
        settings = ["raw", "fil"] if self.valid_or_test == "test" else ["fil"]

        # Split the data into batches
        for batch_data in self.batch_data_generator(self.dataset.data[self.valid_or_test], batch_size):
            for head_or_tail in ["head", "tail"]:
                # Create queries for all facts in the batch
                queries = []
                for fact in batch_data:
                    queries.extend(self.create_queries(fact, head_or_tail))

                for raw_or_fil in settings:
                    # Split queries into batches
                    for j in range(0, len(queries), batch_size):
                        batch_queries = queries[j:j + batch_size]
                        # Process all queries in the batch
                        for fact in batch_data:
                            h, r, t = self.add_fact_and_shred(fact, batch_queries, raw_or_fil)
                            sim_scores, recon_error = self.model(h, r, t)
                            sim_scores.cpu().data.numpy()
                            rank = self.get_rank(sim_scores)
                            self.measure.update(rank, raw_or_fil)

        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()
        return self.measure.hit10["fil"]

    def batch_data_generator(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def shred_facts(self, triples):
        heads  = [triples[i][0] for i in range(len(triples))]
        rels   = [triples[i][1] for i in range(len(triples))]
        tails  = [triples[i][2] for i in range(len(triples))]
        return torch.LongTensor(heads).to(self.device), torch.LongTensor(rels).to(self.device), torch.LongTensor(tails).to(self.device)

    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data[spl]:
                tuples.append(tuple(fact))
        
        return tuples



    
    
