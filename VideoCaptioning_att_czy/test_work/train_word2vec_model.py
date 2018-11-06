#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os
import sys
import multiprocessing
 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# /home/ytkj/czy/data/text8
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    # if len(sys.argv) < 4:
    #     # print(globals()['__doc__'] % locals())
    #     sys.exit(1)
    inp='..\\word2vec\\train_sent_ch.txt'
    outp2 = '..\\word2vec\\word_vec_video_ch.txt'
 
    model = Word2Vec(LineSentence(inp), size=512, window=3, min_count=1,
                     workers=multiprocessing.cpu_count())
 
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    #model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
