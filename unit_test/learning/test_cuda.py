import torch
import numpy
from learning import cuda
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def main():
    logging.info('testing cuda functions')
    spend_time = []
    for i in range(100):
        start = time.time()
        cuda.use_cuda()
        cuda.variable(torch.randn(5, 3))
        spend_time.append(time.time()-start)
    print(numpy.mean(numpy.asarray(spend_time)))
    logging.info('testing cuda functions done')

if __name__ == '__main__':
    main()
