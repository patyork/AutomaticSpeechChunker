__author__ = 'pat'
'''
Connectionist Temporal Classification (CTC)
  courtesy of https://github.com/shawntan/rnn-experiment
  courtesy of https://github.com/rakeshvar/rnn_ctc
implemented in Theano and optimized for use on a GPU

To be instantiated and used to find the cost for each element in a set of possible labels.
'''

import theano
import theano.tensor as T


class CTCLayer():
    def __init__(self, inpt, labels):
        '''
        Recurrent Relation:
        A matrix that specifies allowed transistions in paths.
        At any time, one could
        0) Stay at the same label (diagonal is identity)
        1) Move to the next label (first upper diagonal is identity)
        2) Skip to the next to next label if
            a) next label is blank and
            b) the next to next label is different from the current
            (Second upper diagonal is product of conditons a & b)
        '''
        n_labels = labels.shape[0]

        big_I = T.cast(T.eye(n_labels+2), 'float64')
        recurrence_relation1 = T.cast(T.eye(n_labels), 'float64') + big_I[2:,1:-1] + big_I[2:,:-2] * T.cast((T.arange(n_labels) % 2), 'float64')
        recurrence_relation = T.cast(recurrence_relation1, 'float64')

        '''
        Forward path probabilities
        '''
        pred_y = inpt[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, prev: curr * T.dot(prev, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.cast(T.eye(n_labels)[0], 'float64')]
        )

        # Final Costs
        labels_probab = T.sum(probabilities[-1, -2:])
        self.cost = -T.log(labels_probab)
        self.params = []


class Chunker:
    def __init__(self):

        input_probabilities = T.fmatrix('input_probs')
        label_seq = T.ivector('label')


        ctc = CTCLayer(input_probabilities, label_seq)


        self.cost = theano.function(
            inputs=[input_probabilities, label_seq],
            outputs=[ctc.cost],
            allow_input_downcast=True
        )