import tensorflow as tf


def NLL(y_true, y_dist):
    return -y_dist.log_prob(y_true)
